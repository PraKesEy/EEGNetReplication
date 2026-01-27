"""Model definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eegnet_repl.logger import logger

import numpy as np
import torch
import torch.nn as nn

#@dataclass(frozen=True)
# Define model
class EEGNet(nn.Module):
    def __init__(self, C, T, F1=8, D=2, p=0.5):
        # C = number of channels, input.shape[1], C = 22 based on 02_preprocessing_pipeline
        # T = number of timepoints per batch, input.shape[2], T = 257 based on 02_preprocessing_pipeline
        # F1 = temporal filters
        # D = spatial filters
        # p = dropout probability: 0.5 for within-subject classification, 0.25 for cross-subject classification

        super().__init__()
        F2 = F1*D # pointwise convolutions, can be any number but authors choose this
        self.temporal = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1,32),
                padding='same', # keeps initial dims per channel
                groups=1, # default
                bias=False # "We omit the use of bias units in all convolutional layers."
            ),
            # Linear activation = do nothing?
            nn.BatchNorm2d(num_features=F1)) # "We apply batch normalization along the feature map dimension"
        
        self.spatial = nn.Conv2d(
            in_channels=F1,
            out_channels=D*F1,
            kernel_size=(C,1), 
            padding='valid', # no padding -> collapses channel dimension
            groups=F1, # produces depthwise convolution
            bias=False # "We omit the use of bias units in all convolutional layers."
            )
        # "We also regularize each spatial filter by using a maximum norm constraint of 1 on its weights"
        max_norm_value = 1.0 
        self.spatial.weight.register_hook(lambda x: torch.clamp(x, min=-max_norm_value, max=max_norm_value))

        self.aggregation = nn.Sequential(
            nn.BatchNorm2d(num_features=D*F1), # "We apply batch normalization along the feature map dimension"
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)), # default stride = kernel_size 
            nn.Dropout(p=p)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d( # depthwise part of separable convolution
            in_channels=D*F1,
            out_channels=D*F1,
            kernel_size=(1,16), 
            padding='same', # keeps initial dims per channel
            groups=D*F1, # produces depthwise convolution
            bias=False # "We omit the use of bias units in all convolutional layers."
            ),
            nn.Conv2d( # pointwise part of separable convolution
            in_channels=D*F1,
            out_channels=F2,
            kernel_size=(1,1), 
            padding='same', # keeps initial dims per channel, shouldn't matter in this case
            groups=1, # default
            bias=False # "We omit the use of bias units in all convolutional layers."
            ),
            # Linear activation = do nothing?
            nn.BatchNorm2d(num_features=F2), # "We apply batch normalization along the feature map dimension"
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)), # default stride = kernel_size
            nn.Dropout(p=p),
            nn.Flatten()
        )

        self.classifier = nn.Linear(
                in_features=F2*(T//32),
                out_features=4, # number of classes
                bias=True # since it's not a convolutional layer..?
            )
        max_norm_value_1 = 0.25 
        self.classifier.weight.register_hook(lambda x: torch.clamp(x, min=-max_norm_value_1, max=max_norm_value_1))
        
        # nn.Softmax(dim=1) # dim = 0 is for batch number; comented out because of nn.CrossEntropyLoss() documentation:
        # "The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general)."



    def forward(self, x):
        x = torch.unsqueeze(x,dim=1) # inserts new dim at specified position, shape = (n_batches, C, T) -> shape = (n_batches, 1, C, T)
        filter_bank = self.temporal(x)
        spatial_pattern = self.spatial(filter_bank)
        block_1_out = self.aggregation(spatial_pattern)
        block_2_out = self.block_2(block_1_out)
        output = self.classifier(block_2_out)
        
        return output
    
def train(model, optimizer, loss_fn, train_loader, val_loader, nepochs=500):
    '''
    Train a pytorch model and evaluate it every epoch.
    Params:
    model - a pytorch model to train
    optimizer - an optimizer
    loss_fn - the criterion (loss function)
    nepochs - number of training epochs
    train_loader - dataloader for the trainset
    val_loader - dataloader for the valset

    Returns: 
    1. state_dict of the model with the lowest validation loss, 
    2. train losses across epochs
    3. validation losses across epochs
    4. validation accuracies across epochs
    '''
    
    # Detect device (GPU) and move model to it
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logger.info(f"Training on {device} device")
    
    # Move model to device
    model = model.to(device)
    
    
    train_losses, val_losses, val_accuracies = [], [], []
    best_model = model.state_dict()

    for e in range(nepochs):
        running_loss = 0
        running_val_loss = 0
        for signals, labels in train_loader: # signals = (batch, C, T), labels = (batch, label)
            
            signals = signals.float() # added to avoid dtype mismatch error

            # Move data to device
            signals, labels = signals.to(device), labels.to(device)

            # Training pass
            model.train() # set model in train mode
            preds = model(signals)
            loss = loss_fn(preds,labels)

            running_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    #else:
        val_loss = 0
        correct = 0
        total = 0
        # Evalaute model on validation at the end of each epoch.
        with torch.no_grad():
            for signals, labels in val_loader: # signals = (batch, C, T), labels = (batch, label)

                signals = signals.float() # added to avoid dtype mismatch error
                
                # Move data to device
                signals, labels = signals.to(device), labels.to(device)
                
                preds = model(signals)
                val_loss = loss_fn(preds,labels)

                running_val_loss += val_loss.item()
                
                # Calculate accuracy
                predicted_classes = torch.argmax(preds, dim=1)
                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)

        # track train loss, validation loss, and validation accuracy
        train_losses.append(running_loss/len(train_loader))
        val_losses.append(running_val_loss/len(val_loader))
        val_accuracies.append(100 * correct / total)

        if running_val_loss == np.min(np.array(val_losses)):
            best_model = model.state_dict()

        if e%50==0:
            logger.info("Epoch: {}/{}.. Training Loss: {:.3f}.. Validation Loss: {:.3f}.. Validation Accuracy: {:.2f}%.. ".format(
                e+1, nepochs, 
                running_loss/len(train_loader),
                running_val_loss/len(val_loader),
                val_accuracies[-1]
            ))

    return best_model, train_losses, val_losses, val_accuracies

def test(model, test_loader, loss_fn) -> float:
    '''
    Test a pytorch model.
    Params:
    model - a pytorch model to test
    test_loader - dataloader for the testset
    loss_fn - the criterion (loss function)

    Returns: 
    test accuracy (percentage)
    '''
    
    # Detect device (GPU) and move model to it
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logger.info(f"Testing on {device} device")
    
    # Move model to device
    model = model.to(device)
    
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in test_loader: # signals = (batch, C, T), labels = (batch, label)

            signals = signals.float() # added to avoid dtype mismatch error
            
            # Move data to device
            signals, labels = signals.to(device), labels.to(device)
            
            preds = model(signals)
            
            # Calculate accuracy
            predicted_classes = torch.argmax(preds, dim=1)
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total