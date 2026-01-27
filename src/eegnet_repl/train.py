"""Train a hazard classifier from cached NASA NeoWs data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eegnet_repl.config import Paths
from eegnet_repl.dataset import BCICI2ADataset, build_dataset_from_preprocessed
from eegnet_repl.logger import logger
from eegnet_repl.model import EEGNet, train, test
# from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import KFold

import numpy as np
import torch
# import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.optim as optim


BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# def save_model(model: TrainedModel, out_path: Path) -> None:
#     """Persist a trained model bundle using pickle."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with out_path.open("wb") as f:
#         pickle.dump(model, f)
#     logger.info("Saved model: %s", out_path)


# def load_model(path: Path) -> TrainedModel:
#     """Load a trained model bundle."""
#     with path.open("rb") as f:
#         obj = pickle.load(f)
#     assert isinstance(obj, TrainedModel)  # noqa: S101
#     return obj


# def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, out_path: Path) -> None:
#     """Save a confusion matrix plot."""
#     fig, ax = plt.subplots()
#     ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
#     ax.set_title("Confusion matrix")
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, bbox_inches="tight", dpi=150)
#     plt.close(fig)

def within_subject_training():
    '''
    Train and validate EEGNet model for within-subject classification.
    
    Returns:
    - Per subject test accuracies array
    - Average test accuracy across subjects
    - Best model state dicts per subject array
    '''
    
    paths = Paths.from_here()
    
    # Initial split into train and test sets
    per_subject_test_acc = []
    best_model_states_all_subjects = []

    # Per subject training in a loop
    for subject_id in range(1, 10):  # Assuming subject IDs are from 1 to 9
        logger.info(f"Training for Subject {subject_id}")
        
        # Load preprocessed data for the subject
        subject_data = build_dataset_from_preprocessed(subject=subject_id)

        # test accuracy array for this subject
        subject_test_acc = []

        # track best model for this subject
        best_val_acc = 0
        best_model_state = None

        # K-Fold Cross Validation on train_data
        splits = 4
        kf = KFold(n_splits=splits, shuffle=True, random_state=42)
        
        for fold, (train_val_ids, test_ids) in enumerate(kf.split(subject_data)):
            logger.info(f"  Fold {fold+1}/{splits}")

            # Further split train_val into train and validation (75-25 split)
            val_size = len(train_val_ids) // 4
            train_ids = train_val_ids[val_size:]
            val_ids = train_val_ids[:val_size]

            # Create data subsets
            train_subset = Subset(subject_data, train_ids)
            val_subset = Subset(subject_data, val_ids)
            test_subset = Subset(subject_data, test_ids)

            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

            # Initialize model, loss function, and optimizer
            model = EEGNet(C=subject_data.X.shape[1], T=subject_data.X.shape[2], p=0.5)  # p=0.5 for within-subject
            
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=LEARNING_RATE,
                eps=1e-07,  # the only change from default eps=1e-8
                foreach=None,  # default
                fused=None,  # default
            )
            loss_fn = nn.CrossEntropyLoss()

            # Train the model
            best_fold_model, train_losses, val_losses, val_accuracies = train(
                model, optimizer, loss_fn, train_loader, val_loader, nepochs=EPOCHS
            )
            
            # Load best model from training
            model.load_state_dict(best_fold_model)
            
            # Get final validation accuracy
            final_val_acc = val_accuracies[-1]
            
            # Test the model
            test_acc = test(model, test_loader, loss_fn)
            
            # Log validation and test accuracy
            logger.info(f"    Validation Accuracy: {final_val_acc:.2f}%")
            logger.info(f"    Test Accuracy: {test_acc:.2f}%")
            
            # Store test accuracy for this fold
            subject_test_acc.append(test_acc)
            
            # Track best model based on validation accuracy
            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_model_state = best_fold_model
        
        # Calculate average test accuracy for this subject
        avg_test_acc = sum(subject_test_acc) / len(subject_test_acc)
        per_subject_test_acc.append(avg_test_acc)
        logger.info(f"Subject {subject_id} - Average Test Accuracy: {avg_test_acc:.2f}%")
        
        # Save best model for this subject
        model_path = paths.models / f"subject_{subject_id:02d}_best_model.pth"
        paths.models.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, model_path)
        logger.info(f"Saved best model for Subject {subject_id}: {model_path}")
        
        # Store best model state dict
        best_model_states_all_subjects.append(best_model_state)
    
    # Calculate average test accuracy across all subjects
    avg_test_acc_all_subjects = sum(per_subject_test_acc) / len(per_subject_test_acc)
    logger.info(f"\nOverall Average Test Accuracy across all subjects: {avg_test_acc_all_subjects:.2f}%")
    
    return per_subject_test_acc, avg_test_acc_all_subjects, best_model_states_all_subjects


def cross_subject_training():
    '''
    Train and validate EEGNet model for cross-subject classification.
    For each subject: train on 5 random subjects, validate on 3 others, test on target subject.
    Repeat 10 times per subject (90 folds total).

    Returns:
    - Best model state dict based on validation accuracy
    - Per subject test accuracies array
    - Average test accuracy across subjects
    '''
    
    paths = Paths.from_here()
    
    # Load data for all subjects
    logger.info("Loading data for all subjects...")
    all_subjects_data = {}
    for subject_id in range(1, 10):
        all_subjects_data[subject_id] = build_dataset_from_preprocessed(subject=subject_id)
    
    # Track results
    all_fold_test_acc = []
    per_subject_test_acc = []
    best_val_acc = 0
    best_model_state = None
    fold_count = 0
    
    # For each subject as test subject
    for subject_id in range(1, 10):
        logger.info(f"\nTraining with Subject {subject_id} as test subject")
        
        subject_test_acc = []
        
        # Get test data for this subject
        test_data = all_subjects_data[subject_id]
        
        # Get other subjects
        other_subjects = [s for s in range(1, 10) if s != subject_id]
        
        # Repeat 10 times
        for repeat in range(10):
            fold_count += 1
            logger.info(f"  Fold {fold_count}/90 (Subject {subject_id}, Repeat {repeat+1}/10)")
            
            # Randomly select 5 subjects for training, 3 for validation
            rng = np.random.RandomState(42 + fold_count)  # For reproducibility
            shuffled_subjects = rng.permutation(other_subjects)
            train_subjects = shuffled_subjects[:5]
            val_subjects = shuffled_subjects[5:]
            
            # Combine training data from selected subjects
            train_X_list = []
            train_y_list = []
            for subj in train_subjects:
                train_X_list.append(all_subjects_data[subj].X)
                train_y_list.append(all_subjects_data[subj].y)
            
            train_X = np.concatenate(train_X_list, axis=0)
            train_y = np.concatenate(train_y_list, axis=0)
            
            # Combine validation data from selected subjects
            val_X_list = []
            val_y_list = []
            for subj in val_subjects:
                val_X_list.append(all_subjects_data[subj].X)
                val_y_list.append(all_subjects_data[subj].y)
            
            val_X = np.concatenate(val_X_list, axis=0)
            val_y = np.concatenate(val_y_list, axis=0)
            
            # Create datasets
            train_dataset = BCICI2ADataset(train_X, train_y)
            val_dataset = BCICI2ADataset(val_X, val_y)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
            
            # Initialize model with p=0.25 for cross-subject
            model = EEGNet(C=train_X.shape[1], T=train_X.shape[2], p=0.25)
            
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=LEARNING_RATE,
                eps=1e-07,
                foreach=None,
                fused=None,
            )
            loss_fn = nn.CrossEntropyLoss()
            
            # Train the model
            best_fold_model, train_losses, val_losses, val_accuracies = train(
                model, optimizer, loss_fn, train_loader, val_loader, nepochs=EPOCHS
            )
            
            # Load best model from training
            model.load_state_dict(best_fold_model)
            
            # Get final validation accuracy
            final_val_acc = val_accuracies[-1]
            
            # Test the model
            test_acc = test(model, test_loader, loss_fn)
            
            # Log results
            logger.info(f"    Validation Accuracy: {final_val_acc:.2f}%")
            logger.info(f"    Test Accuracy: {test_acc:.2f}%")
            
            # Store test accuracy
            subject_test_acc.append(test_acc)
            all_fold_test_acc.append(test_acc)
            
            # Track best model globally
            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_model_state = best_fold_model
        
        # Calculate average test accuracy for this subject
        avg_test_acc = sum(subject_test_acc) / len(subject_test_acc)
        per_subject_test_acc.append(avg_test_acc)
        logger.info(f"Subject {subject_id} - Average Test Accuracy: {avg_test_acc:.2f}%")
    
    # Calculate overall statistics
    avg_test_acc_all = sum(all_fold_test_acc) / len(all_fold_test_acc)
    std_error = np.std(all_fold_test_acc) / np.sqrt(len(all_fold_test_acc))
    
    logger.info(f"\nOverall Average Test Accuracy: {avg_test_acc_all:.2f}% ± {std_error:.2f}%")
    logger.info(f"Mean per-subject test accuracy: {np.mean(per_subject_test_acc):.2f}%")
    
    # Save best model
    model_path = paths.models / "cross_subject_best_model.pth"
    paths.models.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_state, model_path)
    logger.info(f"Saved best cross-subject model: {model_path}")
    
    return best_model_state, per_subject_test_acc, avg_test_acc_all

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train a EEGNet model.")
    parser.add_argument("--training_type", type=str, help="Training type [Cross-Subject, Within-Subject].", default="Within-Subject")
    # parser.add_argument("--subject", type=int, default=None, help="Subject ID for Within-Subject training.")
    args = parser.parse_args()

    if args.training_type == "Within-Subject":
        logger.info("Training Within-Subject models for all subjects...")
    else:
        logger.info("Training Cross-Subject model...")

    paths = Paths.from_here()

    if args.training_type == "Within-Subject":
        within_subject_training()
    else:
        cross_subject_training()

if __name__ == "__main__":
    main()
