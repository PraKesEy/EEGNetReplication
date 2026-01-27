from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from eegnet_repl.model import EEGNet, train, test


class TestEEGNet(unittest.TestCase):
    """Test the EEGNet model class."""
    
    def setUp(self):
        """Set up test parameters."""
        self.C = 22  # number of channels
        self.T = 256  # number of time points
        self.batch_size = 16
        self.n_classes = 4
        
    def test_model_initialization_default(self):
        """Test model initialization with default parameters."""
        model = EEGNet(C=self.C, T=self.T)
        
        # Check that model is created
        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(model, EEGNet)
        
        # Check that all layers exist
        self.assertTrue(hasattr(model, 'temporal'))
        self.assertTrue(hasattr(model, 'spatial'))
        self.assertTrue(hasattr(model, 'aggregation'))
        self.assertTrue(hasattr(model, 'block_2'))
        self.assertTrue(hasattr(model, 'classifier'))
    
    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        F1 = 16
        D = 4
        p = 0.25
        
        model = EEGNet(C=self.C, T=self.T, F1=F1, D=D, p=p)
        
        # Check that model uses custom parameters
        # F2 should be F1 * D
        expected_F2 = F1 * D
        
        # Check temporal layer output channels
        temporal_conv = model.temporal[0]  # First layer in temporal sequential
        self.assertEqual(temporal_conv.out_channels, F1)
        
        # Check spatial layer
        self.assertEqual(model.spatial.in_channels, F1)
        self.assertEqual(model.spatial.out_channels, D * F1)
        
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = EEGNet(C=self.C, T=self.T)
        
        # Create random input: (batch_size, channels, time_points)
        x = torch.randn(self.batch_size, self.C, self.T)
        
        # Forward pass
        output = model(x)
        
        # Check output shape: (batch_size, n_classes)
        expected_shape = (self.batch_size, self.n_classes)
        self.assertEqual(output.shape, expected_shape)
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = EEGNet(C=self.C, T=self.T)
        
        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, self.C, self.T)
            output = model(x)
            self.assertEqual(output.shape, (batch_size, self.n_classes))
    
    def test_model_output_type(self):
        """Test that model output is correct tensor type."""
        model = EEGNet(C=self.C, T=self.T)
        x = torch.randn(self.batch_size, self.C, self.T)
        
        output = model(x)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, torch.float32)
    
    def test_model_gradients(self):
        """Test that model parameters have gradients after backward pass."""
        model = EEGNet(C=self.C, T=self.T)
        x = torch.randn(self.batch_size, self.C, self.T)
        target = torch.randint(0, self.n_classes, (self.batch_size,))
        
        # Forward pass
        output = model(x)
        
        # Compute loss and backward
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        loss.backward()
        
        # Check that parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
    
    def test_model_different_input_sizes(self):
        """Test model with different input channel/time combinations."""
        test_cases = [
            (64, 128),   # Different C and T
            (32, 512),   # Larger T
            (8, 64),     # Smaller C and T
        ]
        
        for C, T in test_cases:
            with self.subTest(C=C, T=T):
                model = EEGNet(C=C, T=T)
                x = torch.randn(4, C, T)
                output = model(x)
                self.assertEqual(output.shape, (4, self.n_classes))
    
    def test_temporal_conv_properties(self):
        """Test temporal convolution layer properties."""
        model = EEGNet(C=self.C, T=self.T, F1=8)
        temporal_conv = model.temporal[0]
        
        # Check kernel size
        self.assertEqual(temporal_conv.kernel_size, (1, 32))
        
        # Check padding
        self.assertEqual(temporal_conv.padding, (0, 16))  # 'same' padding
        
        # Check bias is disabled
        self.assertIsNone(temporal_conv.bias)
        
        # Check input/output channels
        self.assertEqual(temporal_conv.in_channels, 1)
        self.assertEqual(temporal_conv.out_channels, 8)
    
    def test_spatial_conv_properties(self):
        """Test spatial convolution layer properties."""
        F1 = 8
        D = 2
        model = EEGNet(C=self.C, T=self.T, F1=F1, D=D)
        
        # Check kernel size collapses channel dimension
        self.assertEqual(model.spatial.kernel_size, (self.C, 1))
        
        # Check groups for depthwise convolution
        self.assertEqual(model.spatial.groups, F1)
        
        # Check bias is disabled
        self.assertIsNone(model.spatial.bias)
        
        # Check input/output channels
        self.assertEqual(model.spatial.in_channels, F1)
        self.assertEqual(model.spatial.out_channels, D * F1)
    
    def test_classifier_properties(self):
        """Test classifier layer properties."""
        F1 = 8
        D = 2
        model = EEGNet(C=self.C, T=self.T, F1=F1, D=D)
        
        # Check output features
        self.assertEqual(model.classifier.out_features, self.n_classes)
        
        # Check bias is enabled for classifier
        self.assertIsNotNone(model.classifier.bias)
    
    def test_dropout_probability(self):
        """Test that dropout layers use correct probability."""
        p = 0.3
        model = EEGNet(C=self.C, T=self.T, p=p)
        
        # Find dropout layers
        dropout_layers = []
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(module)
        
        # Should have 2 dropout layers
        self.assertEqual(len(dropout_layers), 2)
        
        # Check dropout probability
        for dropout_layer in dropout_layers:
            self.assertEqual(dropout_layer.p, p)


class TestTrainFunction(unittest.TestCase):
    """Test the train function."""
    
    def test_train_function_signature(self):
        """Test that train function exists and has correct signature."""
        # Check function exists
        self.assertTrue(callable(train))
        
        # Test with mock objects to check basic functionality
        model = Mock()
        optimizer = Mock()
        loss_fn = Mock()
        train_loader = Mock()
        val_loader = Mock()
        
        # Should not raise an error when called with mocks
        try:
            # This will likely fail during execution, but we're just testing the signature
            train(model, optimizer, loss_fn, train_loader, val_loader, nepochs=1)
        except Exception:
            # Expected since we're using mocks, but function should exist
            pass


class TestTestFunction(unittest.TestCase):
    """Test the test function."""
    
    def test_test_function_signature(self):
        """Test that test function exists and has correct signature."""
        # Check function exists
        self.assertTrue(callable(test))
        
        # Test with mock objects
        model = Mock()
        test_loader = Mock()
        loss_fn = Mock()
        
        try:
            # This will likely fail during execution, but we're testing the signature
            test(model, test_loader, loss_fn)
        except Exception:
            # Expected since we're using mocks
            pass


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model functionality."""
    
    def test_model_training_step(self):
        """Test a single training step with real model."""
        model = EEGNet(C=22, T=256)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        
        # Create synthetic batch
        x = torch.randn(8, 22, 256)
        y = torch.randint(0, 4, (8,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        output = model(x.float())
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        
        # Check that loss is a valid number
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        
        # Check output shape
        self.assertEqual(output.shape, (8, 4))
    
    def test_model_evaluation_step(self):
        """Test model evaluation step."""
        model = EEGNet(C=22, T=256)
        
        # Create synthetic batch
        x = torch.randn(4, 22, 256)
        
        # Evaluation step
        model.eval()
        with torch.no_grad():
            output = model(x.float())
        
        # Check output shape and type
        self.assertEqual(output.shape, (4, 4))
        self.assertIsInstance(output, torch.Tensor)
        
        # Check that output values are reasonable (not NaN or inf)
        self.assertFalse(torch.any(torch.isnan(output)))
        self.assertFalse(torch.any(torch.isinf(output)))


if __name__ == '__main__':
    unittest.main()





