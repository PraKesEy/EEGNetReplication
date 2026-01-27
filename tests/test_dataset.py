from __future__ import annotations

import unittest
import numpy as np
from pathlib import Path

from eegnet_repl.dataset import (
    BCICI2ADataset, 
    raw_exponential_moving_standardize, 
    map_labels,
    build_dataset_from_preprocessed
)


class TestBCICI2ADataset(unittest.TestCase):
    """Test the BCICI2ADataset class."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic EEG data: 100 samples, 22 channels, 256 time points
        self.X = np.random.randn(100, 22, 256).astype(np.float32)
        self.y = np.random.randint(0, 4, 100)  # 4 classes (0-3)
        self.dataset = BCICI2ADataset(X=self.X, y=self.y)
    
    def test_dataset_length(self):
        """Test that dataset returns correct length."""
        self.assertEqual(len(self.dataset), 100)
        self.assertEqual(len(self.dataset), self.X.shape[0])
    
    def test_dataset_getitem(self):
        """Test that dataset returns correct item format."""
        sample, label = self.dataset[0]
        
        # Check shapes
        self.assertEqual(sample.shape, (22, 256))
        self.assertIsInstance(label, int)
        
        # Check values match
        np.testing.assert_array_equal(sample, self.X[0])
        self.assertEqual(label, int(self.y[0]))
    
    def test_dataset_getitem_range(self):
        """Test dataset getitem with different indices."""
        for i in [0, 10, 50, 99]:
            sample, label = self.dataset[i]
            np.testing.assert_array_equal(sample, self.X[i])
            self.assertEqual(label, int(self.y[i]))


class TestRawExponentialMovingStandardize(unittest.TestCase):
    """Test the raw_exponential_moving_standardize function."""
    
    def test_output_shape(self):
        """Test that output has same shape as input."""
        x = np.random.randn(22, 1000)
        x_std = raw_exponential_moving_standardize(x)
        self.assertEqual(x_std.shape, x.shape)
    
    def test_standardization_basic(self):
        """Test basic standardization properties."""
        # Create data with known mean and std
        np.random.seed(42)
        x = np.random.randn(5, 2000) * 10 + 5  # mean=5, std=10
        
        x_std = raw_exponential_moving_standardize(x, factor_new=0.001, init_block_size=1000)
        
        # After standardization, data should have values roughly around 0
        # (though not exactly due to exponential moving nature)
        self.assertTrue(np.abs(np.mean(x_std[:, -1000:])) < 1.0)
    
    def test_factor_new_parameter(self):
        """Test different factor_new values produce different results."""
        np.random.seed(42)
        x = np.random.randn(3, 1500)
        
        x_std_small = raw_exponential_moving_standardize(x, factor_new=0.001)
        x_std_large = raw_exponential_moving_standardize(x, factor_new=0.01)
        
        # Results should be different with different smoothing factors
        self.assertFalse(np.array_equal(x_std_small, x_std_large))
    
    def test_init_block_size_parameter(self):
        """Test different init_block_size values."""
        np.random.seed(42)
        x = np.random.randn(3, 2000)
        
        x_std_small = raw_exponential_moving_standardize(x, init_block_size=500)
        x_std_large = raw_exponential_moving_standardize(x, init_block_size=1500)
        
        # Results should be different with different initialization blocks
        self.assertFalse(np.array_equal(x_std_small, x_std_large))
    
    def test_single_channel(self):
        """Test with single channel data."""
        x = np.random.randn(1, 1000)
        x_std = raw_exponential_moving_standardize(x)
        self.assertEqual(x_std.shape, (1, 1000))
    
    def test_constant_signal(self):
        """Test with constant signal."""
        x = np.ones((3, 1000)) * 5.0
        x_std = raw_exponential_moving_standardize(x)
        
        # Constant signal should result in near-zero output after initial block
        # (due to division by sqrt(variance + epsilon))
        self.assertTrue(np.all(np.abs(x_std[:, -500:]) < 10.0))


class TestMapLabels(unittest.TestCase):
    """Test the map_labels function."""
    
    def test_basic_mapping(self):
        """Test basic label mapping functionality."""
        labels = np.array([1, 2, 3, 1, 2, 3])
        map_dict = {1: 0, 2: 1, 3: 2}
        
        expected = np.array([0, 1, 2, 0, 1, 2])
        result = map_labels(labels, map_dict)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_non_contiguous_mapping(self):
        """Test mapping with non-contiguous original labels."""
        labels = np.array([7, 8, 9, 10, 7, 8])
        map_dict = {7: 0, 8: 1, 9: 2, 10: 3}
        
        expected = np.array([0, 1, 2, 3, 0, 1])
        result = map_labels(labels, map_dict)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_output_dtype(self):
        """Test that output has same dtype as input."""
        labels = np.array([1, 2, 3], dtype=np.int32)
        map_dict = {1: 0, 2: 1, 3: 2}
        
        result = map_labels(labels, map_dict)
        self.assertEqual(result.dtype, labels.dtype)
    
    def test_output_shape(self):
        """Test that output has same shape as input."""
        labels = np.array([[1, 2], [3, 1]])
        map_dict = {1: 0, 2: 1, 3: 2}
        
        result = map_labels(labels, map_dict)
        self.assertEqual(result.shape, labels.shape)
    
    def test_unmapped_labels_error(self):
        """Test that unmapped labels raise an error."""
        labels = np.array([1, 2, 3, 4])  # 4 is not in the map
        map_dict = {1: 0, 2: 1, 3: 2}
        
        with self.assertRaises(RuntimeError):
            map_labels(labels, map_dict)
    
    def test_single_label_mapping(self):
        """Test mapping with single unique label."""
        labels = np.array([5, 5, 5, 5])
        map_dict = {5: 0}
        
        expected = np.array([0, 0, 0, 0])
        result = map_labels(labels, map_dict)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_empty_labels(self):
        """Test with empty label array."""
        labels = np.array([])
        map_dict = {1: 0, 2: 1}
        
        result = map_labels(labels, map_dict)
        self.assertEqual(len(result), 0)


class TestDatasetIntegration(unittest.TestCase):
    """Integration tests for dataset functionality."""
    
    def test_dataset_creation_with_processed_data(self):
        """Test creating dataset from processed arrays."""
        # Simulate the output of break_data_into_epochs
        X = np.random.randn(50, 22, 256)  # 50 epochs, 22 channels, 256 time points
        y = np.random.randint(0, 4, 50)   # 50 labels from 4 classes
        
        dataset = BCICI2ADataset(X=X, y=y)
        
        # Test basic functionality
        self.assertEqual(len(dataset), 50)
        
        # Test that we can iterate through the dataset
        for i in range(min(5, len(dataset))):  # Test first 5 samples
            sample, label = dataset[i]
            self.assertEqual(sample.shape, (22, 256))
            self.assertIn(label, [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()



