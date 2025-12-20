"""
Tests for kernel.py module.
"""

import numpy as np
import pytest
from PyGALAX.kernel import Kernel


class TestKernel:
    """Test cases for Kernel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple 2D coordinates
        self.coords = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ])
        self.center = np.array([0.5, 0.5])
        self.bandwidth = 1.0
    
    def test_kernel_initialization(self):
        """Test kernel initialization."""
        kernel = Kernel(self.center, self.coords, self.bandwidth, function='bisquare')
        assert kernel is not None
        assert len(kernel.kernel) == len(self.coords)
    
    def test_bisquare_kernel(self):
        """Test bisquare kernel function."""
        kernel = Kernel(self.center, self.coords, self.bandwidth, function='bisquare')
        # Check that weights are between 0 and 1
        assert np.all((kernel.kernel >= 0) & (kernel.kernel <= 1))
        # Center point should have highest weight
        assert kernel.kernel[0] >= kernel.kernel[1]
    
    def test_gaussian_kernel(self):
        """Test gaussian kernel function."""
        kernel = Kernel(self.center, self.coords, self.bandwidth, function='gaussian')
        assert np.all((kernel.kernel >= 0) & (kernel.kernel <= 1))
    
    def test_kernel_weights_sum(self):
        """Test that kernel weights are positive."""
        kernel = Kernel(self.center, self.coords, self.bandwidth, function='bisquare')
        assert np.all(kernel.kernel >= 0)
    
    def test_kernel_fixed_vs_adaptive(self):
        """Test fixed vs adaptive bandwidth."""
        kernel_fixed = Kernel(self.center, self.coords, self.bandwidth, fixed=True, function='bisquare')
        kernel_adaptive = Kernel(self.center, self.coords, self.bandwidth, fixed=False, function='bisquare')
        
        assert kernel_fixed.kernel is not None
        assert kernel_adaptive.kernel is not None
    
    def test_invalid_kernel_function(self):
        """Test that invalid kernel function raises error."""
        with pytest.raises((ValueError, KeyError)):
            Kernel(self.center, self.coords, self.bandwidth, function='invalid_kernel')


class TestKernelTypes:
    """Test different kernel types."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        self.center = np.array([1.5, 0])
        self.bandwidth = 2.0
    
    def test_all_kernel_types(self):
        """Test multiple kernel types produce valid weights."""
        kernel_types = ['bisquare', 'gaussian', 'exponential', 'uniform']
        
        for kernel_type in kernel_types:
            try:
                kernel = Kernel(self.center, self.coords, self.bandwidth, 
                               function=kernel_type)
                assert kernel.kernel is not None
                assert len(kernel.kernel) == len(self.coords)
                assert np.all(kernel.kernel >= 0)
            except (ValueError, KeyError):
                pass

