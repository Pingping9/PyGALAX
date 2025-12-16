"""
Kernel functions for spatial weighting in GALAX.
"""

import numpy as np
np.float = float


class Kernel:
    """
    Kernel function specifications for GALAX.
    
    Parameters
    ----------
    coords_i : array
        Coordinates of the target location
    coords : array
        Coordinates of all locations
    bw : float or int
        Bandwidth parameter
    fixed : bool, default=False
        Whether to use fixed or adaptive bandwidth
    function : str, default='bisquare'
        Kernel function type
    spherical : bool, default=False
        Whether to use spherical distance calculation
    """
    def __init__(self, coords_i, coords, bw, fixed=False, function='bisquare', spherical=False):
        self.coords_i = coords_i
        self.coords = coords
        self.bw = bw
        self.fixed = fixed
        self.function = function.lower()
        self.spherical = spherical
        self.kernel = self._compute_kernel()

    def local_cdist(self):
        """Compute distance between points"""
        if self.spherical:
            # Haversine formula for spherical coordinates
            dLat = np.radians(self.coords[:, 1] - self.coords_i[1])
            dLon = np.radians(self.coords[:, 0] - self.coords_i[0])
            lat1 = np.radians(self.coords[:, 1])
            lat2 = np.radians(self.coords_i[1])
            a = np.sin(dLat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            R = 6371.0
            return R * c
        else:
            # Euclidean distance for projected coordinates
            return np.sqrt(np.sum((self.coords_i - self.coords)**2, axis=1))

    def _compute_kernel(self):
        """Compute kernel weights"""
        dvec = self.local_cdist()

        if not self.fixed:
            bandwidth = np.partition(dvec, int(self.bw)-1)[int(self.bw)-1] * 1.0000001
        else:
            bandwidth = self.bw

        zs = dvec / bandwidth

        # Different kernel functions
        if self.function == 'bisquare':
            kernel = (1 - zs**2)**2
            kernel[dvec >= bandwidth] = 0
        elif self.function == 'gaussian':
            kernel = np.exp(-0.5 * (zs)**2)
        elif self.function == 'exponential':
            kernel = np.exp(-zs)
        elif self.function == 'triangular':
            kernel = 1 - zs
            kernel[dvec >= bandwidth] = 0
        elif self.function == 'uniform':
            kernel = np.ones(len(zs)) * 0.5
            kernel[dvec >= bandwidth] = 0
        elif self.function == 'quadratic':
            kernel = (3./4) * (1 - zs**2)
            kernel[dvec >= bandwidth] = 0
        elif self.function == 'quartic':
            kernel = (15./16) * (1 - zs**2)**2
            kernel[dvec >= bandwidth] = 0
        else:
            raise ValueError(f"Unsupported kernel function: {self.function}")

        return kernel
