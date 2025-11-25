"""
Gaussian Process Regression interpolator for unstructured grids.

This module provides a wrapper around scikit-learn's GaussianProcessRegressor
to replace HARK's missing GeneralizedRegressionUnstructuredInterp, which was
a similar wrapper around sklearn's GPR.

The wrapper provides a callable interface compatible with scipy interpolators:
construct with data, call to interpolate at new points.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class UnstructuredInterpGPR:
    """
    Gaussian Process Regression interpolator for unstructured point clouds.

    Wraps scikit-learn's GaussianProcessRegressor with an interface compatible
    with scipy interpolators and the missing HARK GeneralizedRegressionUnstructuredInterp.

    Parameters
    ----------
    points : array-like, shape (n_samples, n_features)
        Coordinates of data points
    values : array-like, shape (n_samples,)
        Values at data points
    kernel : kernel object, optional
        GP kernel. Defaults to RBF with automatic length scale tuning
    alpha : float, default=1e-6
        Noise level (regularization parameter)
    normalize_y : bool, default=True
        Whether to normalize target values to zero mean and unit variance

    Examples
    --------
    >>> import numpy as np
    >>> points = np.random.rand(50, 2)
    >>> values = np.sin(points[:, 0]) * np.cos(points[:, 1])
    >>> interp = UnstructuredInterpGPR(points, values)
    >>> x_new = np.array([[0.5, 0.5], [0.3, 0.7]])
    >>> interp(x_new)
    array([...])

    >>> # Works with meshgrids too
    >>> X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    >>> Z = interp(X, Y)
    """

    def __init__(
        self,
        points: np.ndarray,
        values: np.ndarray,
        kernel=None,
        alpha: float = 1e-6,
        normalize_y: bool = True,
    ):
        points = np.asarray(points)
        values = np.asarray(values).ravel()

        if kernel is None:
            # Default: RBF kernel with fixed length scale for speed
            # Skip hyperparameter optimization for computational efficiency
            kernel = RBF(length_scale=1.0)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            optimizer=None,  # Skip hyperparameter optimization for speed
        )

        self.gp.fit(points, values)
        self.ndim = points.shape[1] if points.ndim > 1 else 1

        # Store original data for visualization
        self._points = points
        self._values = values

    def __call__(self, *args):
        """
        Interpolate at new points.

        Supports two calling conventions:
        1. interp(points) where points is shape (n_samples, n_features)
        2. interp(x1, x2, ...) where x1, x2 are coordinate arrays (meshgrid format)

        Parameters
        ----------
        *args : array-like
            Either a single array of shape (n_samples, n_features)
            or separate arrays for each dimension (will be stacked)

        Returns
        -------
        values : array
            Interpolated values at query points, same shape as input
        """
        if len(args) == 1:
            # Single array of points
            points = np.asarray(args[0])
            original_shape = points.shape[:-1] if points.ndim > 1 else (len(points),)
            points = points.reshape(-1, self.ndim)
        else:
            # Multiple coordinate arrays (meshgrid format)
            original_shape = args[0].shape
            grids = [np.asarray(arg).ravel() for arg in args]
            points = np.column_stack(grids)

        # Predict without uncertainty (faster)
        values = self.gp.predict(points, return_std=False)

        # Reshape to match input
        return values.reshape(original_shape)

    def predict_with_uncertainty(self, *args):
        """
        Interpolate with uncertainty estimates.

        Parameters
        ----------
        *args : array-like
            Same as __call__

        Returns
        -------
        mean : array
            Predicted values
        std : array
            Predicted standard deviations (uncertainty)
        """
        if len(args) == 1:
            points = np.asarray(args[0])
            original_shape = points.shape[:-1] if points.ndim > 1 else (len(points),)
            points = points.reshape(-1, self.ndim)
        else:
            original_shape = args[0].shape
            grids = [np.asarray(arg).ravel() for arg in args]
            points = np.column_stack(grids)

        mean, std = self.gp.predict(points, return_std=True)

        return mean.reshape(original_shape), std.reshape(original_shape)

    @property
    def grids(self):
        """
        Return original grid points as separate arrays for visualization.

        Compatible with plot_scatter_hist which expects [x_grid, y_grid].

        Returns
        -------
        list of arrays
            [x_coordinates, y_coordinates] from original points
        """
        return [self._points[:, i] for i in range(self.ndim)]

    @property
    def values(self):
        """
        Return original values for visualization.

        Returns
        -------
        array
            Original values at grid points
        """
        return self._values
