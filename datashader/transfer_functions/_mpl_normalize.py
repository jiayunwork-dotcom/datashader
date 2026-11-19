"""Matplotlib ``Normalize`` backed by a :class:`ColorCalibration`."""

from __future__ import annotations

import numpy as np

from matplotlib.colors import Normalize

__all__ = ["CalibratedNormalize"]


class CalibratedNormalize(Normalize):
    """A Matplotlib norm using a fixed :class:`ColorCalibration`.

    The norm maps data values to colour scale positions in ``[0, 1]`` with the
    rank based mapping of ``calibration``, and maps positions back to data
    values in the original aggregate units with ``calibration.inverse``.
    It is deliberately *not* auto-scaling: the scale is fully determined by
    the calibration and never depends on the currently displayed data.
    Consequently :attr:`vmin` and :attr:`vmax` are read-only views of the
    calibration range; assigning to them (as Matplotlib's colorbar does) is
    silently ignored.

    Instances are usually created with :meth:`ColorCalibration.as_normalize`.
    """

    def __init__(self, calibration, clip=False):
        # Set the calibration first: `vmin` and `vmax` are derived from it.
        self.calibration = calibration
        super().__init__(calibration.vmin, calibration.vmax, clip)

    # -- keep the scale fixed ---------------------------------------------
    @property
    def vmin(self):
        return self.calibration.vmin

    @vmin.setter
    def vmin(self, value):
        # The scale is defined by the calibration; ignore assignments (in
        # particular the ones made by Matplotlib's colorbar machinery).
        self._vmin = self.calibration.vmin

    @property
    def vmax(self):
        return self.calibration.vmax

    @vmax.setter
    def vmax(self, value):
        self._vmax = self.calibration.vmax

    # -- the core mapping -------------------------------------------------
    def process_value(self, value):
        """Return ``(masked float array, is_scalar)`` for *value*."""
        is_scalar = not np.iterable(value)
        if is_scalar:
            value = [value]
        arr = np.ma.asarray(value)
        if arr.dtype.kind not in "fc":
            arr = arr.astype(np.float64)
        else:
            arr = arr.astype(np.float64, copy=False)
        return np.ma.asarray(arr), is_scalar

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        data = np.ma.getdata(result)
        mask = np.ma.getmaskarray(result)
        # NaN is invalid, but +/-inf is a valid (saturating) value.
        with np.errstate(invalid="ignore"):
            mask = mask | np.isnan(data)
        pos = self.calibration.forward(data)
        out = np.ma.masked_array(pos, mask=mask, copy=False)
        if is_scalar:
            out = out[0]
        return out

    def inverse(self, value):
        if np.ma.isMaskedArray(value):
            value = np.ma.getdata(value)
        out = self.calibration.inverse(np.asarray(value))
        # Keep the scalar / array distinction of the input.
        if np.ndim(value) == 0:
            return float(out)
        return out

    def autoscale(self, A):
        """No-op: the scale is defined by the calibration."""

    def autoscale_None(self, A):
        """No-op: the scale is defined by the calibration."""

    def scaled(self):
        return True
