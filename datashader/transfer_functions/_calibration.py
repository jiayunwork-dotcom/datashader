"""Rank based colour calibration built from a reference aggregate.

The purpose of a :class:`ColorCalibration` is to freeze the colour mapping of
a scalar aggregate to a *reference* raster, so that colours can be compared
between different views of the same data.  A Datashader aggregate is normally
shaded relative to the minimum and maximum of the currently visible window;
with a calibration the colour of a cell only depends on the value of that
cell and on the (fixed) reference distribution.

Representation
--------------
The reference distribution is represented by a piecewise approximation of
its cumulative distribution function (CDF).  The value axis is partitioned
into intervals; for every interval we store

* ``[pos_lo, pos_hi]``: the fraction of reference cells below the interval
  and the fraction at or below it (so ``pos_lo == pos_hi`` for empty
  intervals), and
* ``[bmin, bmax]``: the smallest and largest reference value inside it.

A value is mapped to the midpoint of the rank interval of the interval it
belongs to, i.e. values are coloured by their percentile rank in the
reference.  Values outside the reference range saturate at ``0`` and ``1``.
Isolated intervals that hold a single distinct value have an exactly known
rank interval, so tied values share the exact midpoint of their tied group.

Construction
------------
The partition is seeded with quantiles of a sample of the reference, and is
then refined: every interval holding more than ``max_error`` of the reference
mass is split further.  Heavy tie groups (repeated values holding a
significant fraction of the mass) are detected with a bounded heavy hitter
scan and given dedicated intervals, so a spike cannot blur the values next to
it.  Every pass streams the reference in fixed size blocks of at most
:data:`_CHUNK` cells, so the construction working memory is bounded by a few
MiB and does not grow with the size of the reference raster.  No temporary
files are used.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["ColorCalibration", "calibrate"]


#: Maximum number of intervals retained by a calibration.  This bounds both
#: the memory retained by the calibration and the working memory used during
#: construction, independent of the size of the reference raster.
_MAX_BINS = 1 << 14

#: Number of sub-intervals a "heavy" interval is split into per pass (when
#: the bin budget allows it).
_SPLIT = 128

#: Number of reference cells processed per block while scanning.  Together
#: with :data:`_MAX_BINS` this bounds the construction working memory.
_CHUNK = 1 << 15

#: Maximum number of refinement passes over the reference.
_MAX_PASSES = 16

#: Approximate size of the sample used to seed the partition.
_SAMPLE = 8192


def _iter_blocks(data, mask, chunk):
    """Yield ``(values, mask)`` blocks with at most *chunk* elements."""
    rows, cols = data.shape
    if rows == 0 or cols == 0:
        return
    if cols <= chunk:
        step = max(1, chunk // cols)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            block = data[r0:r1].reshape(-1)
            bmask = None if mask is None else mask[r0:r1].reshape(-1)
            yield block, bmask
    else:
        for r in range(rows):
            row = data[r]
            mrow = None if mask is None else mask[r]
            for c0 in range(0, cols, chunk):
                c1 = min(cols, c0 + chunk)
                bmask = None if mrow is None else mrow[c0:c1]
                yield row[c0:c1], bmask


def _invalid(block, block_mask):
    """Boolean array of cells that must be skipped in the reference."""
    kind = block.dtype.kind
    if kind == "u":
        invalid = block == 0
    elif kind == "b":
        invalid = ~block
    elif kind in "iu":
        invalid = None
    else:
        # NaN and both infinities are not valid reference samples.
        invalid = ~np.isfinite(block)
    if block_mask is not None:
        if invalid is None:
            invalid = np.asarray(block_mask, dtype=bool)
        else:
            invalid = invalid | block_mask
    return invalid


def _valid_values(block, block_mask):
    invalid = _invalid(block, block_mask)
    if invalid is None:
        return block
    return block[~invalid]


def _min_max_count(data, mask, chunk):
    """Return ``(vmin, vmax, count)`` of the valid reference cells."""
    vmin = np.inf
    vmax = -np.inf
    count = 0
    for block, bmask in _iter_blocks(data, mask, chunk):
        values = _valid_values(block, bmask)
        if values.size:
            count += values.size
            vmin = min(vmin, values.min())
            vmax = max(vmax, values.max())
    return vmin, vmax, count


def _sample_values(data, mask, chunk, ndata, size=_SAMPLE, seed=0):
    """Return a random sample of the valid reference values.

    Values are drawn uniformly at random (with a fixed seed) rather than at
    a fixed stride, so that the sample cannot systematically miss a value
    that happens to be laid out periodically in the reference raster.
    """
    if ndata == 0:
        return np.empty(0, dtype=np.float64)
    if ndata <= size:
        # Small reference: use every valid cell.
        picked = []
        for block, bmask in _iter_blocks(data, mask, chunk):
            values = _valid_values(block, bmask)
            if values.size:
                picked.append(values)
        return np.concatenate(picked).astype(np.float64, copy=False)

    rate = size / ndata
    rng = np.random.default_rng(seed)
    picked = []
    for block, bmask in _iter_blocks(data, mask, chunk):
        values = _valid_values(block, bmask)
        if not values.size:
            continue
        # Draw the number of samples expected from this block, from the
        # block's own cells.  This is unbiased irrespective of the layout.
        take = int(round(values.size * rate))
        if take:
            idx = rng.integers(0, values.size, size=take)
            picked.append(values[idx])
    if not picked:
        return np.empty(0, dtype=np.float64)
    sample = np.concatenate(picked).astype(np.float64, copy=False)
    if sample.size > size:
        sample = sample[rng.permutation(sample.size)[:size]]
    return sample


def _split_edges(lo, hi, n):
    """Return ``n + 1`` increasing floats spanning ``[lo, hi]``.

    Returns ``None`` if the interval cannot be split into at least two
    distinct edges.
    """
    if not (hi > lo):
        return None
    lo = float(lo)
    hi = float(hi)
    if np.isfinite(hi - lo):
        edges = lo + (hi - lo) * np.arange(n + 1) / n
        edges[0] = lo
        edges[-1] = hi
    else:  # pragma: no cover - only for astronomically large ranges
        edges = np.linspace(lo, hi, n + 1)
    edges = np.unique(edges)
    if edges.size < 2 or not np.all(np.isfinite(edges)):
        return None
    return edges


def _scan(data, mask, edges, chunk):
    """Count reference cells per interval and record value extremes."""
    nbins = edges.size - 1
    counts = np.zeros(nbins, dtype=np.int64)
    bmin = np.full(nbins, np.inf)
    bmax = np.full(nbins, -np.inf)
    for block, bmask in _iter_blocks(data, mask, chunk):
        values = _valid_values(block, bmask)
        if not values.size:
            continue
        idx = np.searchsorted(edges, values, side="right")
        idx -= 1
        np.clip(idx, 0, nbins - 1, out=idx)
        counts += np.bincount(idx, minlength=nbins)

        # Per interval extremes.  Sorting the (small) block by bin index and
        # reducing contiguous groups keeps this vectorised and bound by the
        # block size.
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        sorted_values = values[order]
        starts = np.flatnonzero(
            np.concatenate(([True], sorted_idx[1:] != sorted_idx[:-1]))
        )
        bins = sorted_idx[starts]
        mins = np.minimum.reduceat(sorted_values, starts)
        maxs = np.maximum.reduceat(sorted_values, starts)
        bmin[bins] = np.minimum(bmin[bins], mins)
        bmax[bins] = np.maximum(bmax[bins], maxs)
    return counts, bmin, bmax


def _count_values(data, mask, chunk, values):
    """Exact counts of *values* in the reference."""
    values = np.asarray(values, dtype=np.float64)
    k = values.size
    equal = np.zeros(k, dtype=np.int64)
    if k == 0:
        return equal
    for block, bmask in _iter_blocks(data, mask, chunk):
        block = _valid_values(block, bmask)
        if not block.size:
            continue
        pos = np.searchsorted(values, block.astype(np.float64, copy=False),
                              side="left")
        hist = np.bincount(pos, minlength=k + 1)
        equal += hist[:k]
    return equal


def _refine_edges(edges, counts, heavy, bmin, bmax, budget, sample, spikes):
    """Split the intervals flagged *heavy*, keeping within *budget* bins."""
    heavy_idx = np.flatnonzero(heavy)
    if heavy_idx.size == 0 or budget <= 0:
        return edges
    if budget >= heavy_idx.size:
        factor = min(_SPLIT, 1 + budget // heavy_idx.size)
        if factor < 2:
            return edges
    else:
        # Not enough budget to touch every heavy interval: refine the
        # heaviest ones first.
        order = np.argsort(counts[heavy_idx])[::-1][: int(budget)]
        heavy_idx = np.sort(heavy_idx[order])
        factor = 2

    # Group the sample values by the interval they fall in, so that split
    # points can be placed at rank quantiles rather than uniform offsets.
    sample_bins = np.empty(0, dtype=np.intp)
    sample_starts = np.empty(0, dtype=np.intp)
    if sample.size:
        sample = np.sort(sample)
        sidx = np.searchsorted(edges, sample, side="right") - 1
        np.clip(sidx, 0, edges.size - 2, out=sidx)
        starts = np.flatnonzero(
            np.concatenate(([True], sidx[1:] != sidx[:-1]))
        )
        sample_bins = sidx[starts]
        sample_starts = np.append(starts, sample.size)

    interiors = []
    for i in heavy_idx:
        sub = None
        if sample.size:
            j = np.searchsorted(sample_bins, i)
            if j < sample_bins.size and sample_bins[j] == i:
                vals = sample[sample_starts[j]:sample_starts[j + 1]]
                if vals.size >= 2 and vals[0] < vals[-1]:
                    probs = np.arange(1, factor) / factor
                    sub = np.quantile(vals, probs)
        if sub is None:
            sub = _split_edges(bmin[i], bmax[i], factor)
            if sub is not None:
                sub = sub[1:-1]
        if sub is not None and sub.size:
            interiors.append(np.asarray(sub, dtype=np.float64))

    pieces = [edges]
    pieces.extend(interiors)
    if spikes.size:
        # Isolate each tie group in a bin of its own: the neighbour of a
        # double is the closest representable value, so the interval only
        # holds the spike value itself.
        pieces.append(spikes)
        pieces.append(np.nextafter(spikes, np.inf))
        pieces.append(np.nextafter(spikes, -np.inf))
    new_edges = np.unique(np.concatenate(pieces))
    new_edges = new_edges[(new_edges >= edges[0]) & (new_edges <= edges[-1])]
    return new_edges


class ColorCalibration:
    """A rank based colour scale calibrated on a reference aggregate.

    Instances are normally created with :func:`calibrate` or
    :meth:`ColorCalibration.from_reference`.  Once built the reference raster
    is not needed anymore: the calibration only retains a compact description
    of the reference distribution and can be reused for any number of views.

    Mapping
    -------
    For every valid reference cell the *rank interval* of a value ``v`` is
    ``[P(<v), P(<=v)]`` where ``P`` is the probability in the reference.  A
    value is mapped to the midpoint of that rank interval: tied values
    therefore share the exact midpoint of their tied group, and other values
    are accurate to well within ``max_error``.  Values outside the reference
    range saturate at ``0`` and ``1``.  :meth:`inverse` is the monotone
    inverse of that mapping and returns finite values in the original units.

    Attributes
    ----------
    vmin, vmax : float
        Minimum and maximum valid value in the reference raster.
    ndata : int
        Number of valid cells in the reference raster.
    max_error : float
        Maximum reference probability mass of a retained interval.  The
        forward mapping is accurate to about half this many rank units and
        the inverse to about this many, whenever the bin budget allows the
        refinement to run to completion.
    max_interval_mass : float
        Largest fraction of the reference held by a retained interval that
        contains more than one distinct value.  It is at most ``max_error``
        unless the interval budget was exhausted (intervals holding a single
        repeated value cannot be split and are exact, so they are excluded).
    single_value : bool
        True when the reference contains a single distinct valid value.
    """

    __slots__ = (
        "_edges", "_pos_lo", "_pos_hi", "_bmin", "_bmax",
        "_inv_pos_lo", "_inv_pos_hi", "_inv_bmin", "_inv_bmax",
        "vmin", "vmax", "ndata", "max_error", "max_interval_mass",
        "single_value",
    )

    def __init__(self, edges, pos_lo, pos_hi, bmin, bmax, vmin, vmax, ndata,
                 max_error, single_value=False):
        self._edges = edges
        self._pos_lo = pos_lo
        self._pos_hi = pos_hi
        self._bmin = bmin
        self._bmax = bmax
        self.vmin = vmin
        self.vmax = vmax
        self.ndata = ndata
        self.max_error = max_error
        self.single_value = single_value
        self.max_interval_mass = 0.0
        # Inverse mapping tables; by default every retained interval is
        # assumed to be non-empty.  The builder narrows them down.
        self._inv_pos_lo = pos_lo
        self._inv_pos_hi = pos_hi
        self._inv_bmin = bmin
        self._inv_bmax = bmax

    # -- construction -----------------------------------------------------
    @classmethod
    def from_reference(cls, reference, max_error=0.01):
        """Build a calibration from a 2D reference aggregate.

        Parameters
        ----------
        reference : xarray.DataArray or ndarray
            NumPy backed 2D array.  Every valid cell contributes one sample
            to the reference distribution.  For unsigned integer data ``0``
            denotes an empty cell; for floating point data ``0`` is a valid
            value while ``NaN``, ``+/-inf`` and explicitly masked cells are
            skipped.
        max_error : float, optional
            Upper bound on the reference probability mass of a retained
            colour scale interval, and hence on the rank error of the
            mapping.  Default ``0.01``.

        Raises
        ------
        ValueError
            If the reference is not 2D, ``max_error`` is out of range, or
            the reference contains no valid cells.
        TypeError
            If the reference is not backed by a NumPy array (e.g. a dask
            array), or is not a DataArray / ndarray.
        """
        if not 0 < max_error < 1:
            raise ValueError("max_error must be in (0, 1)")

        if isinstance(reference, xr.DataArray):
            if reference.ndim != 2:
                raise ValueError("reference must be a 2D DataArray")
            data = reference.data
        elif isinstance(reference, np.ndarray):
            data = reference
            if data.ndim != 2:
                raise ValueError("reference must be 2D")
        else:
            raise TypeError(
                "reference must be a 2D xarray.DataArray or numpy.ndarray, "
                f"got {type(reference).__name__}"
            )

        mask = None
        if np.ma.isMaskedArray(data):
            mask = np.ma.getmaskarray(data)
            data = np.ma.getdata(data)
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "reference must be backed by a NumPy array, got "
                f"{type(data).__name__}"
            )

        vmin, vmax, ndata = _min_max_count(data, mask, _CHUNK)
        if ndata == 0:
            raise ValueError("reference aggregate contains no valid cells")
        vmin = float(vmin)
        vmax = float(vmax)
        ndata = int(ndata)

        if vmin == vmax:
            half = np.full(1, 0.5)
            return cls(
                np.array([vmin, vmax]),
                half, half.copy(),
                np.array([vmin]), np.array([vmax]),
                vmin, vmax, ndata, max_error, single_value=True,
            )

        sample = _sample_values(data, mask, _CHUNK, ndata)
        edges = _initial_edges(sample, vmin, vmax)
        if edges is None:  # pragma: no cover - degenerate float range
            raise ValueError("reference values cannot be separated")

        threshold = max_error * ndata
        spike_threshold = max(1.0, 0.5 * threshold)
        spikes = np.empty(0, dtype=np.float64)
        counts = bmin = bmax = None

        for _ in range(_MAX_PASSES):
            counts, bmin, bmax = _scan(data, mask, edges, _CHUNK)
            heavy = (counts > threshold) & (bmin < bmax)

            # Look for heavy tie groups among the candidates: extremes of
            # heavy intervals plus values that repeat a lot in the sample.
            candidates = [cand for cand in (bmin[heavy], bmax[heavy]) if cand.size]
            if sample.size:
                values, sample_counts = np.unique(sample, return_counts=True)
                loud = values[sample_counts > max(2.0, 0.25 * max_error * sample.size)]
                if loud.size:
                    candidates.append(loud)
            new_spikes = np.empty(0, dtype=np.float64)
            if candidates:
                cands = np.unique(np.concatenate(candidates))
                cands = cands[np.isfinite(cands)]
                if cands.size:
                    equal = _count_values(data, mask, _CHUNK, cands)
                    new_spikes = cands[equal > spike_threshold]

            if not heavy.any() and new_spikes.size == 0:
                break

            budget = _MAX_BINS - (edges.size - 1)
            combined_spikes = np.unique(np.concatenate([spikes, new_spikes]))
            new_edges = _refine_edges(edges, counts, heavy, bmin, bmax,
                                      budget, sample, combined_spikes)
            spikes = combined_spikes
            if new_edges.size <= edges.size:
                # No further refinement is possible (e.g. all heavy
                # intervals consist of a single repeated value).
                break
            edges = new_edges
            counts = None

        if counts is None:
            # The last refinement was not followed by a scan.
            counts, bmin, bmax = _scan(data, mask, edges, _CHUNK)

        empty = counts == 0
        lo_values = np.where(empty, edges[:-1], np.minimum(bmin, bmax))
        hi_values = np.where(empty, edges[1:], np.maximum(bmin, bmax))
        # Keep the value ranges ordered: they are disjoint by construction,
        # but float round-off can make neighbours touch or cross.
        lo_values = np.maximum.accumulate(lo_values)
        hi_values = np.maximum.accumulate(hi_values)
        hi_values = np.maximum(hi_values, lo_values)

        cum = np.cumsum(counts)
        pos_lo = (cum - counts) / ndata
        pos_hi = cum / ndata

        # The largest mass of an interval holding more than one distinct
        # value.  It is <= max_error whenever the refinement could run to
        # completion; heavier intervals are necessarily single-valued tie
        # groups, whose rank interval is still exact.
        multi = bmin < bmax
        max_mass = float(counts[multi].max() / ndata) if multi.any() else 0.0

        # For the inverse mapping only non-empty intervals matter: they tile
        # [0, 1] without gaps and their value ranges are real reference
        # values, so inverse() always returns a value the reference contains.
        nonempty = counts > 0
        instance = cls(edges, pos_lo, pos_hi, lo_values, hi_values, vmin, vmax,
                       ndata, max_error)
        instance.max_interval_mass = max_mass
        instance._inv_pos_lo = pos_lo[nonempty]
        instance._inv_pos_hi = pos_hi[nonempty]
        instance._inv_bmin = lo_values[nonempty]
        instance._inv_bmax = hi_values[nonempty]
        return instance

    # -- mapping ----------------------------------------------------------
    def _positions(self, values):
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        isnan = np.isnan(flat)
        safe = np.where(isnan, self.vmin, flat)
        idx = np.searchsorted(self._edges, safe, side="right")
        idx -= 1
        np.clip(idx, 0, self._pos_lo.size - 1, out=idx)
        pos = 0.5 * (self._pos_lo[idx] + self._pos_hi[idx])
        pos = np.where(safe < self.vmin, 0.0, pos)
        pos = np.where(safe > self.vmax, 1.0, pos)
        return np.where(isnan, np.nan, pos)

    def forward(self, values):
        """Map values in original units to colour scale positions in ``[0, 1]``.

        Values below :attr:`vmin` map to ``0``, values above :attr:`vmax`
        map to ``1`` and ``NaN`` maps to ``NaN``.  Infinities saturate at the
        corresponding end.
        """
        values = np.asarray(values)
        pos = self._positions(values).reshape(values.shape)
        return pos[()] if values.ndim == 0 else pos

    def inverse(self, positions):
        """Map colour scale positions in ``[0, 1]`` back to original units.

        The result is always a finite value in the original units.  Positions
        are clipped to ``[0, 1]``; the endpoints map to :attr:`vmin` and
        :attr:`vmax` exactly.  ``NaN`` positions map to :attr:`vmin`.
        """
        positions = np.asarray(positions)
        flat = np.asarray(positions, dtype=np.float64).reshape(-1)
        flat = np.where(np.isnan(flat), 0.0, flat)
        flat = np.clip(flat, 0.0, 1.0)
        if self.single_value:
            out = np.full(flat.shape, self.vmin)
        else:
            idx = np.searchsorted(self._inv_pos_hi, flat, side="left")
            np.clip(idx, 0, self._inv_pos_lo.size - 1, out=idx)
            lo = self._inv_pos_lo[idx]
            hi = self._inv_pos_hi[idx]
            bmin = self._inv_bmin[idx]
            bmax = self._inv_bmax[idx]
            width = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(width > 0, (flat - lo) / width, 0.0)
            frac = np.clip(frac, 0.0, 1.0)
            out = bmin + frac * (bmax - bmin)
            out = np.maximum(out, bmin)
            out = np.minimum(out, bmax)
        out = out.reshape(positions.shape)
        return out[()] if positions.ndim == 0 else out

    def __call__(self, values):
        """Alias for :meth:`forward`."""
        return self.forward(values)

    # -- introspection ----------------------------------------------------
    @property
    def nbins(self):
        """Number of value intervals retained by the calibration."""
        return self._pos_lo.size

    def rank_interval(self, values):
        """Approximate ``[P(<v), P(<=v)]`` rank intervals for *values*.

        The bounds are accurate to :attr:`max_error`.  This is mostly useful
        for diagnostics and tests.
        """
        values = np.asarray(values)
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        isnan = np.isnan(flat)
        safe = np.where(isnan, self.vmin, flat)
        idx = np.searchsorted(self._edges, safe, side="right")
        idx -= 1
        np.clip(idx, 0, self._pos_lo.size - 1, out=idx)
        below = safe < self.vmin
        above = safe > self.vmax
        lo = np.where(below, 0.0, self._pos_lo[idx])
        hi = np.where(above, 1.0, self._pos_hi[idx])
        lo = np.where(above, 1.0, lo)
        hi = np.where(below, 0.0, hi)
        lo = np.where(isnan, np.nan, lo).reshape(values.shape)
        hi = np.where(isnan, np.nan, hi).reshape(values.shape)
        return (lo[()], hi[()]) if values.ndim == 0 else (lo, hi)

    def as_normalize(self):
        """Return a Matplotlib ``Normalize`` implementing this mapping.

        The Matplotlib import is deferred so that Datashader keeps working
        without Matplotlib installed.
        """
        from ._mpl_normalize import CalibratedNormalize

        return CalibratedNormalize(self)

    def __repr__(self):
        return (
            f"{type(self).__name__}(ndata={self.ndata}, nbins={self.nbins}, "
            f"vmin={self.vmin!r}, vmax={self.vmax!r})"
        )


def _initial_edges(sample, vmin, vmax):
    """Seed the partition from quantiles of a reference sample."""
    if sample.size >= 2:
        probs = np.linspace(0.0, 1.0, _SPLIT + 1)
        quantiles = np.quantile(sample, probs)
        edges = np.concatenate(([vmin], quantiles, [vmax]))
        edges = np.unique(edges)
        if edges.size >= 2:
            return edges
    return _split_edges(vmin, vmax, _SPLIT)


def calibrate(reference, max_error=0.01):
    """Build a :class:`ColorCalibration` from a 2D reference aggregate.

    See :meth:`ColorCalibration.from_reference` for details.
    """
    return ColorCalibration.from_reference(reference, max_error=max_error)
