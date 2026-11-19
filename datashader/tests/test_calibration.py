"""Regression tests for the explicit colour calibration mode.

The tests only use locally generated data (no downloads), so they can be
re-run offline.  From the repository root::

    python -m pytest datashader/tests/test_calibration.py -q

To run just the matplotlib / Agg integration tests::

    python -m pytest datashader/tests/test_calibration.py -q \\
        -k "dsshow or parity"

To run only the construction working memory checks::

    python -m pytest datashader/tests/test_calibration.py -q \
        -k "working_memory"

The memory checks use :mod:`tracemalloc`, which accounts for the NumPy
allocations made through Python's allocator, including the large array
buffers allocated while tracing is running.  No temporary files are written
by the implementation.
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

import datashader.transfer_functions as tf


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ranks(reference):
    """Exact rank intervals ``[P(<v), P(<=v)]`` keyed by unique value."""
    data = np.asarray(reference).ravel()
    if data.dtype.kind == "u":
        data = data[data != 0]
    else:
        data = data[np.isfinite(data)]
    values, counts = np.unique(data, return_counts=True)
    cum = np.cumsum(counts)
    n = cum[-1]
    return values, (cum - counts) / n, cum / n


def _assert_forward_accurate(cal, reference, tol=0.02):
    values, lo, hi = _ranks(reference)
    pos = cal.forward(values)
    # Values outside the reference range saturate.
    inside = (values >= cal.vmin) & (values <= cal.vmax)
    # The target of a value is its rank; for a tied value any rank inside its
    # own tied group is acceptable, so the distance to the rank interval
    # bounds the error.
    gap = np.maximum(lo - pos, pos - hi)
    assert np.all(np.maximum(gap[inside], 0) <= tol), (
        values[inside][gap[inside] > tol][:5], np.max(gap[inside]))
    assert np.all(pos[values < cal.vmin] == 0)
    assert np.all(pos[values > cal.vmax] == 1)

    # Values sharing a tied group all get the same position, its midpoint.
    order = np.argsort(values)
    tied = np.flatnonzero(hi[order] - lo[order] > 1.5 * tol)
    for k in tied[:20]:
        mid = 0.5 * (lo[order][k] + hi[order][k])
        assert pos[order][k] == pytest.approx(mid, abs=0.02)


def _assert_inverse_accurate(cal, reference, n=2001, tol=0.02):
    values, lo, hi = _ranks(reference)
    positions = np.linspace(0, 1, n)
    out = cal.inverse(positions)
    assert np.all(np.isfinite(out))
    assert np.all(out >= cal.vmin) and np.all(out <= cal.vmax)
    # Rank interval of the returned value must be within `tol` of position.
    k = np.clip(np.searchsorted(values, out, side="left"), 0, values.size - 1)
    equal = values[k] == out
    below = np.where(k > 0, hi[np.clip(k - 1, 0, None)], 0.0)
    rlo = np.where(equal, lo[k], below)
    rhi = np.where(equal, hi[k], rlo)
    gap = np.maximum(rlo - positions, positions - rhi)
    assert np.max(np.maximum(gap, 0)) <= tol, np.max(gap)


@pytest.fixture
def skewed_reference():
    rng = np.random.default_rng(42)
    ref = rng.pareto(1.2, size=(400, 500))
    ref[0, :10] = np.nan
    return ref


# ---------------------------------------------------------------------------
# construction and validation behaviour
# ---------------------------------------------------------------------------

def test_reference_is_not_retained():
    ref = np.arange(64.0).reshape(8, 8)
    cal = tf.calibrate(xr.DataArray(ref))
    lo, hi = cal.rank_interval(np.array([0.0, 63.0]))
    npt.assert_allclose(lo, [0.0, 63 / 64])
    npt.assert_allclose(hi, [1 / 64, 1.0])
    # Mutating (or dropping) the reference must not affect the calibration.
    ref[:] = -1
    del ref
    gc.collect()
    lo2, hi2 = cal.rank_interval(np.array([0.0, 63.0]))
    npt.assert_allclose(lo2, lo)
    npt.assert_allclose(hi2, hi)
    npt.assert_allclose(cal.inverse([0.0, 1.0]), [0.0, 63.0])


def test_invalid_modes():
    ok = tf.calibrate(np.arange(16.0).reshape(4, 4))

    with pytest.raises(TypeError):
        tf.shade(np.arange(4.0), calibration=ok)

    with pytest.raises(TypeError):
        tf.shade(xr.DataArray(np.arange(4.0).reshape(2, 2)), calibration="nope")

    three_d = xr.DataArray(np.zeros((2, 2, 2)), dims=["y", "x", "cat"],
                           coords={"y": [0, 1], "x": [0, 1], "cat": ["a", "b"]})
    with pytest.raises(ValueError):
        tf.shade(three_d, calibration=ok)

    two_d = xr.DataArray(np.arange(4.0).reshape(2, 2))
    with pytest.raises(ValueError):
        tf.shade(two_d, calibration=ok, color_key={"a": "red"})

    with pytest.raises(ValueError):
        tf.calibrate(np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        tf.calibrate(np.arange(4.0).reshape(2, 2), max_error=0.0)
    with pytest.raises(TypeError):
        tf.calibrate([1.0, 2.0])


def test_all_empty_reference():
    empty_uint = np.zeros((5, 6), dtype="uint32")
    with pytest.raises(ValueError, match="no valid cells"):
        tf.calibrate(xr.DataArray(empty_uint))

    empty_float = np.full((5, 6), np.nan)
    with pytest.raises(ValueError, match="no valid cells"):
        tf.calibrate(empty_float)

    all_inf = np.full((3, 3), np.inf)
    with pytest.raises(ValueError, match="no valid cells"):
        tf.calibrate(all_inf)


def test_valid_cell_rules():
    float_ref = np.arange(25.0).reshape(5, 5)
    float_ref[0, 0] = 0.0        # valid: zero is a value for floats
    float_ref[0, 1] = np.nan     # skipped
    float_ref[0, 2] = np.inf     # skipped
    float_ref[0, 3] = -np.inf    # skipped
    cal = tf.calibrate(xr.DataArray(float_ref))
    assert cal.ndata == 22
    assert cal.vmin == 0.0
    assert cal.vmax == 24.0

    uint_ref = np.arange(25, dtype="uint32").reshape(5, 5)
    cal_u = tf.calibrate(xr.DataArray(uint_ref))
    assert cal_u.ndata == 24
    assert cal_u.vmin == 1.0

    masked = np.ma.array(np.arange(16.0).reshape(4, 4),
                         mask=np.zeros((4, 4), dtype=bool))
    masked.mask[0, 0] = True
    masked.mask[1, 1] = True
    cal_m = tf.calibrate(xr.DataArray(masked))
    assert cal_m.ndata == 14
    assert cal_m.vmin == 1.0


def test_single_value_reference():
    cal = tf.calibrate(xr.DataArray(np.full((6, 7), 3.5)))
    assert cal.single_value
    npt.assert_allclose(cal.forward([0.0, 3.5, 100.0, -100.0]),
                        [0.0, 0.5, 1.0, 0.0])
    npt.assert_allclose(cal.inverse([0.0, 0.25, 0.5, 0.75, 1.0]),
                        [3.5] * 5)
    assert cal.inverse(0.3) == pytest.approx(3.5)

    # The same value drawn by shade is the middle of the colormap.
    img = tf.shade(xr.DataArray(np.array([[3.5]])), calibration=cal,
                   cmap="black", alpha=255)
    rgba = img.data.view(np.uint8).reshape(-1, 4)
    npt.assert_array_equal(rgba[0, :3], [0, 0, 0])


# ---------------------------------------------------------------------------
# rank accuracy
# ---------------------------------------------------------------------------

def test_forward_rank_accuracy_skewed(skewed_reference):
    cal = tf.calibrate(xr.DataArray(skewed_reference))
    _assert_forward_accurate(cal, skewed_reference)
    _assert_inverse_accurate(cal, skewed_reference)
    # The refinement ran to completion, so no multi-valued interval is wider
    # than the requested rank error.
    assert cal.max_interval_mass <= cal.max_error


@pytest.mark.parametrize("max_error", [0.02, 0.01, 0.002])
def test_forward_rank_accuracy_parameterised(max_error):
    rng = np.random.default_rng(7)
    ref = rng.standard_normal((300, 300))
    cal = tf.calibrate(ref, max_error=max_error)
    _assert_forward_accurate(cal, ref, tol=max_error + 1e-9)


def test_saturation_outside_reference_range():
    ref = np.linspace(1.0, 20.0, 400).reshape(20, 20)
    cal = tf.calibrate(ref)
    npt.assert_allclose(cal.forward([-5.0, 0.0, 21.0, 1e9]), [0, 0, 1, 1])


def test_monotone_forward_and_inverse():
    rng = np.random.default_rng(11)
    ref = np.concatenate([rng.exponential(1.0, 40000) * 1e-3,
                          rng.exponential(1.0, 10000) * 1e3,
                          np.zeros(5000), np.full(5000, 2.5)])
    cal = tf.calibrate(ref.reshape(-1, 1))
    grid = np.linspace(cal.vmin, cal.vmax, 5001)
    assert np.all(np.diff(cal.forward(grid)) >= 0)
    assert np.all(np.diff(cal.inverse(np.linspace(0, 1, 5001))) >= 0)


def test_ties_and_missing_values():
    rng = np.random.default_rng(3)
    # A large tie group plus a skewed continuous tail and missing cells.
    ref = np.where(rng.random((300, 300)) < 0.4, 5.0, rng.pareto(2.0, (300, 300)))
    ref[ref > 50] = np.nan
    ref[0, 0] = np.inf
    cal = tf.calibrate(ref)

    # Tied values share the exact midpoint of the tied group.
    values, lo, hi = _ranks(ref)
    k = int(np.flatnonzero(values == 5.0)[0])
    assert cal.forward(5.0) == pytest.approx(0.5 * (lo[k] + hi[k]), abs=0.005)
    # The rank interval of the tie is recovered accurately.
    rlo, rhi = cal.rank_interval(5.0)
    assert rlo == pytest.approx(lo[k], abs=0.02)
    assert rhi == pytest.approx(hi[k], abs=0.02)
    _assert_forward_accurate(cal, ref)
    _assert_inverse_accurate(cal, ref)
    npt.assert_array_equal(cal.forward([np.nan, np.inf, -np.inf]), [np.nan, 1.0, 0.0])


def test_inverse_endpoints():
    rng = np.random.default_rng(5)
    ref = rng.lognormal(size=(100, 100))
    cal = tf.calibrate(ref)
    assert cal.inverse(0.0) == cal.vmin
    assert cal.inverse(1.0) == cal.vmax
    # Clipping outside [0, 1]
    assert cal.inverse(-3.0) == cal.vmin
    assert cal.inverse(4.0) == cal.vmax


# ---------------------------------------------------------------------------
# shade integration
# ---------------------------------------------------------------------------

def test_shade_ignores_window_min_and_max():
    cmap = pytest.importorskip("matplotlib").colormaps["viridis"]
    ref = np.arange(1, 101, dtype="uint32").reshape(10, 10)
    cal = tf.calibrate(xr.DataArray(ref))

    window_small = xr.DataArray(np.array([[10, 20], [30, 40]], dtype="uint32"))
    window_large = xr.DataArray(np.array([[10, 20], [30, 60000]], dtype="uint32"))
    img_small = tf.shade(window_small, calibration=cal, cmap=cmap)
    img_large = tf.shade(window_large, calibration=cal, cmap=cmap)

    # The three values shared by both windows render identically; only the
    # out-of-range 60000 differs (it saturates at the top of the scale).
    npt.assert_array_equal(img_small.data.ravel()[:3], img_large.data.ravel()[:3])
    assert img_large.data[1, 1] != img_small.data[1, 1]
    assert cal.forward(60000) == 1.0

    # And `span`/`how` have no influence either.
    img_span = tf.shade(window_small, calibration=cal, cmap=cmap,
                        span=(0, 100000), how="linear")
    npt.assert_array_equal(img_small.data, img_span.data)

    # Same value -> same colour, regardless of what else is in the window.
    col = tf.shade(xr.DataArray(np.array([[10]], dtype="uint32")),
                   calibration=cal, cmap=cmap)
    assert img_small.data[0, 0] == col.data[0, 0]


def test_shade_uncalibrated_behaviour_unchanged():
    rng = np.random.default_rng(0)
    agg = xr.DataArray(rng.poisson(3, (30, 40)).astype("uint32"))
    # Existing calls do not accept or need a calibration.
    npt.assert_array_equal(tf.shade(agg, how="eq_hist").data,
                           tf.shade(agg, calibration=None, how="eq_hist").data)
    # A calibration is opt-in: without it, the window autoscales.
    a = tf.shade(agg, how="linear", span=None)
    b = tf.shade(agg[::2], how="linear", span=None)
    assert not np.array_equal(a.data, b.data[: a.shape[0] // 2 + 1])


def test_shade_empty_window_transparent():
    ref = np.arange(1, 50, dtype="uint32").reshape(7, 7)
    cal = tf.calibrate(ref)
    empty = xr.DataArray(np.zeros((4, 4), dtype="uint32"))
    img = tf.shade(empty, calibration=cal)
    npt.assert_array_equal(img.data, np.zeros((4, 4), dtype="uint32"))
    empty_f = xr.DataArray(np.full((4, 4), np.nan))
    img_f = tf.shade(empty_f, calibration=cal)
    npt.assert_array_equal(img_f.data, np.zeros((4, 4), dtype="uint32"))


@pytest.mark.parametrize("kind", ["mpl", "list"])
def test_shade_all_colormap_argument_kinds(kind):
    mpl = pytest.importorskip("matplotlib")
    cmap = {"mpl": mpl.colormaps["viridis"],
            "list": ["red", "yellow", "blue"]}[kind]
    ref = np.arange(1, 101, dtype="uint32").reshape(10, 10)
    cal = tf.calibrate(ref)
    agg = xr.DataArray(np.array([[1, 50, 100]], dtype="uint32"))
    img = tf.shade(agg, cmap=cmap, calibration=cal)
    rgba = img.data.view(np.uint8).reshape(1, 3, 4)
    # The three distinct values get three distinct colours.
    assert len({tuple(c) for c in rgba[0]}) == 3
    # Alpha is the requested fixed alpha for all non-empty cells.
    npt.assert_array_equal(rgba[0, :, 3], [255, 255, 255])
    # Empty cells stay transparent.
    empty = tf.shade(xr.DataArray(np.zeros((1, 1), dtype="uint32")),
                     cmap=cmap, calibration=cal)
    npt.assert_array_equal(empty.data.view(np.uint8).reshape(4),
                           np.zeros(4, np.uint8))


def test_shade_single_colour_alpha_ramp():
    ref = np.arange(1, 101, dtype="uint32").reshape(10, 10)
    cal = tf.calibrate(ref)
    agg = xr.DataArray(np.array([[1, 50, 100]], dtype="uint32"))
    rgba = tf.shade(agg, cmap="black", calibration=cal).data.view(np.uint8).reshape(3, 4)
    npt.assert_array_equal(rgba[:, :3], np.zeros((3, 3), np.uint8))
    # `min_alpha` .. `alpha` ramp follows the calibrated position.
    assert rgba[0, 3] < rgba[1, 3] < rgba[2, 3]
    assert rgba[0, 3] >= 40


# ---------------------------------------------------------------------------
# pipeline integration
# ---------------------------------------------------------------------------

def test_pipeline_calibrated_transform():
    import datashader as ds

    rng = np.random.default_rng(0)
    n = 5000
    df = {"x": rng.random(n), "y": rng.random(n)}

    import pandas as pd
    df = pd.DataFrame(df)

    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 1), y_range=(0, 1))
    agg = cvs.points(df, "x", "y", ds.count())
    cal = tf.calibrate(xr.DataArray(agg.data.astype("uint32")))

    def transform_fn(a):
        return a + 7

    pipeline = ds.Pipeline(df, ds.Point("x", "y"), ds.count(),
                           transform_fn=transform_fn, calibration=cal,
                           spread_fn=lambda img: img)
    img = pipeline((0, 1), (0, 1), 20, 20)
    expected = tf.shade(transform_fn(agg), calibration=cal)
    assert img.equals(expected)


def test_pipeline_calibrated_spread_still_runs():
    import pandas as pd
    import datashader as ds

    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.random(2000), "y": rng.random(2000)})
    cvs = ds.Canvas(plot_width=16, plot_height=16, x_range=(0, 1), y_range=(0, 1))
    agg = cvs.points(df, "x", "y", ds.count())
    cal = tf.calibrate(agg)

    calls = []

    def spread_fn(img):
        calls.append(img)
        return tf.spread(img, px=1)

    pipeline = ds.Pipeline(df, ds.Point("x", "y"), ds.count(),
                           calibration=cal, spread_fn=spread_fn)
    img = pipeline((0, 1), (0, 1), 16, 16)
    assert len(calls) == 1
    assert img.shape == (16, 16)


def test_pipeline_without_calibration_unchanged():
    import datashader as ds
    import pandas as pd

    df = pd.DataFrame({"x": np.array([0.0] * 10 + [1] * 10),
                       "y": np.array([0.0] * 5 + [1] * 5 + [0] * 5 + [1] * 5)})
    cvs = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))
    agg = cvs.points(df, "x", "y", ds.count())
    pipeline = ds.Pipeline(df, ds.Point("x", "y"))
    assert pipeline((0, 1), (0, 1), 2, 2).equals(tf.shade(agg))


# ---------------------------------------------------------------------------
# matplotlib integration
# ---------------------------------------------------------------------------

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

import datashader as ds  # noqa: E402
from datashader.mpl_ext import dsshow  # noqa: E402


def _draw(fig):
    fig.canvas.draw()


def test_dsshow_pixel_parity_with_shade():
    rng = np.random.default_rng(4)
    ref = np.where(rng.random((200, 200)) < 0.3, 2.0,
                   rng.exponential(3.0, (200, 200)))
    cal = tf.calibrate(ref)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    import pandas as pd
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    artist = dsshow(df, ds.Point("x", "y"), ax=ax, calibration=cal,
                    cmap=cm.viridis, alpha=1.0)

    values = np.array([[0.0, 0.5, 1.0, 2.0],
                       [3.0, 7.5, 20.0, 100.0]])
    binned = xr.DataArray(values, dims=["y", "x"],
                          coords={"y": np.arange(2), "x": np.arange(4)})
    artist.plot_width = 4
    artist.plot_height = 2
    rgba_artist = artist.shade(binned)

    img = tf.shade(binned, cmap=cm.viridis, calibration=cal, alpha=255)
    rgba_shade = img.data.view(np.uint8).reshape(2, 4, 4)
    npt.assert_array_equal(rgba_artist, rgba_shade)


@pytest.mark.parametrize("dtype", ["uint32", "float64"])
def test_dsshow_pixel_parity_default_opacity(dtype):
    """With no spreading and default opacity, shade and dsshow agree pixel for
    pixel, including the 8 bit RGBA channels."""
    rng = np.random.default_rng(8)
    if dtype == "uint32":
        reference = rng.poisson(4, (120, 120)).astype("uint32")
    else:
        reference = rng.exponential(2.0, (120, 120))
    cal = tf.calibrate(xr.DataArray(reference))

    import pandas as pd
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    artist = dsshow(df, ds.Point("x", "y"), ax=ax, calibration=cal,
                    cmap=cm.viridis)

    values = np.arange(24, dtype="float64").reshape(4, 6)
    values[0, 0] = 0.0
    if dtype == "uint32":
        values = values.astype("uint32")
    else:
        values[0, 1] = np.nan
        values[0, 2] = 500.0     # saturates at the top of the scale
        values[0, 3] = -1.0      # saturates at the bottom of the scale
    binned = xr.DataArray(values, dims=["y", "x"],
                          coords={"y": np.arange(4), "x": np.arange(6)})
    artist.plot_width = 6
    artist.plot_height = 4
    rgba_artist = artist.shade(binned)

    img = tf.shade(binned, cmap=cm.viridis, calibration=cal, alpha=255)
    rgba_shade = img.data.view(np.uint8).reshape(4, 6, 4)
    npt.assert_array_equal(rgba_artist, rgba_shade)
    if dtype == "uint32":
        # An unsigned 0 is an empty cell: fully transparent in both paths.
        npt.assert_array_equal(rgba_artist[0, 0], np.zeros(4, np.uint8))
    else:
        # A float 0 is a valid value: opaque, at the bottom of the scale.
        assert rgba_artist[0, 0, 3] == 255
        # NaN is missing: fully transparent in both paths.
        npt.assert_array_equal(rgba_artist[0, 1], np.zeros(4, np.uint8))


def test_dsshow_calibrated_windows_and_colorbar():
    rng = np.random.default_rng(10)
    n = 4000
    import pandas as pd
    x = rng.random(n)
    y = rng.random(n)
    df = pd.DataFrame({"x": x, "y": y})

    # Reference from the full extent, on a coarse grid.
    ref_canvas = ds.Canvas(plot_width=60, plot_height=60,
                           x_range=(0, 1), y_range=(0, 1))
    ref_agg = ref_canvas.points(df, "x", "y", ds.count())
    cal = tf.calibrate(ref_agg)

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    artist = dsshow(df, ds.Point("x", "y"), ax=ax, calibration=cal,
                    cmap=cm.viridis, alpha=1.0)
    _draw(fig)
    cbar = fig.colorbar(artist)
    _draw(fig)

    # Colors of any given count value do not change when the window changes.
    def check_window():
        data = np.asarray(artist.get_ds_data().data)
        rgba = artist.get_array()
        present = np.unique(data)
        present = present[present > 0]
        assert present.size
        for v in present[:8]:
            expected = np.array(cm.viridis(cal.forward(v), bytes=True))
            ys, xs = np.nonzero(data == v)
            for i, j in zip(ys[:3], xs[:3]):
                npt.assert_array_equal(rgba[i, j], expected)
        return present

    check_window()
    before_agg = np.array(artist.get_ds_data().data, copy=True)

    ax.set_xlim(0.2, 0.6)
    ax.set_ylim(0.2, 0.6)
    _draw(fig)
    check_window()

    # The zoom changed the aggregate, but not the colours assigned to values.
    assert not np.array_equal(before_agg, np.asarray(artist.get_ds_data().data))

    # The colorbar inverse mapping stays tied to the calibration.
    assert artist.norm.inverse(0.0) == pytest.approx(cal.vmin)
    assert artist.norm.inverse(1.0) == pytest.approx(cal.vmax)
    _draw(fig)
    assert cbar.mappable is artist
    boundaries = np.asarray(cbar._boundaries)
    assert np.all(np.diff(boundaries) >= 0)
    # The colorbar spans exactly the reference range in inverse units.
    assert boundaries.min() == pytest.approx(cal.vmin)
    assert boundaries.max() == pytest.approx(cal.vmax)
    # Colorbar positions map through the calibration's inverse function.
    inside = np.linspace(0.05, 0.95, 9)
    npt.assert_allclose(artist.norm.inverse(inside),
                        cal.inverse(inside), rtol=1e-12)


def test_dsshow_calibrated_empty_window_and_cursor():
    rng = np.random.default_rng(12)
    import pandas as pd
    n = 3000
    df = pd.DataFrame({"x": rng.random(n), "y": rng.random(n)})

    cvs = ds.Canvas(plot_width=50, plot_height=50, x_range=(0, 1), y_range=(0, 1))
    cal = tf.calibrate(cvs.points(df, "x", "y", ds.count()))

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    artist = dsshow(df, ds.Point("x", "y"), ax=ax, calibration=cal)
    _draw(fig)

    # Empty window: re-aggregated (all zeros) and fully transparent.
    ax.set_xlim(5, 6)
    ax.set_ylim(5, 6)
    _draw(fig)
    binned = artist.get_ds_data()
    assert np.all(np.asarray(binned.data) == 0)
    rgba = artist.get_array()
    assert rgba.shape[:2] == binned.shape
    assert np.all(rgba[..., 3] == 0)

    # Cursor queries report the raw aggregate of the current window.
    class Event:
        xdata = 5.5
        ydata = 5.5

    result = artist.get_cursor_data(Event)
    assert result == 0

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _draw(fig)
    binned = artist.get_ds_data()
    Event.xdata = 0.5
    Event.ydata = 0.5
    cursor = artist.get_cursor_data(Event)
    assert cursor == np.asarray(binned.data)[binned.shape[0] // 2,
                                             binned.shape[1] // 2]


def test_dsshow_calibrated_single_value_reference():
    """A single-valued reference draws every value at the middle of the scale."""
    cal = tf.calibrate(np.full((10, 10), 3.0))
    import pandas as pd
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    artist = dsshow(df, ds.Point("x", "y"), ax=ax, calibration=cal,
                    cmap=cm.viridis)
    _draw(fig)
    # Drawing a colorbar must not blow up on a degenerate range.
    with pytest.warns(UserWarning, match="identical low and high"):
        cbar = fig.colorbar(artist)
        _draw(fig)
    assert artist.norm.vmin == artist.norm.vmax == 3.0
    npt.assert_array_equal(artist.norm.inverse([0.0, 0.5, 1.0]), [3.0, 3.0, 3.0])
    assert np.all(np.asarray(cbar._boundaries) == 3.0)


def test_dsshow_calibrated_rejects_categorical():
    import pandas as pd
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0],
                       "c": ["a", "b"]})
    cal = tf.calibrate(np.arange(16.0).reshape(4, 4))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError):
        dsshow(df, ds.Point("x", "y"), ds.count_cat("c"), ax=ax,
               calibration=cal)
    plt.close(fig)


def test_dsshow_calibrated_pan_and_zoom_stability():
    import pandas as pd
    rng = np.random.default_rng(21)
    n = 20000
    df = pd.DataFrame({"x": rng.random(n), "y": rng.random(n)})
    cvs = ds.Canvas(plot_width=40, plot_height=40, x_range=(0, 1), y_range=(0, 1))
    cal = tf.calibrate(cvs.points(df, "x", "y", ds.count()))

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    artist = dsshow(df, ds.Point("x", "y"), ax=ax, calibration=cal,
                    cmap=cm.viridis, alpha=1.0)
    _draw(fig)

    seen = {}
    for xlim, ylim in [((0, 1), (0, 1)), ((0.3, 0.7), (0.3, 0.7)),
                       ((0.45, 0.55), (0.45, 0.55)), ((0, 1), (0, 1))]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _draw(fig)
        rgba = artist.get_array()
        data = np.asarray(artist.get_ds_data().data)
        for target in (1, 2, 5):
            idx = np.nonzero(data == target)
            if len(idx[0]) == 0:
                continue
            color = rgba[idx[0][0], idx[1][0]].copy()
            if target in seen:
                npt.assert_array_equal(color, seen[target])
            seen[target] = color


# ---------------------------------------------------------------------------
# construction working memory
# ---------------------------------------------------------------------------

def _measure_peak(ref):
    gc.collect()
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        cal = tf.calibrate(ref)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return cal, (peak - baseline)


@pytest.mark.parametrize("shape", [(500, 500), (1000, 1000), (2000, 2000)])
def test_build_working_memory_is_bounded(shape):
    rng = np.random.default_rng(0)
    # Skewed reference with a heavy tie group and missing cells.
    ref = rng.pareto(1.2, size=shape)
    flat = ref.reshape(-1)
    flat[::1000] = 5.0
    flat[::3000] = np.nan

    cal, extra = _measure_peak(ref)
    limit = 8 * 1024 * 1024
    assert extra < limit, f"{extra / 2**20:.2f} MiB of extra memory"
    assert cal.nbins > 1


def test_build_working_memory_independent_of_reference_size():
    rng = np.random.default_rng(1)
    small = rng.lognormal(size=(400, 400))
    large = rng.lognormal(size=(1600, 1600))
    _, mem_small = _measure_peak(small)
    _, mem_large = _measure_peak(large)
    # A 16x larger reference must not need significantly more working memory.
    assert mem_large < max(mem_small * 1.5, mem_small + 2**20)
    assert mem_large < 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# offline re-run helper
# ---------------------------------------------------------------------------

def test_offline_reproduction(tmp_path=None):
    """Build a calibration twice from the same data: fully deterministic."""
    rng = np.random.default_rng(123)
    ref = rng.exponential(2.0, (250, 250))
    ref[0, 0] = np.nan
    a = tf.calibrate(ref)
    b = tf.calibrate(ref.copy())
    grid = np.linspace(a.vmin, a.vmax, 1000)
    npt.assert_array_equal(a.forward(grid), b.forward(grid))
    npt.assert_array_equal(a.inverse(np.linspace(0, 1, 1000)),
                           b.inverse(np.linspace(0, 1, 1000)))
