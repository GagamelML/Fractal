"""
Colorizer functions.  Each has signature:
    colorize(counts, max_iter, *, invert=False, force_bg=None,
             interior_color=..., bounds=None, **extra) -> PIL.Image (RGB)

Universal parameters
--------------------
* ``invert``    – flip the gradient direction (t -> 1 - t).
* ``force_bg``  – None, "black", or "white".  Interior pixels are forced
                  to that colour and the colorizer output is smoothly
                  blended toward it near the fractal boundary.
* ``bounds``    – explicit (lo, hi) percentile bounds for normalisation.
"""
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _hsv_to_rgb(hue, sat, val):
    h6 = hue * np.float32(6.0)
    sector = h6.astype(np.int8) % 6
    f = h6 - np.floor(h6)
    del h6
    p = val * (np.float32(1) - sat)
    q = val * (np.float32(1) - sat * f)
    t = val * (np.float32(1) - sat * (np.float32(1) - f))
    del f
    rgb = np.zeros(hue.shape + (3,), dtype=np.float32)
    for idx, (v1, v2, v3) in enumerate(
        [(val, t, p), (q, val, p), (p, val, t),
         (p, q, val), (t, p, val), (val, p, q)]
    ):
        m = sector == idx
        rgb[m, 0] = v1[m]
        rgb[m, 1] = v2[m]
        rgb[m, 2] = v3[m]
    return (rgb * 255).astype(np.uint8)


def _normalize(counts, max_iter, plow=1.0, phigh=99.0, bounds=None):
    esc = counts < max_iter
    if not esc.any():
        return None, esc
    vals = counts[esc].astype(np.float32)
    if bounds is not None:
        lo, hi = bounds
    else:
        lo = float(np.percentile(vals, plow))
        hi = float(np.percentile(vals, phigh))
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((vals - np.float32(lo)) / np.float32(hi - lo), 0.0, 1.0)
    return t, esc


def _calc_bounds(counts, max_iter, plow=1.0, phigh=99.0):
    esc = counts < max_iter
    if not esc.any():
        return None
    vals = counts[esc].astype(np.float64)
    lo = np.percentile(vals, plow)
    hi = np.percentile(vals, phigh)
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def _resolve_bg(force_bg, interior_color):
    if force_bg == "black":
        return (0, 0, 0)
    if force_bg == "white":
        return (255, 255, 255)
    return tuple(interior_color)


def _apply_forced_bg(rgb, counts, max_iter, t, esc, force_bg, blend_width=0.12):
    """Force interior colour and softly blend near-boundary pixels toward it."""
    if force_bg is None:
        return rgb
    bg_rgb = np.array(_resolve_bg(force_bg, (0, 0, 0)), dtype=np.float32)
    rgb[counts >= max_iter] = bg_rgb.astype(np.uint8)
    if t is not None:
        bt = np.clip(t / np.float32(blend_width), 0.0, 1.0)
        arr = rgb[esc].astype(np.float32)
        arr = bg_rgb * (1.0 - bt[:, None]) + arr * bt[:, None]
        rgb[esc] = arr.astype(np.uint8)
    return rgb


def _gradient(t, stops):
    """Piecewise-linear RGB gradient over t in [0, 1]."""
    positions = np.array([s[0] for s in stops], dtype=np.float32)
    colors    = np.array([s[1] for s in stops], dtype=np.float32)
    flat = t.ravel()
    out  = np.zeros((flat.shape[0], 3), dtype=np.float32)
    for i in range(len(stops) - 1):
        p0, p1 = positions[i], positions[i + 1]
        span = max(p1 - p0, 1e-9)
        mask = (flat >= p0) & (flat <= p1)
        if mask.any():
            local = (flat[mask] - p0) / span
            out[mask] = (colors[i] * (1.0 - local[:, None])
                         + colors[i + 1] * local[:, None])
    out[flat > positions[-1]] = colors[-1]
    return out.reshape(t.shape + (3,)).astype(np.uint8)


# ---------------------------------------------------------------------------
#  Colorizers
# ---------------------------------------------------------------------------
def greyscale(counts, max_iter, *, invert=False, force_bg=None,
              interior_color=(0, 0, 0), bounds=None):
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[counts >= max_iter] = _resolve_bg(force_bg, interior_color)
    t, esc = _normalize(counts, max_iter, bounds=bounds)
    if t is not None:
        if invert: t = 1.0 - t
        g = (t * 255).astype(np.uint8)
        rgb[esc, 0] = g; rgb[esc, 1] = g; rgb[esc, 2] = g
    rgb = _apply_forced_bg(rgb, counts, max_iter, t, esc, force_bg)
    return Image.fromarray(rgb, mode='RGB')


def hsv_cyclic(counts, max_iter, *, cycles=3, saturation=0.9, value=0.85,
               reverse=True, invert=False, force_bg=None,
               interior_color=(15, 5, 30), bounds=None):
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[counts >= max_iter] = _resolve_bg(force_bg, interior_color)
    t, esc = _normalize(counts, max_iter, bounds=bounds)
    if t is not None:
        if invert: t = 1.0 - t
        hue = (t * cycles) % 1.0
        if reverse: hue = (1.0 - hue) % 1.0
        rgb[esc] = _hsv_to_rgb(hue, np.full_like(hue, saturation),
                               np.full_like(hue, value))
    rgb = _apply_forced_bg(rgb, counts, max_iter, t, esc, force_bg)
    return Image.fromarray(rgb, mode='RGB')


def twilight(counts, max_iter, *, invert=False, force_bg=None,
             interior_color=(30, 15, 40), bounds=None):
    from matplotlib import colormaps
    cmap = colormaps['twilight_shifted']
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[counts >= max_iter] = _resolve_bg(force_bg, interior_color)
    t, esc = _normalize(counts, max_iter, bounds=bounds)
    if t is not None:
        if invert: t = 1.0 - t
        rgb[esc] = (cmap(t)[:, :3] * 255).astype(np.uint8)
    rgb = _apply_forced_bg(rgb, counts, max_iter, t, esc, force_bg)
    return Image.fromarray(rgb, mode='RGB')


def log_fire(counts, max_iter, *, gamma=1.8, intensity=2.5, blue_suppress=0.4,
             invert=False, force_bg=None,
             interior_color=(40, 5, 0), bounds=None):
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[counts >= max_iter] = _resolve_bg(force_bg, interior_color)
    esc = counts < max_iter
    t_out = None
    if esc.any():
        raw = counts[esc].astype(np.float32) + 1
        log_val = np.log(raw)
        del raw
        if bounds is not None:
            lo = np.float32(np.log(bounds[0] + 1))
            hi = np.float32(np.log(bounds[1] + 1))
        else:
            lo = np.float32(np.percentile(log_val, 1.0))
            hi = np.float32(np.percentile(log_val, 99.0))
        if hi <= lo: hi = lo + 1.0
        t_lin = np.clip((log_val - lo) / (hi - lo), 0.0, 1.0)
        del log_val
        if invert: t_lin = 1.0 - t_lin
        t_out = t_lin
        t = t_lin ** np.float32(gamma)
        r = np.clip(t * np.float32(intensity), 0, 1)
        g = np.clip(t * np.float32(intensity) - 1, 0, 1)
        b = np.clip(t * np.float32(intensity) - 2, 0, 1) * np.float32(blue_suppress)
        del t
        rgb[esc, 0] = (r * 255).astype(np.uint8)
        rgb[esc, 1] = (g * 255).astype(np.uint8)
        rgb[esc, 2] = (b * 255).astype(np.uint8)
    rgb = _apply_forced_bg(rgb, counts, max_iter, t_out, esc, force_bg)
    return Image.fromarray(rgb, mode='RGB')


def black_blue_gold_white(counts, max_iter, *, invert=False, force_bg=None,
                          interior_color=(0, 0, 0), bounds=None):
    """Deep galactic palette: black -> dark blue -> gold -> white."""
    stops = [(0.00, (0, 0, 0)),
             (0.35, (15, 40, 110)),
             (0.70, (220, 180, 40)),
             (1.00, (255, 255, 255))]
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[counts >= max_iter] = _resolve_bg(force_bg, interior_color)
    t, esc = _normalize(counts, max_iter, bounds=bounds)
    if t is not None:
        if invert: t = 1.0 - t
        rgb[esc] = _gradient(t, stops)
    rgb = _apply_forced_bg(rgb, counts, max_iter, t, esc, force_bg)
    return Image.fromarray(rgb, mode='RGB')


def red_white_black(counts, max_iter, *, invert=False, force_bg=None,
                    interior_color=(20, 20, 20), bounds=None):
    """High-contrast tricolour: deep red -> white -> near-black."""
    stops = [(0.00, (180, 15, 15)),
             (0.50, (255, 255, 255)),
             (1.00, (10, 10, 10))]
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[counts >= max_iter] = _resolve_bg(force_bg, interior_color)
    t, esc = _normalize(counts, max_iter, bounds=bounds)
    if t is not None:
        if invert: t = 1.0 - t
        rgb[esc] = _gradient(t, stops)
    rgb = _apply_forced_bg(rgb, counts, max_iter, t, esc, force_bg)
    return Image.fromarray(rgb, mode='RGB')


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------
COLORIZERS = {
    "greyscale":              greyscale,
    "hsv_cyclic":             hsv_cyclic,
    "twilight":               twilight,
    "log_fire":               log_fire,
    "black_blue_gold_white":  black_blue_gold_white,
    "red_white_black":        red_white_black,
}
