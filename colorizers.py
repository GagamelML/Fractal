"""
Colorizer functions.  Each has signature:
    colorize(counts, max_iter, **kwargs) → PIL.Image (RGB)
"""
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _hsv_to_rgb(hue, sat, val):
    """All inputs: float64 arrays in [0,1].  Returns uint8 array (..., 3)."""
    h6 = hue * 6.0
    sector = h6.astype(int) % 6
    f = h6 - np.floor(h6)
    p = val * (1 - sat)
    q = val * (1 - sat * f)
    t = val * (1 - sat * (1 - f))
    rgb = np.zeros(hue.shape + (3,), dtype=np.float64)
    for idx, (v1, v2, v3) in enumerate(
        [(val, t, p), (q, val, p), (p, val, t),
         (p, q, val), (t, p, val), (val, p, q)]
    ):
        m = sector == idx
        rgb[m, 0] = v1[m]
        rgb[m, 1] = v2[m]
        rgb[m, 2] = v3[m]
    return (rgb * 255).astype(np.uint8)


def _normalize(counts, max_iter, plow=1.0, phigh=99.0):
    """Normalize escaped pixel counts to [0, 1] using percentile stretch.
    Returns (t, esc_mask) where t is a float64 array for escaped pixels only."""
    esc = counts < max_iter
    if not esc.any():
        return None, esc
    vals = counts[esc].astype(np.float64)
    lo = np.percentile(vals, plow)
    hi = np.percentile(vals, phigh)
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
    return t, esc


# ---------------------------------------------------------------------------
#  Colorizers
# ---------------------------------------------------------------------------

def hsv_cyclic(counts, max_iter, *, cycles=3, saturation=0.9, value=0.85,
               reverse=True, interior_color=(15, 5, 30)):
    """HSV hue cycle, dynamically scaled per frame."""
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[counts >= max_iter] = interior_color
    t, esc = _normalize(counts, max_iter)
    if t is not None:
        hue = (t * cycles) % 1.0
        if reverse:
            hue = (1.0 - hue) % 1.0
        rgb[esc] = _hsv_to_rgb(
            hue,
            np.full_like(hue, saturation),
            np.full_like(hue, value),
        )
    return Image.fromarray(rgb, mode='RGB')


def twilight(counts, max_iter, *, interior_color=(30, 15, 40)):
    """matplotlib twilight_shifted, dynamically scaled per frame."""
    from matplotlib import colormaps
    cmap = colormaps['twilight_shifted']
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[counts >= max_iter] = interior_color
    t, esc = _normalize(counts, max_iter)
    if t is not None:
        rgb[esc] = (cmap(t)[:, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def log_fire(counts, max_iter, *, gamma=1.8, intensity=2.5, blue_suppress=0.4,
             interior_color=(40, 5, 0)):
    """Log-scaled fire palette, dynamically scaled per frame."""
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[counts >= max_iter] = interior_color
    esc = counts < max_iter
    if esc.any():
        raw = counts[esc].astype(np.float64) + 1
        log_val = np.log(raw)
        lo = np.percentile(log_val, 1.0)
        hi = np.percentile(log_val, 99.0)
        if hi <= lo:
            hi = lo + 1.0
        t = (np.clip((log_val - lo) / (hi - lo), 0.0, 1.0)) ** gamma
        r = np.clip(t * intensity, 0, 1)
        g = np.clip(t * intensity - 1, 0, 1)
        b = np.clip(t * intensity - 2, 0, 1) * blue_suppress
        rgb[esc, 0] = (r * 255).astype(np.uint8)
        rgb[esc, 1] = (g * 255).astype(np.uint8)
        rgb[esc, 2] = (b * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------
COLORIZERS = {
    "hsv_cyclic": hsv_cyclic,
    "twilight":   twilight,
    "log_fire":   log_fire,
}

