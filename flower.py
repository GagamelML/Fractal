"""
Render z²+c Julia set, then radially mirror it N-fold to create a flower.
"""
import numpy as np
import os
from math import sqrt, pi
from time import perf_counter as clock
from PIL import Image
import colorsys

width = 640
height = 640
max_iter = 512
num_frames = 30
scale_start = 2.0
zoom_factor = 0.92
petals = 4
mirror_segments = False
interest_angle = 110
align_north = "edge"
c = -0.54 + 0.54j
center = 0.0 + 0.0j

# --- Color schemes -----------------------------------------------------------

def colorize_hsv_cyclic(counts):
    """Escape count mapped to HSV hue cycle, interior = black."""
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    escaped = counts < max_iter
    if escaped.any():
        t = counts[escaped].astype(np.float64) / max_iter
        hue = t % 1.0
        sat = np.ones_like(hue)
        val = np.ones_like(hue)
        for i, (hh, ss, vv) in enumerate(zip(hue, sat, val)):
            r, g, b = colorsys.hsv_to_rgb(hh, ss, vv)
            rgb[escaped][i] = (int(r * 255), int(g * 255), int(b * 255))
    return Image.fromarray(rgb, mode='RGB')


def _hsv_array(hue, sat, val):
    """Vectorised HSV→RGB, all inputs float arrays in [0,1]."""
    # Using the sector method
    h6 = hue * 6.0
    sector = h6.astype(int) % 6
    f = h6 - np.floor(h6)
    p = val * (1 - sat)
    q = val * (1 - sat * f)
    t = val * (1 - sat * (1 - f))
    rgb = np.zeros(hue.shape + (3,), dtype=np.float64)
    for idx, (v1, v2, v3) in enumerate([(val,t,p),(q,val,p),(p,val,t),(p,q,val),(t,p,val),(val,p,q)]):
        mask = sector == idx
        rgb[mask, 0] = v1[mask]
        rgb[mask, 1] = v2[mask]
        rgb[mask, 2] = v3[mask]
    return (rgb * 255).astype(np.uint8)


def colorize_hsv_cyclic_fast(counts):
    """HSV hue cycle – vectorised."""
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    escaped = counts < max_iter
    if escaped.any():
        t = counts[escaped].astype(np.float64) / max_iter
        hue = (1.0 - (t * 3) % 1.0) % 1.0   # 3 full hue rotations, reversed
        sat = np.full_like(hue, 0.9)
        val = np.full_like(hue, 0.85)
        rgb[escaped] = _hsv_array(hue, sat, val)
    return Image.fromarray(rgb, mode='RGB')


def colorize_smooth_twilight(counts):
    """Smooth banding with matplotlib's twilight_shifted (cyclic) colormap."""
    from matplotlib import colormaps
    cmap = colormaps['twilight_shifted']
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    escaped = counts < max_iter
    if escaped.any():
        t = counts[escaped].astype(np.float64) / max_iter
        colors = cmap(t)[:, :3]       # RGBA → RGB
        rgb[escaped] = (colors * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def colorize_log_fire(counts):
    """Log-scaled escape count → fire palette (black→red→orange→yellow→white)."""
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    escaped = counts < max_iter
    if escaped.any():
        raw = counts[escaped].astype(np.float64) + 1
        log_val = np.log(raw) / np.log(max_iter + 1)   # normalise to [0,1]
        t = log_val ** 1.8                            # push values darker
        # Fire gradient: 0→black, 0.4→deep red, 0.7→orange, 1→yellow (no white)
        r = np.clip(t * 2.5, 0, 1)
        g = np.clip(t * 2.5 - 1, 0, 1)
        b = np.clip(t * 2.5 - 2, 0, 1) * 0.4          # suppress blue → no white
        rgb[escaped, 0] = (r * 255).astype(np.uint8)
        rgb[escaped, 1] = (g * 255).astype(np.uint8)
        rgb[escaped, 2] = (b * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


COLOR_SCHEMES = {
    "hsv_cyclic":      colorize_hsv_cyclic_fast,
    "twilight":        colorize_smooth_twilight,
    "log_fire":        colorize_log_fire,
}

# Precompute polar pixel grid
# Each pixel → complex coordinate → polar → fold angle into [0, 2π/petals] → back to complex
factor = sqrt((width / 2.) ** 2 + (height / 2.) ** 2)
h_vals = np.arange(height // 2, -height // 2, -1, dtype=np.float64)
w_vals = np.arange(-width // 2, width // 2, dtype=np.float64)
ww, hh = np.meshgrid(w_vals, h_vals)
raw_grid = ww / factor + (1.0j / factor) * hh

# Fold into 1/petals slice, mirror every other segment for smooth boundaries
r = np.abs(raw_grid)
theta = np.angle(raw_grid)                       # in [-π, π], 0 = east
slice_angle = 2 * pi / petals

# Rotation so the desired output orientation points north:
#   "center" → segment centre at north (π/2)
#   "edge"   → segment boundary at north (π/2)
if align_north == "center":
    output_rot = pi / 2                           # centre of slice 0 → north
elif align_north == "edge":
    output_rot = pi / 2 - slice_angle / 2         # edge of slice 0 → north
else:
    output_rot = pi / 2

theta_shifted = theta - output_rot                # rotate output
theta_shifted = theta_shifted % (2 * pi)          # into [0, 2π)

# Which slice each pixel is in
slice_idx = np.floor(theta_shifted / slice_angle).astype(int)
# Angle within the slice
local_theta = theta_shifted - slice_idx * slice_angle
# Mirror odd slices for smooth boundaries
if mirror_segments:
    odd = (slice_idx % 2) != 0
    local_theta[odd] = slice_angle - local_theta[odd]

# Map the folded angle back, offsetting by interest_angle so that
# the "interesting" fractal direction lands in the middle of every segment.
m_rad = np.deg2rad(interest_angle)                # user-chosen direction (0=north cw)
# Convert from "0=north, clockwise" to standard math angle:
m_math = pi / 2 - m_rad
fractal_theta = local_theta + m_math - slice_angle / 2   # centre slice on m

grid = r * np.exp(1j * fractal_theta)


def render_frame(scale, c_val):
    z = scale * (grid + center)
    counts = np.full(z.shape, max_iter, dtype=np.int16)
    mask = np.ones(z.shape, dtype=bool)
    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c_val
        escaped = mask & (np.abs(z) > 2.0)
        counts[escaped] = i
        mask[escaped] = False
        if not mask.any():
            break
    return counts


folder = (
    f"flower{petals}_c{c.real:+.2f}{c.imag:+.2f}j"
    f"_s{scale_start}_z{zoom_factor}_f{num_frames}"
    f"_{width}x{height}"
)

if __name__ == "__main__":
    total_start = clock()
    for scheme_name, colorize_fn in COLOR_SCHEMES.items():
        sub = os.path.join(folder, scheme_name)
        os.makedirs(sub, exist_ok=True)

    for frame in range(num_frames):
        scale = scale_start * (zoom_factor ** frame)
        t0 = clock()
        counts = render_frame(scale, c)
        for scheme_name, colorize_fn in COLOR_SCHEMES.items():
            img = colorize_fn(counts)
            path = os.path.join(folder, scheme_name, f"frame_{frame:04d}.png")
            img.save(path)
        elapsed = clock() - t0
        print(f"[{frame+1}/{num_frames}] scale={scale:.6f}  ({elapsed:.1f}s)")

    print(f"\nDone. {num_frames} frames in {clock() - total_start:.1f}s → {folder}/")
