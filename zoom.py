"""
Zoom series generator – renders frames by progressively shrinking `scale`.
Vectorized with NumPy for performance.
"""
import numpy as np
import os
from math import sqrt
from time import perf_counter as clock
from PIL import Image

# ── Parameters ──────────────────────────────────────────────────────────────
sets = [
    {"c": 0.0 + 0.8j},
    {"c": -0.4 + 0.6j},
]
center = 0.0 + 0.0j
scale_start = 2.0
zoom_factor = 0.92
num_frames = 60
start_frame = 30  # skip already rendered frames
width = 640
height = 384
max_iter = 256

# ── Precompute pixel grid ──────────────────────────────────────────────────
factor = sqrt((width / 2.) ** 2 + (height / 2.) ** 2)
h_vals = np.arange(height // 2, -height // 2, -1, dtype=np.float64)
w_vals = np.arange(-width // 2, width // 2, dtype=np.float64)
ww, hh = np.meshgrid(w_vals, h_vals)
grid = ww / factor + (1.0j / factor) * hh

# ── Vectorized iteration ───────────────────────────────────────────────────
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

# ── PGM writer ─────────────────────────────────────────────────────────────
def write_pgm(data, fname):
    with open(fname, "wb") as f:
        f.write(b"P5\r\n")
        f.write(f"{width} {height}\r\n".encode())
        f.write(b"65535\r\n")
        f.write(data.astype('>i2').tobytes())

# ── Render loop ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for s in sets:
        c = s["c"]
        folder = (
            f"zoom_c{c.real:+.2f}{c.imag:+.2f}j"
            f"_center{center.real:+.2f}{center.imag:+.2f}j"
            f"_s{scale_start}_z{zoom_factor}_f{num_frames}"
            f"_{width}x{height}"
        )
        pgm_folder = os.path.join(folder, "pgm")
        png_folder = os.path.join(folder, "png")
        os.makedirs(pgm_folder, exist_ok=True)
        os.makedirs(png_folder, exist_ok=True)

        print(f"=== c = {c} ===")
        total_start = clock()

        for frame in range(start_frame, num_frames):
            scale = scale_start * (zoom_factor ** frame)
            t0 = clock()

            counts = render_frame(scale, c)

            # Normalize counts to full 0–65535 range
            cmax = counts.max()
            if cmax > 0:
                normalized = (counts.astype(np.float64) / cmax * 65535).astype(np.int16)
            else:
                normalized = counts

            pgm_path = os.path.join(pgm_folder, f"frame_{frame:04d}.pgm")
            png_path = os.path.join(png_folder, f"frame_{frame:04d}.png")
            write_pgm(normalized, pgm_path)

            # For PNG: scale to 0–255
            png_data = (counts.astype(np.float64) / max(cmax, 1) * 255).astype(np.uint8)
            Image.fromarray(png_data, mode='L').save(png_path)
            elapsed = clock() - t0
            print(f"  [{frame+1}/{num_frames}] scale={scale:.6f}  ({elapsed:.1f}s)")

        print(f"  Done in {clock() - total_start:.1f}s → {folder}/\n")
