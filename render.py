"""
Render a zoom series: Generator × Colorizer(s) → PNG frames.

Usage examples:
    python render.py                           # uses defaults below
    python render.py --generator flower --colorizers hsv_cyclic twilight
"""
import os
import argparse
from time import perf_counter as clock
from concurrent.futures import ProcessPoolExecutor, as_completed

from generators import GENERATORS
from colorizers import COLORIZERS


# ── defaults (edit here for quick runs) ──────────────────────────────────────
DEFAULTS = dict(
    generator="flower",
    colorizers=list(COLORIZERS.keys()),
    c_real=-0.54, c_imag=0.54,
    center_real=0.0, center_imag=0.0,
    width=640, height=640,
    max_iter=1024, power=2,
    scale_start=2.0, zoom_factor=0.92,
    num_frames=30,
    petals=4, mirror_segments=False,
    interest_angle=110, align_north="edge",
    workers=12,
    kernel="poly", bailout=2.0,
    mask_svg=None,
)


# ── worker process ───────────────────────────────────────────────────────────
_worker_gen = None
_worker_colors = None
_worker_max_iter = None
_worker_folder = None
_worker_mask = None


def _worker_init(gen_cls, gen_kwargs, color_names, folder, mask_svg):
    """Called once per worker process – builds its own generator."""
    global _worker_gen, _worker_colors, _worker_max_iter, _worker_folder, _worker_mask
    _worker_gen = gen_cls(**gen_kwargs)
    _worker_colors = {n: COLORIZERS[n] for n in color_names}
    _worker_max_iter = gen_kwargs["max_iter"]
    _worker_folder = folder
    if mask_svg:
        from mask import load_svg_mask
        _worker_mask, _ = load_svg_mask(mask_svg,
                                        gen_kwargs["width"],
                                        gen_kwargs["height"])
    else:
        _worker_mask = None


def _worker_render(frame, scale):
    """Render one frame, save all colorizations. Returns (frame, elapsed)."""
    t0 = clock()
    counts = _worker_gen.render(scale, mask=_worker_mask)
    for name, fn in _worker_colors.items():
        img = fn(counts, _worker_max_iter)
        # Apply mask: set outside pixels to transparent black
        if _worker_mask is not None:
            import numpy as np
            arr = np.array(img)
            arr[~_worker_mask] = 0
            from PIL import Image
            img = Image.fromarray(arr, mode='RGB')
        img.save(os.path.join(_worker_folder, name, f"frame_{frame:04d}.png"))
    return frame, clock() - t0


# ── main ─────────────────────────────────────────────────────────────────────

def build_gen_kwargs(args):
    common = dict(
        c=complex(args.c_real, args.c_imag),
        center=complex(args.center_real, args.center_imag),
        width=args.width, height=args.height,
        max_iter=args.max_iter, power=args.power,
        kernel=args.kernel, bailout=args.bailout,
    )
    if args.generator == "flower":
        common.update(
            petals=args.petals,
            mirror_segments=args.mirror_segments,
            interest_angle=args.interest_angle,
            align_north=args.align_north,
        )
    return common


def render(args):
    gen_cls = GENERATORS[args.generator]
    gen_kwargs = build_gen_kwargs(args)
    color_names = args.colorizers

    # Build once locally just for folder name
    gen_tmp = gen_cls(**gen_kwargs)
    folder = os.path.join("results",
                          gen_tmp.folder_name(args.scale_start, args.zoom_factor,
                                              args.num_frames))
    for name in color_names:
        os.makedirs(os.path.join(folder, name), exist_ok=True)

    frames = [(f, args.scale_start * (args.zoom_factor ** f))
              for f in range(args.start_frame,
                             args.start_frame + args.num_frames,
                             args.frame_step)]

    total_start = clock()
    n_total = len(frames)
    n_workers = min(args.workers, n_total)

    if n_workers <= 1:
        # Single-process fallback
        _worker_init(gen_cls, gen_kwargs, color_names, folder, args.mask_svg)
        for i, (frame, scale) in enumerate(frames, 1):
            f, elapsed = _worker_render(frame, scale)
            print(f"[{i}/{n_total}] scale={scale:.6f}  ({elapsed:.1f}s)")
    else:
        done = 0
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(gen_cls, gen_kwargs, color_names, folder, args.mask_svg),
        ) as pool:
            futures = {pool.submit(_worker_render, f, s): f for f, s in frames}
            for fut in as_completed(futures):
                f, elapsed = fut.result()
                done += 1
                scale = args.scale_start * (args.zoom_factor ** f)
                print(f"[{done}/{n_total}] frame {f:3d}  "
                      f"scale={scale:.6f}  ({elapsed:.1f}s)")

    print(f"\nDone. {n_total} frames in {clock()-total_start:.1f}s "
          f"({n_workers} workers) → {folder}/")


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fractal zoom renderer")
    d = DEFAULTS
    p.add_argument("--generator", default=d["generator"],
                   choices=GENERATORS.keys())
    p.add_argument("--colorizers", nargs="+", default=d["colorizers"],
                   choices=COLORIZERS.keys())
    p.add_argument("--c-real", type=float, default=d["c_real"])
    p.add_argument("--c-imag", type=float, default=d["c_imag"])
    p.add_argument("--center-real", type=float, default=d["center_real"])
    p.add_argument("--center-imag", type=float, default=d["center_imag"])
    p.add_argument("--width", type=int, default=d["width"])
    p.add_argument("--height", type=int, default=d["height"])
    p.add_argument("--max-iter", type=int, default=d["max_iter"])
    p.add_argument("--power", type=int, default=d["power"])
    p.add_argument("--kernel", default=d["kernel"],
                   choices=["poly", "sin", "cos"])
    p.add_argument("--bailout", type=float, default=d["bailout"])
    p.add_argument("--scale-start", type=float, default=d["scale_start"])
    p.add_argument("--zoom-factor", type=float, default=d["zoom_factor"])
    p.add_argument("--num-frames", type=int, default=d["num_frames"])
    p.add_argument("--start-frame", type=int, default=0,
                   help="First frame index (to continue a previous run)")
    p.add_argument("--frame-step", type=int, default=1,
                   help="Only render every N-th frame")
    p.add_argument("--petals", type=int, default=d["petals"])
    p.add_argument("--mirror-segments", action="store_true",
                   default=d["mirror_segments"])
    p.add_argument("--interest-angle", type=float,
                   default=d["interest_angle"])
    p.add_argument("--align-north", default=d["align_north"],
                   choices=["center", "edge"])
    p.add_argument("--workers", type=int, default=d["workers"],
                   help="Number of parallel worker processes")
    p.add_argument("--mask-svg", type=str, default=d["mask_svg"],
                   help="Optional SVG file for image mask")
    return p.parse_args()


if __name__ == "__main__":
    render(parse_args())
