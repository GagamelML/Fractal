"""
Load an SVG file and rasterise it into a boolean mask at any resolution.
The shape is centred and scaled to just touch the image boundary.
"""
import numpy as np
from PIL import Image, ImageDraw
import svgpathtools


def _split_subpaths(path, tol=0.5):
    """Split an svgpathtools Path at discontinuities into sub-paths."""
    subpaths = []
    current = []
    for seg in path:
        if current and abs(seg.start - current[-1].end) > tol:
            subpaths.append(current)
            current = []
        current.append(seg)
    if current:
        subpaths.append(current)
    return subpaths


def _segments_to_polygon(segments, n_per_seg=50):
    """Sample a list of segments into (x, y) tuples."""
    pts = []
    for seg in segments:
        for t in np.linspace(0, 1, n_per_seg, endpoint=False):
            p = seg.point(t)
            pts.append((p.real, p.imag))
    # Close with the endpoint of the last segment
    p = segments[-1].point(1.0)
    pts.append((p.real, p.imag))
    return pts


def load_svg_mask(svg_path, width, height):
    """
    Render all paths/subpaths in *svg_path* as a boolean mask of shape (height, width).

    The shape is centred in the image and uniformly scaled so that it
    just touches whichever image edge it reaches first.

    Returns
    -------
    mask : ndarray[bool]  – True where the fractal should be computed
    ratio : float          – fraction of True pixels vs total (for logging)
    """
    paths, _ = svgpathtools.svg2paths(svg_path)
    if not paths:
        raise ValueError(f"No paths found in {svg_path}")

    # Split all paths into subpaths
    all_subpaths = []
    for p in paths:
        all_subpaths.extend(_split_subpaths(p))

    # Collect all points for bounding box
    all_pts = []
    polygons = []
    for sp in all_subpaths:
        poly = _segments_to_polygon(sp)
        polygons.append(poly)
        all_pts.extend(poly)

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    svg_w = max(xs) - min(xs)
    svg_h = max(ys) - min(ys)
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    # Scale to fit, with a tiny margin
    margin = 2  # pixels
    scale = min((width - 2 * margin) / svg_w,
                (height - 2 * margin) / svg_h)
    ox = width / 2
    oy = height / 2

    # Rasterise each subpath as a separate filled polygon
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        img_pts = [((x - cx) * scale + ox, (y - cy) * scale + oy)
                   for x, y in poly]
        draw.polygon(img_pts, fill=255)

    mask = np.array(img) > 127
    ratio = mask.sum() / mask.size
    return mask, ratio


def radial_fill(mask, center=None):
    """
    Fill all radial gaps in a mask relative to a center point.

    For every angle from center, find the outermost True pixel and fill
    everything from center to that radius. This ensures no radial gaps
    exist during zoom interpolation.

    Parameters
    ----------
    mask   : ndarray[bool] (H, W)
    center : (cy, cx) tuple, defaults to image center

    Returns
    -------
    filled : ndarray[bool] (H, W)
    """
    h, w = mask.shape
    if center is None:
        cy, cx = h / 2.0, w / 2.0
    else:
        cy, cx = center

    # Build polar coords for every pixel
    yy, xx = np.mgrid[0:h, 0:w]
    dy = yy - cy
    dx = xx - cx
    r = np.sqrt(dy**2 + dx**2)
    theta = np.arctan2(dy, dx)  # -pi..pi

    # Quantise angle into bins
    n_angles = max(w, h) * 4  # enough angular resolution
    theta_bin = ((theta + np.pi) / (2 * np.pi) * n_angles).astype(int) % n_angles

    # For each angle bin, find max radius that is in the mask
    r_max = np.zeros(n_angles, dtype=np.float64)
    mask_r = r.copy()
    mask_r[~mask] = 0
    np.maximum.at(r_max, theta_bin.ravel(), mask_r.ravel())

    # Fill: pixel is in render mask if r <= r_max for its angle bin
    filled = r <= r_max[theta_bin]
    return filled

