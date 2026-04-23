"""
Mandelbrot set generator – iterates z = z^p + c where c = pixel coordinate.
"""
import numpy as np
from math import sqrt


class MandelbrotGenerator:
    """
    Classic Mandelbrot set on a rectangular viewport.

    Unlike Julia sets, here c varies per pixel (c = grid position)
    and z starts at 0 (or a configurable z0).

    Parameters
    ----------
    center   : complex – viewport centre in the complex plane
    width    : int     – image width  (px)
    height   : int     – image height (px)
    max_iter : int     – iteration cap
    power    : int     – exponent for z^power + c
    bailout  : float   – escape radius
    """

    def __init__(self, *, center=-0.5+0j, width=640, height=640,
                 max_iter=1024, power=2, bailout=2.0, recenter=False,
                 # Accept and ignore Julia-specific kwargs for compatibility
                 c=None, kernel="poly", **_ignored):
        self.center = center
        self.recenter = recenter
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.power = power
        self.bailout = bailout
        self._build_grid()

    def _build_grid(self):
        factor = sqrt((self.width / 2.) ** 2 + (self.height / 2.) ** 2)
        h_vals = np.arange(self.height // 2, -self.height // 2, -1,
                           dtype=np.float64)
        w_vals = np.arange(-self.width // 2, self.width // 2,
                           dtype=np.float64)
        ww, hh = np.meshgrid(w_vals, h_vals)
        self._grid = ww / factor + (1.0j / factor) * hh

    def render(self, scale, mask=None):
        """Return iteration-count array for the given scale."""
        if self.recenter:
            c = scale * self._grid + self.center
        else:
            c = scale * (self._grid + self.center)

        if mask is not None and not mask.all():
            counts = np.full(c.shape, self.max_iter, dtype=np.int16)
            c_sub = c[mask].copy()
            counts[mask] = self._iterate(c_sub)
            return counts
        return self._iterate(c)

    def _iterate(self, c):
        """Vectorised Mandelbrot escape-time iteration."""
        z = np.zeros_like(c)
        counts = np.full(c.shape, self.max_iter, dtype=np.int16)
        mask = np.ones(c.shape, dtype=bool)

        for i in range(self.max_iter):
            z[mask] = z[mask] ** self.power + c[mask]
            escaped = mask & (np.abs(z) > self.bailout)
            counts[escaped] = i
            mask[escaped] = False
            if not mask.any():
                break
        return counts

    def folder_name(self, scale_start, zoom_factor, num_frames):
        return (
            f"mandelbrot_center{self.center.real:+.2f}{self.center.imag:+.2f}j"
            f"_s{scale_start}_z{zoom_factor}_f{num_frames}"
            f"_{self.width}x{self.height}"
        )

