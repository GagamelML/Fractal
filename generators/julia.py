"""
Plain Julia set generator – no symmetry transformations.
"""
import numpy as np
from math import sqrt
from ._core import iterate_escape


class JuliaGenerator:
    """
    Standard Julia set on a rectangular viewport.
    Supports kernels: "poly" (z^p+c), "sin" (c·sin(z)), "cos" (c·cos(z)).

    Parameters
    ----------
    c        : complex – Julia constant
    center   : complex – viewport centre in the complex plane
    width    : int     – image width  (px)
    height   : int     – image height (px)
    max_iter : int     – iteration cap
    power    : int     – exponent (default 2)
    """

    def __init__(self, *, c, center=0+0j, width=640, height=640,
                 max_iter=512, power=2, kernel="poly", bailout=2.0):
        self.c = c
        self.center = center
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.power = power
        self.kernel = kernel
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
        z_full = scale * (self._grid + self.center)
        if mask is not None and not mask.all():
            counts = np.full(z_full.shape, self.max_iter, dtype=np.int16)
            z_sub = z_full[mask].copy()
            counts[mask] = iterate_escape(z_sub, self.c, self.max_iter,
                                          power=self.power,
                                          bailout=self.bailout,
                                          kernel=self.kernel)
            return counts
        return iterate_escape(z_full, self.c, self.max_iter, power=self.power,
                              bailout=self.bailout, kernel=self.kernel)

    def folder_name(self, scale_start, zoom_factor, num_frames):
        k = f"_{self.kernel}" if self.kernel != "poly" else ""
        return (
            f"julia{k}_c{self.c.real:+.2f}{self.c.imag:+.2f}j"
            f"_s{scale_start}_z{zoom_factor}_f{num_frames}"
            f"_{self.width}x{self.height}"
        )

