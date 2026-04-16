"""
Flower generator – radial N-fold symmetry over a Julia kernel.
"""
import numpy as np
from math import sqrt, pi
from ._core import iterate_escape


class FlowerGenerator:
    """
    Julia set with radial petal symmetry.

    Parameters
    ----------
    c               : complex – Julia constant
    center          : complex – viewport centre
    width, height   : int     – image size (px)
    max_iter        : int     – iteration cap
    power           : int     – exponent for z^power + c
    petals          : int     – number of symmetric segments
    mirror_segments : bool    – mirror every other segment for seamless edges
    interest_angle  : float   – degrees (0 = north, clockwise) selecting the
                                fractal direction placed at segment centres
    align_north     : str     – "center" or "edge" – what points north
    """

    def __init__(self, *, c, center=0+0j, width=640, height=640,
                 max_iter=512, power=2, kernel="poly", bailout=2.0,
                 petals=8, mirror_segments=True,
                 interest_angle=0, align_north="center"):
        self.c = c
        self.center = center
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.power = power
        self.kernel = kernel
        self.bailout = bailout
        self.petals = petals
        self.mirror_segments = mirror_segments
        self.interest_angle = interest_angle
        self.align_north = align_north
        self._build_grid()

    def _build_grid(self):
        w, h = self.width, self.height
        factor = sqrt((w / 2.) ** 2 + (h / 2.) ** 2)
        h_vals = np.arange(h // 2, -h // 2, -1, dtype=np.float64)
        w_vals = np.arange(-w // 2, w // 2, dtype=np.float64)
        ww, hh = np.meshgrid(w_vals, h_vals)
        raw = ww / factor + (1.0j / factor) * hh

        r = np.abs(raw)
        theta = np.angle(raw)
        sa = 2 * pi / self.petals  # slice angle

        # Output rotation
        if self.align_north == "center":
            rot = pi / 2
        elif self.align_north == "edge":
            rot = pi / 2 - sa / 2
        else:
            rot = pi / 2

        ts = (theta - rot) % (2 * pi)
        sidx = np.floor(ts / sa).astype(int)
        local = ts - sidx * sa

        if self.mirror_segments:
            odd = (sidx % 2) != 0
            local[odd] = sa - local[odd]

        m_math = pi / 2 - np.deg2rad(self.interest_angle)
        fractal_theta = local + m_math - sa / 2

        full_grid = r * np.exp(1j * fractal_theta)

        # ── Symmetry optimisation ──────────────────────────────────────
        # Every pixel's fractal coordinate depends only on (r, local_theta).
        # Pixels in different slices with same (r, local_theta) are identical.
        # We find unique coordinates and build a reverse lookup.
        flat = full_grid.ravel()
        # Quantise real+imag separately to build a stable hash key
        precision = 1e8  # plenty for float64
        key_r = np.round(flat.real * precision).astype(np.int64)
        key_i = np.round(flat.imag * precision).astype(np.int64)
        # Pack into a single int128-like via structured array
        keys = np.empty(len(flat), dtype=[('r', np.int64), ('i', np.int64)])
        keys['r'] = key_r
        keys['i'] = key_i

        # np.unique on structured array
        _, unique_idx, inverse_idx = np.unique(
            keys, return_index=True, return_inverse=True
        )

        self._master_grid = flat[unique_idx]
        self._pixel_to_master = inverse_idx.astype(np.int32)
        self._full_shape = (h, w)

        n_total = h * w
        n_unique = len(unique_idx)
        print(f"  Flower symmetry: {n_total} pixels → {n_unique} unique "
              f"({n_total / n_unique:.1f}× speedup)")

    def render(self, scale, mask=None):
        z = scale * (self._master_grid + self.center)
        # If mask provided, only compute pixels that are inside the mask
        if mask is not None:
            flat_mask = mask.ravel()
            # Map image-space mask to master-space: a master pixel is needed
            # if ANY image pixel mapping to it is inside the mask.
            master_needed = np.zeros(len(self._master_grid), dtype=bool)
            np.maximum.at(master_needed, self._pixel_to_master,
                          flat_mask)
            master_counts = np.full(len(self._master_grid), self.max_iter,
                                    dtype=np.int16)
            if master_needed.any():
                z_sub = z[master_needed].copy()
                master_counts[master_needed] = iterate_escape(
                    z_sub, self.c, self.max_iter,
                    power=self.power, bailout=self.bailout,
                    kernel=self.kernel)
        else:
            master_counts = iterate_escape(z, self.c, self.max_iter,
                                           power=self.power,
                                           bailout=self.bailout,
                                           kernel=self.kernel)
        return master_counts[self._pixel_to_master].reshape(self._full_shape)

    def folder_name(self, scale_start, zoom_factor, num_frames):
        return (
            f"flower{self.petals}_c{self.c.real:+.2f}{self.c.imag:+.2f}j"
            f"_s{scale_start}_z{zoom_factor}_f{num_frames}"
            f"_{self.width}x{self.height}"
        )
