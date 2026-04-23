"""Fractal generators – each exposes a class with .render(scale) → counts array."""

from .julia import JuliaGenerator
from .flower import FlowerGenerator
from .mandelbrot import MandelbrotGenerator

GENERATORS = {
    "julia": JuliaGenerator,
    "flower": FlowerGenerator,
    "mandelbrot": MandelbrotGenerator,
}

