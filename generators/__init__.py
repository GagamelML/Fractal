"""Fractal generators – each exposes a class with .render(scale) → counts array."""

from .julia import JuliaGenerator
from .flower import FlowerGenerator

GENERATORS = {
    "julia": JuliaGenerator,
    "flower": FlowerGenerator,
}

