from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass
from typing import Set
from collections import defaultdict
from typing import Tuple
from enum import Enum


@dataclass(frozen=True)
class Pattern:
    hashed_pixels: bytes
    original_shape: Tuple[int, int, int]

    @classmethod
    def from_ndarray(cls, pixels: NDArray[np.byte]):
        return cls(
            bytes(pixels.flatten().tolist()),
            (pixels.shape[0], pixels.shape[1], pixels.shape[2]),
        )

    def get_ndarray(self) -> NDArray[np.byte]:
        deserialized = np.fromiter(self.hashed_pixels, np.int32)
        deserialized = np.reshape(deserialized, self.original_shape)
        return deserialized


class Adjacency:
    """Parses an image. Identifies patterns in the image, counts their
    frequency, and identifies "constraints" in the adjacency of patterns.
    For the ESTM model, patterns are pixels of a single color."""

    def __init__(self, source_texture: NDArray[np.byte], N: int = 3) -> None:
        self.source_texture = source_texture
        self.N = N
        self.patterns: Set[Pattern] = set()
        # fmt: off
        self.pattern_frequencies: defaultdict[Pattern, int] = (
            defaultdict(lambda: 0)
        )
        self._collect_patterns()


    def _collect_patterns(self):
        """Runs a NxN convolution across the whole image, and stores each
        convolution in a flat, deduplicated list of patterns.
        """
        height, width, _ = self.source_texture.shape
        for y in range(height):
            for x in range(width):
                window = self.source_texture.take(
                    range(x, x + self.N), mode="wrap", axis=0
                ).take(range(y, y + self.N), mode="wrap", axis=1)
                pattern = Pattern.from_ndarray(window)
                self.patterns.add(pattern)
                self.pattern_frequencies[pattern] += 1


