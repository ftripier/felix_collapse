import itertools
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
    N: int

    @classmethod
    def from_ndarray(cls, pixels: NDArray[np.byte]):
        assert pixels.shape[0] == pixels.shape[1]
        return cls(
            bytes(pixels.flatten().tolist()),
            (pixels.shape[0], pixels.shape[1], pixels.shape[2]),
            pixels.shape[0],
        )

    def get_ndarray(self) -> NDArray[np.byte]:
        deserialized = np.fromiter(self.hashed_pixels, np.int32)
        deserialized = np.reshape(deserialized, self.original_shape)
        return deserialized

    def overlaps_right(self, other: "Pattern") -> bool:
        """
        if all but the last column of our pattern is equal to the last few
        columns of another, we overlap the other pattern to the right and it
        may be placed to the left of us.
        """
        self_nd = self.get_ndarray()
        other_nd = other.get_ndarray()
        # fmt: off
        our_first_columns = self_nd[:, 0: self.N - 1]
        # fmt: off
        their_last_columns = other_nd[:, 1: self.N]
        return np.array_equal(our_first_columns, their_last_columns)

    def overlaps_bottom(self, other: "Pattern") -> bool:
        """
        if all but the last row of our pattern is equal to the last few
        rows of another, we overlap the other pattern by the bottom and it
        may be placed on top of us
        """
        self_nd = self.get_ndarray()
        other_nd = other.get_ndarray()
        # fmt: off
        our_first_rows = self_nd[0: self.N - 1]
        # fmt: off
        their_last_rows = other_nd[1: self.N]
        return np.array_equal(our_first_rows, their_last_rows)


class Directions(Enum):
    TOP = (-1, 0)
    BOTTOM = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class SourcePatterns:
    """Parses an image. Identifies patterns in the image, counts their
    frequency, and identifies "constraints" in the adjacency of patterns.
    For the ESTM model, patterns are pixels of a single color."""

    def __init__(self, source_texture: NDArray[np.byte], N: int = 3) -> None:
        self.source_texture = source_texture
        self.N = N
        self.patterns: Set[Pattern] = set()
        # fmt: off
        self.frequencies: defaultdict[Pattern, int] = (
            defaultdict(lambda: 0)
        )
        self.adjacencies: defaultdict[
            Pattern, defaultdict[Directions, Set[Pattern]]
        ] = (
            defaultdict(lambda: defaultdict(lambda: set()))
        )
        self._collect_patterns()
        self._collect_adjacencies()

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
                # TODO: add reflections and rotations
                self.patterns.add(pattern)
                self.frequencies[pattern] += 1

    def _collect_adjacencies(self):
        # fmt: off
        for pattern_a, pattern_b in itertools.product(
            self.patterns,
            self.patterns
        ):
            """(in case when N = 3) If the first two columns of
            pattern 1 == the last two columns of pattern 2 -->
            pattern 2 can be placed to the left (0) of pattern 1"""
            if pattern_a.overlaps_right(pattern_b):
                self.adjacencies[pattern_b][Directions.RIGHT].add(pattern_a)
                self.adjacencies[pattern_a][Directions.LEFT].add(pattern_b)
            """ (in case when N = 3)
            If the first two rows of pattern 1 == the last two rows of
            pattern 2 --> pattern 2 can be placed on top (2) of pattern 1"""
            if pattern_a.overlaps_bottom(pattern_b):
                self.adjacencies[pattern_b][Directions.BOTTOM].add(pattern_a)
                self.adjacencies[pattern_a][Directions.TOP].add(pattern_b)
