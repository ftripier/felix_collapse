import numpy as np
import random
from numpy.typing import NDArray
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from typing import Tuple, Set, Generator, List
from enum import Enum
from .source_patterns import Pattern, SourcePatterns


@dataclass
class ColorPixel:
    amplitude: float
    alpha: float
    value: float


@dataclass
class AlphaPixel:
    amplitude: float
    value: float


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclass
class SuperpositionPixel:
    red: List[ColorPixel]
    green: List[ColorPixel]
    blue: List[ColorPixel]
    alpha: List[AlphaPixel]

    def get_color_channel(self, color: Color) -> int:
        # not gamma corrected
        color_channel: List[ColorPixel] = getattr(self, color.value)
        if len(color_channel) < 1:
            return 0
        total_amplitude = sum([color.amplitude for color in color_channel])
        # sort by amplitude in reverse because we want the highest amplitude colors
        # to layer with each other consistently, and we're using back-to-front alpha
        # blending
        weight_sorted = sorted(color_channel, key=lambda x: x.amplitude, reverse=True)
        color_value = weight_sorted[0].value
        # fold all colors into each other using non-gamma-corrected back-to-front alpha blending
        # weighted by the amplitude share of the color
        for color_pixel in weight_sorted[1:]:
            weighted_alpha = color_pixel.alpha * (
                color_pixel.amplitude / total_amplitude
            )
            color_value = (
                color_value * (255.0 - weighted_alpha)
                + color_pixel.value * weighted_alpha
            ) / 255.0
        return int(color_value)

    def get_alpha_channel(self) -> int:
        alpha_share = 1.0
        for alpha_pixel in self.alpha:
            alpha_share = alpha_share * (1.0 - alpha_pixel.value)
        return int(alpha_share * 255)

    def get_color_vector(self) -> NDArray[np.int32]:
        return np.array(
            [
                self.get_color_channel(Color.RED),
                self.get_color_channel(Color.GREEN),
                self.get_color_channel(Color.BLUE),
                self.get_alpha_channel(),
            ]
        )


@dataclass
class WaveCell:
    entropy: float
    image_coordinates: Tuple[int, int]
    pattern_amplitudes: defaultdict[Pattern, float]

    def get_pixel(self) -> NDArray[np.int32]:
        final_pixel = SuperpositionPixel([], [], [], [])
        for pattern, amplitude in self.pattern_amplitudes.items():
            pattern_image = pattern.get_ndarray()
            top_left_pixel = pattern_image[(0, 0)]
            r, g, b, a = [float(color) for color in top_left_pixel]
            final_pixel.red.append(ColorPixel(amplitude, a, r))
            final_pixel.green.append(ColorPixel(amplitude, a, g))
            final_pixel.blue.append(ColorPixel(amplitude, a, b))
            final_pixel.alpha.append(AlphaPixel(amplitude, a))
        return final_pixel.get_color_vector()


class Contradiction(Exception):
    pass


class Wavefunction:
    """
    Has two methods: propagate and observe.
    1. initialize the cell grid in the requested dimensions
    2. each cell keeps track of a superposition so far. Entropy can be derived
    from the superposition, but it's nice to be able to modify it outside of
    that context.
    """

    def __init__(
        self, dimenstions: Tuple[int, int], source_patterns: SourcePatterns
    ) -> None:
        # height, width
        self.dimensions = dimenstions
        self.source_patterns = source_patterns
        # a graph of cells. nids are the original image coordinates, data
        # values are the cell object, which keeps track of its entropy.
        self.cells = nx.Graph()
        # keeps track of observed (dirty) nids that will act as the source of
        # propoaation.
        self.observed_cells: Set[Tuple[int, int]] = set()

    def get_cells(self) -> Generator[WaveCell, None, None]:
        for _, cell in self.cells.nodes.data("cell"):  # type: ignore
            yield cell

    def propagate(self) -> None:
        """
        starting with observed cells as the source,
        update the allowlist of patterns for other cells in
        breadth-first-search order, by propagating constraints
        from the source pattern adjacencies.
        """
        raise NotImplementedError

    def observe(self) -> None:
        raise NotImplementedError

    def is_fully_collapsed(self) -> bool:
        for cell in self.get_cells():
            nonzero_amplitudes = len(
                [
                    amplitude
                    for amplitude in cell.pattern_amplitudes.values()
                    if amplitude != 0
                ]
            )
            if nonzero_amplitudes > 0:
                raise Contradiction(f"Ran into cell with no nonzero amplitudes {cell}")
            if nonzero_amplitudes > 1:
                return False
        return True

    def produce_image(self) -> NDArray[np.int32]:
        image = np.array(
            [[[] for _ in range(self.dimensions[1])] for _ in range(self.dimensions[0])]
        )
        for cell in self.get_cells():
            image[cell.image_coordinates] = cell.get_pixel()
        return image
