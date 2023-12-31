import numpy as np
import random
import itertools
from numpy.typing import NDArray
from dataclasses import dataclass
import networkx as nx
from typing import Tuple, Set, Generator, List, Dict
from enum import Enum
from .source_patterns import Pattern, SourcePatterns, Directions


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
        total_amplitude = sum([pixel.amplitude for pixel in self.alpha])
        for alpha_pixel in self.alpha:
            alpha_subtraction = 255.0/alpha_pixel.value * alpha_pixel.amplitude/total_amplitude
            alpha_share = alpha_share * (1.0 - alpha_subtraction)
        final_alpha = 1.0 - alpha_share
        return int(final_alpha * 255)

    def get_color_vector(self) -> NDArray[np.uint8]:
        return np.array(
            [
                self.get_color_channel(Color.RED),
                self.get_color_channel(Color.GREEN),
                self.get_color_channel(Color.BLUE),
                self.get_alpha_channel(),
            ],
            dtype=np.uint8
        )


@dataclass
class WaveCell:
    entropy: float
    image_coordinates: Tuple[int, int]
    pattern_amplitudes: Dict[Pattern, float]

    def __post_init__(self):
        self._update_entropy()

    def get_pixel(self) -> NDArray[np.uint8]:
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

    def possibilities(self) -> Set[Pattern]:
        return set(self.pattern_amplitudes.keys())

    def _update_entropy(self):
        # entropy = the number of patterns available to choose from - a small random term
        # to make selecting the minimum entropy node less stable.
        self.entropy = len(self.pattern_amplitudes) - random.uniform(0.0000001, 0.1)

    def observe(self):
        """Use amplitude weighted choice to choose pattern, then collapse
        pattern amplitudes
        """
        if self.is_collapsed():
            return
        total_amplitude = sum(
            [amplitude for amplitude in self.pattern_amplitudes.values()]
        )
        patterns = []
        probabilities = []
        for pattern, amplitude in self.pattern_amplitudes.items():
            patterns.append(pattern)
            probabilities.append(amplitude / total_amplitude)
        choice = np.random.choice(patterns, p=probabilities)
        self.pattern_amplitudes = {choice: 1.0}
        self._update_entropy()

    def set_possibilities(self, allowed_possibilities: Set[Pattern]) -> bool:
        old_possibilities = self.pattern_amplitudes
        needs_update = len(allowed_possibilities) != len(old_possibilities)
        if not needs_update:
            return False
        self.pattern_amplitudes = {}
        for pattern in allowed_possibilities:
            self.pattern_amplitudes[pattern] = old_possibilities[pattern]
        self._update_entropy()
        return True

    def is_collapsed(self):
        possibilities = len(self.pattern_amplitudes)
        if possibilities == 0:
            raise Contradiction("cell with zero possibilities found")
        return possibilities == 1


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
        # propagation.
        self.observed_cells: List[Tuple[int, int]] = []
        # optimization to prevent linear passes over the entire graph when
        # determining cells with entropy.
        self.coords_with_entropy: Set[Tuple[int, int]] = set()
        self._make_cell_graph()

    def _make_cell_graph(self) -> None:
        """
        For each pixel in [height, width]
        1. make a node with an initial WaveCell that has all patterns available to it, update entropy
        2. make edges from each node to valid neighbors
        """
        for y, x in itertools.product(
            range(self.dimensions[0]), range(self.dimensions[1])
        ):
            coordinate = (y, x)
            new_cell = WaveCell(
                0.0,
                coordinate,
                {
                    pattern: float(frequency)
                    for pattern, frequency in self.source_patterns.frequencies.items()
                },
            )
            self.cells.add_node(coordinate, cell=new_cell)
            self.coords_with_entropy.add(coordinate)
        # now that all nodes have been added, we construct the edges
        for y, x in itertools.product(
            range(self.dimensions[0]), range(self.dimensions[1])
        ):
            coordinate = (y, x)
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for direction in directions:
                neighbor = (coordinate[0] + direction[0], coordinate[1] + direction[1])
                if (
                    neighbor[0] >= 0
                    and neighbor[0] < self.dimensions[0]
                    and neighbor[1] >= 0
                    and neighbor[1] < self.dimensions[1]
                ):
                    self.cells.add_edge(coordinate, neighbor)

    def get_cells(self) -> Generator[WaveCell, None, None]:
        for _, cell in self.cells.nodes.data("cell"):  # type: ignore
            yield cell

    def get_minimum_entropy_uncollapsed_cell(self) -> WaveCell:
        cells_with_entropy = [
            self.cells.nodes[coord]["cell"] for coord in self.coords_with_entropy
        ]
        uncollapsed_cells = [
            cell for cell in cells_with_entropy if not cell.is_collapsed()
        ]
        entropy_sorted_cells = sorted(uncollapsed_cells, key=lambda x: x.entropy)
        return entropy_sorted_cells[0]

    def observe(self) -> None:
        """
        Find minimum entropy cell and observe one of its patterns using
        random choice weighted by the frequency of the pattern. Add it to the
        list of observed cells.
        """
        min_entropy_cell = self.get_minimum_entropy_uncollapsed_cell()
        min_entropy_cell.observe()
        self.observed_cells.append(min_entropy_cell.image_coordinates)

    def propagate(self) -> None:
        """
        starting with the most recently observed cell as the source,
        update the allowlist of patterns for other cells in
        depth-first-search order, by propagating constraints
        from the source pattern adjacencies.
        """
        assert len(self.observed_cells) > 0
        last_observed = self.observed_cells[-1]
        propagation_stack = [last_observed]
        while len(propagation_stack) > 0:
            propagater_coord = propagation_stack.pop()
            propagater_cell: WaveCell = self.cells.nodes[propagater_coord]["cell"]
            if propagater_cell.is_collapsed():
                self.coords_with_entropy.discard(propagater_coord)
            neighbors = list(self.cells.neighbors(propagater_coord))
            for neighbor_coord in neighbors:
                neighbor_coord: Tuple[int, int] = neighbor_coord
                neighbor_cell: WaveCell = self.cells.nodes[neighbor_coord]["cell"]
                if not neighbor_cell.is_collapsed():
                    # We can only influence the neighbor cell if it has not been collapsed
                    # 1. query allowed patterns from source patterns
                    # 2. update the cell to consider only the intersection of patterns
                    # that it still has with the allowed patterns.
                    # 3. Add the influenced cell to the propagation stack
                    direction = (
                        int(propagater_coord[0] - neighbor_coord[0]),
                        int(propagater_coord[1] - neighbor_coord[1]),
                    )
                    allowed_patterns_for_neighbor = {
                        allowed_pattern
                        for pattern in propagater_cell.possibilities()
                        for allowed_pattern in self.source_patterns.adjacencies[
                            pattern
                        ][Directions(direction)]
                    }
                    possible_patterns_for_neighbor = neighbor_cell.possibilities()
                    allowed_possibilities = (
                        possible_patterns_for_neighbor & allowed_patterns_for_neighbor
                    )
                    possibilities_collapsed = neighbor_cell.set_possibilities(allowed_possibilities)
                    if possibilities_collapsed:
                        propagation_stack.append(neighbor_coord)
                if propagater_cell.is_collapsed():
                    # if we've been collapsed we can no longer influnce neighbors. We prevent
                    # traversal to and from this node as an optimization, since it no longer
                    # does anything useful in propagation
                    self.cells.remove_edge(propagater_coord, neighbor_coord)

    def is_fully_collapsed(self) -> bool:
        return len(self.coords_with_entropy) == 0

    def produce_image(self) -> NDArray[np.uint8]:
        image = np.array(
            [
                [[0, 0, 0, 0] for _ in range(self.dimensions[1])]
                for _ in range(self.dimensions[0])
            ],
            dtype=np.uint8
        )
        for cell in self.get_cells():
            image[cell.image_coordinates] = cell.get_pixel()
        return image

    def current_progress(self):
        return 1.0 - float(len(self.coords_with_entropy)) / float(self.dimensions[0] * self.dimensions[1])
