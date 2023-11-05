import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from typing import Tuple, Set, Generator
from .source_patterns import Pattern, SourcePatterns


@dataclass
class WaveCell:
    entropy: float
    image_coordinates: Tuple[int, int]
    pattern_amplitudes: defaultdict[Pattern, float]


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

    def __init__(self, dimenstions: Tuple[int, int], source_patterns: SourcePatterns) -> None:
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
        raise NotImplementedError
