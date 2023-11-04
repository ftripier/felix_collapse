from typing import List, Generator, Tuple
from dataclasses import dataclass


class Adjacencies:
    def __init__(self) -> None:
        pass

    def contains(self, edge: Tuple["Pattern", "Pattern"]) -> bool:
        raise NotImplementedError


@dataclass
class Pattern:
    count: int


class Cell:
    def __init__(self) -> None:
        self._neighbors: List["Cell"] = []

    def neighbors(self) -> Generator["Cell", None, None]:
        for neighbor in self._neighbors:
            yield neighbor

    def original_pattern(self) -> Pattern:
        raise NotImplementedError

    def domain(self) -> List[Pattern]:
        raise NotImplementedError

    def remove_pattern(self, pattern: Pattern) -> None:
        raise NotImplementedError


class WaveMatrix:
    cells: List[Cell]

    def __init__(self) -> None:
        self.cells = []

    def most_recently_observed_cell(self) -> Cell:
        raise NotImplementedError
