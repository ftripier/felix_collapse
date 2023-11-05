from numpy.typing import NDArray
import numpy as np
from .source_patterns import SourcePatterns
from .wavefunction import Wavefunction, Contradiction
from typing import Tuple


class WavefunctionCollapse:
    """Runs wavefunction collapse. Returns an image. Does no I/O."""

    def __init__(self, source_texture: NDArray[np.int32]) -> None:
        self.source_texture = source_texture
        self.source_patterns = SourcePatterns(source_texture)

    def run(
        self, requested_dimensions: Tuple[int, int], trials=10
    ) -> NDArray[np.int32]:
        trials_left = trials
        while trials_left > 0:
            try:
                wvf = Wavefunction(requested_dimensions, self.source_patterns)
                while not wvf.is_fully_collapsed():
                    wvf.observe()
                    wvf.propagate()
                return wvf.produce_image()
            except Contradiction:
                trials_left -= 1
        raise Exception("ran into too many contradictions :(")
