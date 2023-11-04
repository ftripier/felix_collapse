from numpy.typing import NDArray
import numpy as np
from .source_patterns import SourcePatterns


class WavefunctionCollapseInstance:
    """Runs wavefunction collapse. Returns an image. Does no I/O."""

    def __init__(self, source_texture: NDArray[np.byte]) -> None:
        self.source_texture = source_texture
        self.source_patterns = SourcePatterns(self.source_texture)

    def run(self) -> NDArray[np.byte]:
        raise NotImplementedError
