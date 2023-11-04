# pyright: reportUnknownVariableType=false

import imageio.v3 as iio
from pathlib import Path
from numpy.typing import NDArray
import numpy as np


def load_png(file: Path) -> NDArray[np.byte]:
    return iio.imread(file)


def save_png(image: NDArray[np.byte], output_file: Path):
    iio.imwrite(output_file, image)
