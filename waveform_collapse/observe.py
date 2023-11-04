from .wave_matrix import Cell, WaveMatrix


def any_cell_has_zero_possibilities(wave_matrix: WaveMatrix) -> bool:
    raise NotImplementedError


def all_cells_have_exactly_one_possibility(wave_matrix: WaveMatrix) -> bool:
    raise NotImplementedError


def find_minimum_entropy(wave_matrix: WaveMatrix) -> Cell:
    raise NotImplementedError


def assign_weighted_random_sample(cell: Cell) -> None:
    """
    I need this to mutate the original cell. The data structures involved here
    are still not clear.
    """
    raise NotImplementedError


def observe(wave_matrix: WaveMatrix) -> WaveMatrix:
    if any_cell_has_zero_possibilities(wave_matrix):
        raise Exception("Cell with no possibilities detected")
    if all_cells_have_exactly_one_possibility(wave_matrix):
        return wave_matrix
    cell = find_minimum_entropy(wave_matrix)
    assign_weighted_random_sample(cell)
    return wave_matrix
