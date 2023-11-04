from .wave_matrix import WaveMatrix, Adjacencies
from .observe import observe
from .propagate import propagate


def contains_contradictions(wave_matrix: WaveMatrix) -> bool:
    # TODO: make this a method of wave matrix?
    raise NotImplementedError


def all_domains_collapsed(wave_matrix: WaveMatrix) -> bool:
    # TODO: make this a method of wave matrix?
    raise NotImplementedError


def solve(adjacencies: Adjacencies, original_wave_matrix: WaveMatrix):
    curr_wm = original_wave_matrix
    while not contains_contradictions(curr_wm):
        curr_wm = propagate(curr_wm, adjacencies)
        if all_domains_collapsed(curr_wm):
            return curr_wm
        curr_wm = observe(curr_wm)
    raise Exception("Generation attempt failed")
