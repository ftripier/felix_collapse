from .wave_matrix import WaveMatrix, Adjacencies


def propagate(wave_matrix: WaveMatrix, adjacencies: Adjacencies) -> WaveMatrix:
    propagation_stack = [wave_matrix.most_recently_observed_cell()]

    while len(propagation_stack) > 0:
        curr_cell = propagation_stack.pop()
        for neighbor in curr_cell.neighbors():
            for neighbor_pattern in neighbor.domain():
                if not adjacencies.contains(
                    (curr_cell.original_pattern(), neighbor_pattern)
                ):
                    neighbor_pattern.count -= 1
                if neighbor_pattern.count <= 0:
                    neighbor.remove_pattern(neighbor_pattern)
                    propagation_stack.append(neighbor)
    return wave_matrix
