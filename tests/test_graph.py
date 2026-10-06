import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("skimage")

from graph import contour_adjacent_pairs, is_valid_neighbor_contour  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorised_adjacency_matches_reference(seed):
    rng = np.random.default_rng(seed)
    # blocky label map with a few regions, plus a random contour mask
    labels = np.kron(rng.integers(0, 6, (6, 7)), np.ones((4, 3), int))
    contour = rng.random(labels.shape) < 0.3
    fast = contour_adjacent_pairs(labels, contour)
    ids = np.unique(labels)
    ref = {(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
           if is_valid_neighbor_contour(a, b, labels, contour)}
    assert fast == {(int(a), int(b)) for a, b in ref}
