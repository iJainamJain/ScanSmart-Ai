import numpy as np

from src.perspective.transform import order_points


def test_order_points_returns_tl_tr_br_bl():
    # Deliberately shuffled corners of a 100x200 rectangle.
    points = np.array(
        [
            [100, 0],    # top-right
            [0, 0],      # top-left
            [100, 200],  # bottom-right
            [0, 200],    # bottom-left
        ],
        dtype=np.float32,
    )

    ordered = order_points(points)

    assert np.array_equal(ordered[0], [0, 0])
    assert np.array_equal(ordered[1], [100, 0])
    assert np.array_equal(ordered[2], [100, 200])
    assert np.array_equal(ordered[3], [0, 200])
