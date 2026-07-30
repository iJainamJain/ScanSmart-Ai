import numpy as np

from src.morphology.operations import closing, dilate, erode, opening


def test_erode_removes_a_small_isolated_speck():
    image = np.zeros((50, 50), dtype=np.uint8)
    image[25, 25] = 255  # 1px speck, smaller than the 3x3 kernel

    result = erode(image, kernel_size=3)

    assert result[25, 25] == 0


def test_dilate_grows_white_region():
    image = np.zeros((50, 50), dtype=np.uint8)
    image[25, 25] = 255

    result = dilate(image, kernel_size=3)

    assert result[24, 25] == 255
    assert result[26, 25] == 255


def test_opening_removes_speck_but_keeps_large_blob():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 255  # large blob
    image[5, 5] = 255  # isolated speck far from the blob

    result = opening(image, kernel_size=3)

    assert result[50, 50] == 255
    assert result[5, 5] == 0


def test_closing_fills_a_small_hole_inside_a_blob():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 255
    image[45:47, 45:47] = 0  # small hole inside the blob

    result = closing(image, kernel_size=5)

    assert result[46, 46] == 255
