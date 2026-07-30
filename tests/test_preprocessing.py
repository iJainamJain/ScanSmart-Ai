import numpy as np

from src.preprocessing.loader import resize_image


def test_resize_image_keeps_small_images_unchanged():
    small = np.zeros((100, 200, 3), dtype=np.uint8)
    resized, scale = resize_image(small, max_dimension=1500)

    assert resized.shape == small.shape
    assert scale == 1.0


def test_resize_image_scales_down_large_images():
    large = np.zeros((2000, 1000, 3), dtype=np.uint8)
    resized, scale = resize_image(large, max_dimension=1000)

    assert max(resized.shape[:2]) == 1000
    assert scale == 0.5
