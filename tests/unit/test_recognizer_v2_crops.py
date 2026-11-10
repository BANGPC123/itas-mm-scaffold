import random
import cv2
import numpy as np
from src.perception.recognizer_v2_crops import augment, exact_crop, make_entry


def image_with_asymmetric_sign():
    image = np.full((200, 200, 3), 90, np.uint8)
    cv2.rectangle(image, (60, 60), (140, 140), (200, 200, 200), -1)
    cv2.rectangle(image, (60, 60), (100, 140), (20, 20, 220), -1)  # red left half
    return image


def test_exact_crop_reproduces_runtime_floor_ceil_crop():
    image = image_with_asymmetric_sign()
    entry = make_entry(image, [59.4, 60.2, 140.1, 139.8])
    expected = image[60:140, 59:141]
    assert exact_crop(entry).shape == expected.shape
    assert np.array_equal(exact_crop(entry), expected)


def test_entry_is_clipped_at_image_border_and_rejects_empty_boxes():
    image = image_with_asymmetric_sign()
    entry = make_entry(image, [-10, -10, 30, 30])
    assert exact_crop(entry).shape[:2] == (30, 30)
    assert make_entry(image, [50, 50, 50, 80]) is None


def test_large_crops_are_downscaled_but_keep_inner_box_inside():
    image = np.zeros((900, 900, 3), np.uint8)
    entry = make_entry(image, [100, 100, 800, 800])
    assert max(entry['img'].shape[:2]) <= 256
    x1, y1, x2, y2 = entry['inner']
    assert 0 <= x1 < x2 <= entry['img'].shape[1] and 0 <= y1 < y2 <= entry['img'].shape[0]


def test_augmentation_never_mirrors_the_sign():
    entry = make_entry(image_with_asymmetric_sign(), [60, 60, 140, 140])
    rng = random.Random(7)
    for _ in range(300):
        crop = augment(entry, rng)
        assert crop.ndim == 3 and crop.shape[2] == 3 and min(crop.shape[:2]) >= 2
        left = crop[:, :crop.shape[1] // 3, 2].astype(float).mean() - crop[:, :crop.shape[1] // 3, 0].astype(float).mean()
        right = crop[:, -crop.shape[1] // 3:, 2].astype(float).mean() - crop[:, -crop.shape[1] // 3:, 0].astype(float).mean()
        # The red half stays on the left: red minus blue is larger there than on the right.
        assert left > right
