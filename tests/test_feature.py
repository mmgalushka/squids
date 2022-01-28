"""
Test for the feature read/write functions form `squids/feature.py`.
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from squids.config import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS
from squids.tfrecords.feature import items_to_features, features_to_items


EPS = tf.constant(0.001, tf.float32)

# -----------------------------------------------------------------------------
# Tests: Image -> Feature -> Image transformars
# -----------------------------------------------------------------------------


def test_items_to_features():
    """Tests the `items_to_features` function."""
    expected_image_id = 123
    expected_image = Image.new(mode="RGB", size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    expected_image_width = IMAGE_WIDTH
    expected_image_height = IMAGE_HEIGHT
    expected_bboxes = [[1.0, 2.0, 3.0, 4.0]]
    expected_segmentations = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    expected_mask = Image.new(
        "RGB",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        "#000000",
    )
    drawing = ImageDraw.Draw(expected_mask)
    drawing.polygon(
        expected_segmentations[0],
        fill="#ffffff",
        outline="#ffffff",
    )
    expected_category_ids = [1]
    expected_category_max_id = 2

    expected_features = {
        "image/id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[expected_image_id])
        ),
        "image/width": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[expected_image_width])
        ),
        "image/height": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[expected_image_height])
        ),
        "image/data": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.array(expected_image).tostring()]
            )
        ),
        "annotations/number": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[1])
        ),
        "bboxes/data": tf.train.Feature(
            float_list=tf.train.FloatList(value=expected_bboxes[0])
        ),
        "masks/data": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.array(expected_mask).tostring()]
            )
        ),
        "category/ids": tf.train.Feature(
            int64_list=tf.train.Int64List(value=expected_category_ids)
        ),
        "category/max": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[expected_category_max_id])
        ),
    }

    actual_features = items_to_features(
        expected_image_id,
        expected_image,
        expected_image_width,
        expected_image_height,
        expected_bboxes,
        expected_segmentations,
        expected_category_ids,
        expected_category_max_id,
    )

    assert actual_features == expected_features


def test_features_to_items():
    """Tests the `features_to_items` function."""
    expected_image_id = 123
    expected_image = Image.new(mode="RGB", size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    expected_image_arr = np.array(expected_image)
    expected_bboxes = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    expected_segmentations = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ]
    expected_masks = [
        Image.new(
            "RGB",
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            "#000000",
        ),
        Image.new(
            "RGB",
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            "#000000",
        ),
    ]
    drawing = ImageDraw.Draw(expected_masks[0])
    drawing.polygon(
        expected_segmentations[0],
        fill="#ffffff",
        outline="#ffffff",
    )
    drawing = ImageDraw.Draw(expected_masks[1])
    drawing.polygon(
        expected_segmentations[1],
        fill="#ffffff",
        outline="#ffffff",
    )
    expected_masks = [
        np.array(expected_masks[0]).flatten(),
        np.array(expected_masks[1]).flatten(),
    ]
    expected_category_ids = [1, 2]
    expected_category_onehot = [[0, 1, 0], [0, 0, 1]]
    expected_category_max_id = 2

    features = {
        "image/id": tf.constant([expected_image_id], tf.int64),
        "image/width": tf.constant([IMAGE_WIDTH], tf.int64),
        "image/height": tf.constant([IMAGE_HEIGHT], tf.int64),
        "image/data": tf.constant(
            [np.array(expected_image).tostring()], tf.string
        ),
        "annotations/number": [2],
        "bboxes/data": tf.constant(
            np.array(expected_bboxes).flatten(), tf.float32
        ),
        "masks/data": tf.constant(
            [
                np.concatenate(
                    (expected_masks[0], expected_masks[1])
                ).tostring()
            ],
            tf.string,
        ),
        "category/ids": expected_category_ids,
        "category/max": [expected_category_max_id],
    }

    (
        actual_image_id,
        actual_image_arr,
        actual_bboxes,
        actual_masks,
        actual_category_onehot,
    ) = features_to_items(features)

    assert actual_image_id == expected_image_id
    assert actual_image_arr.shape == (
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_CHANNELS,
    )
    assert tf.math.less(
        tf.reduce_sum(tf.abs(actual_image_arr - expected_image_arr)), EPS
    )
    assert actual_image_arr.shape == (
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_CHANNELS,
    )
    assert tf.math.less(
        tf.reduce_sum(tf.abs(actual_bboxes - expected_bboxes)), EPS
    )
    assert actual_masks.shape == (
        2,
        IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS,
    )
    assert tf.math.less(
        tf.reduce_sum(tf.abs(actual_masks - expected_masks)), EPS
    )
    assert actual_category_onehot.shape == (2, 3)
    assert tf.math.less(
        tf.reduce_sum(
            tf.abs(actual_category_onehot - expected_category_onehot)
        ),
        EPS,
    )

    for expected_num_detecting_objects in [5, 1]:
        (
            actual_image_id,
            actual_image_arr,
            actual_bboxes,
            actual_masks,
            actual_category_onehot,
        ) = features_to_items(
            features, num_detecting_objects=expected_num_detecting_objects
        )
        assert actual_image_arr.shape == (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            IMAGE_CHANNELS,
        )
        assert actual_bboxes.shape == (expected_num_detecting_objects, 4)
        assert actual_masks.shape == (
            expected_num_detecting_objects,
            IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS,
        )
        assert actual_category_onehot.shape == (
            expected_num_detecting_objects,
            3,
        )
