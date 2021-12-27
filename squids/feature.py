"""
A module for handling TFRecords features.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import tensorflow as tf

# -----------------------------------------------------------------------------
# Image -> Feature -> Image transformars
# -----------------------------------------------------------------------------


def image_to_feature(image: Image, width: int, height: int):
    """Returns a bytes_list from a string / byte."""
    # image = Image.open(fp)
    # if isinstance(width, int) and isinstance(height, int):
    #     array = np.array(image.resize((width, height)))
    # elif width is None and height is None:
    #     array = np.array(image)
    # else:
    #     raise ValueError(
    #         "Invalid arguments for resizing an image. Both arguments "
    #         "representing image width and height must be either integer "
    #         f"or None. But received width is {type(width)} and height is "
    #         f"{type(height)}."
    #     )

    # array = np.array(image.resize((width, height)))
    array = np.array(image)
    return {
        "image/shape": tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(array.shape))
        ),
        "image/content": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[array.tostring()])
        ),
    }


def feature_to_image(parsed_features: dict) -> tf.Tensor:
    image = tf.io.decode_raw(parsed_features["image/content"], tf.uint8)
    image = tf.reshape(image, parsed_features["image/shape"])
    image = tf.cast(image, tf.float32) / 255.0
    return image


# -----------------------------------------------------------------------------
# BBoxes -> Feature -> BBoxes transformars
# -----------------------------------------------------------------------------


def bboxes_to_feature(bboxes: list):
    """Returns an int64_list from a bool / enum / int / uint."""
    data = []
    for bbox in bboxes:
        data.extend(bbox)

    return {
        "bboxes/number": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[len(bboxes)])
        ),
        "bboxes/data": tf.train.Feature(
            int64_list=tf.train.Int64List(value=data)
        ),
    }


def feature_to_bboxes(
    parsed_features, num_detecting_objects: int
) -> tf.Tensor:
    # Loads all binding boxes defined in a TFRecord.

    n = parsed_features["bboxes/number"][0]
    # let's assume it is 2

    bboxes = parsed_features["bboxes/data"]
    bboxes = tf.reshape(bboxes, (-1, 4))
    # [[1 2 3 4]
    #  [5 6 7 8]], shape=(2, 4), dtype=int32

    if n < num_detecting_objects:
        # If the obtained number of record binding boxes is less than the
        # detection capacity, then the binding boxes must be padded with
        # [0, 0, 0, 0].
        bboxes = tf.pad(
            bboxes, [[0, num_detecting_objects - n], [0, 0]], constant_values=0
        )
        # [[1 2 3 4]
        #  [5 6 7 8]
        #  [0 0 0 0]
        #  [0 0 0 0]], shape=(4, 4), dtype=int32)
        # values = [b'rectangle' b'triangle' b'[UNK]' b'[UNK]'], shape=(4,)
    elif n > 4:
        # If the obtained number of record binding boxes is greater than the
        # detection capacity, then the  binding boxes list must be sliced.
        bboxes = tf.slice(bboxes, [0, 0], [num_detecting_objects, 4])

    bboxes = tf.cast(bboxes, dtype=tf.float32)
    # [[1. 2. 3. 4.]
    #  [5. 6. 7. 8.]
    #  [0. 0. 0. 0.]
    #  [0. 0. 0. 0.]], shape=(4, 4), dtype=float32
    return bboxes


# -----------------------------------------------------------------------------
# Segments -> Feature -> Segments transformars
# -----------------------------------------------------------------------------


def segments_to_feature(segments: list):
    """Returns an int64_list from a bool / enum / int / uint."""
    schema = []
    data = []
    for segment in segments:
        schema.append(len(segment))
        data.extend(segment)

    return {
        "segments/schema": tf.train.Feature(
            int64_list=tf.train.Int64List(value=schema)
        ),
        "segments/data": tf.train.Feature(
            float_list=tf.train.FloatList(value=data)
        ),
    }


# -----------------------------------------------------------------------------
# Categories -> Feature -> Categories transformars
# -----------------------------------------------------------------------------


def categories_to_feature(categories: list):
    """Returns an int64_list from a bool / enum / int / uint."""
    return {
        "categories/number": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[len(categories)])
        ),
        # "categories/data": tf.train.Feature(
        #     bytes_list=tf.train.BytesList(
        #         value=[category.encode("utf-8") for category in categories]
        #     )
        # ),
    }


def feature_to_categories(
    parsed_features: dict,
    detecting_categories: list,
    num_detecting_objects: int,
) -> tf.Tensor:
    # Loads all categories values defined in a TFRecord.

    n = parsed_features["categories/number"][0]
    # lets assume it is 2

    categories = parsed_features["categories/data"]
    # [b'rectangle' b'triangle'], shape=(4,), dtype=string

    # if n < num_detecting_objects:
    #     # If the obtained number of record categories less than the
    #     # detection capacity, then the categories list must be padded with
    #     # "unknown category".
    #     categories = tf.pad(
    #         categories, [[0, num_detecting_objects - n]],
    #         constant_values='[UNK]')
    #     # [b'rectangle' b'triangle' b'[UNK]' b'[UNK]'], shape=(4,),
    #       dtype=string
    # elif n > num_detecting_objects:
    #     # If the obtained number of record categories greater than the
    #     # detection capacity, then the categories list must be sliced.
    #     categories = tf.slice(categories, [0], [num_detecting_objects])

    categories = tf.pad(
        categories, [[0, num_detecting_objects - n]], constant_values="[UNK]"
    )

    categories = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=detecting_categories
    )(categories)
    # [2 3 1 1], shape=(4,), dtype=int64

    categories = categories - tf.constant(1, dtype=tf.int64)
    # [1 2 0 0], shape=(4,), dtype=int64

    categories = tf.one_hot(categories, len(detecting_categories) + 1)
    # [[0. 1. 0.]
    #  [0. 0. 1.]
    #  [1. 0. 0.]
    #  [1. 0. 0.]], shape=(4, 3), dtype=float32
    return categories
