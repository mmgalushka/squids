"""
A module for converting a data source to TFRecords.
"""
from __future__ import annotations

import glob
from pathlib import Path
from math import ceil

from tqdm import tqdm

import tensorflow as tf


# ------------------------------------------------------------------------------
#
#    D A T A    G E N E R A T O R
#
# ------------------------------------------------------------------------------


def create_generator(
    tfrecords_path: Path,
    detecting_categories: list,
    num_detecting_objects: int = 10,
    batch_size: int = 128,
    steps_per_epoch=0,
    verbose: bool = False,
):
    def record_parser(proto):
        keys_to_features = {
            "image/shape": tf.io.FixedLenSequenceFeature(
                [], tf.int64, allow_missing=True
            ),
            "image/content": tf.io.FixedLenSequenceFeature(
                [], tf.string, allow_missing=True
            ),
            "bboxes/number": tf.io.FixedLenSequenceFeature(
                [], tf.int64, allow_missing=True
            ),
            "bboxes/data": tf.io.FixedLenSequenceFeature(
                [], tf.int64, allow_missing=True
            ),
            "categories/number": tf.io.FixedLenSequenceFeature(
                [], tf.int64, allow_missing=True
            ),
            "categories/data": tf.io.FixedLenSequenceFeature(
                [], tf.string, allow_missing=True
            ),
        }
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)

        # Gets data artefacts.
        image = _feature_to_image(parsed_features)
        bboxes = _feature_to_bboxes(parsed_features, num_detecting_objects)
        categories = _feature_to_categories(
            parsed_features, detecting_categories, num_detecting_objects
        )

        # Assembling the model input (X) and output (y).
        X = image
        y = tf.concat([bboxes, categories], axis=1)

        return X, (X, y)

    # Selects all TFRecord files stored in the specified directory.
    tfrecord_files = glob.glob(str(tfrecords_path / "part-*.tfrecord"))

    # --- Create Data Generator ---
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(record_parser, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)

    # --- Define Steps Per Epoch ---
    if steps_per_epoch == 0:
        # If training steps per epoch have set to 0, it needs to be
        # computed.

        # Starts counting the total number of TFREcords in all partitions.
        if verbose:
            pbar = tqdm(total=len(tfrecord_files))
        records_count = 0
        for tfrecord_files in tfrecord_files:
            records_count += sum(
                1 for _ in tf.data.TFRecordDataset(tfrecord_files)
            )
            if verbose:
                pbar.update(1)

        # Computes a number of steps per epoch by dividing a number of
        # records by the batch size.
        steps_per_epoch = ceil(records_count / batch_size)

    return dataset, steps_per_epoch


def _feature_to_image(parsed_features: dict) -> tf.Tensor:
    image = tf.io.decode_raw(parsed_features["image/content"], tf.uint8)
    image = tf.reshape(image, parsed_features["image/shape"])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _feature_to_bboxes(
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


def _feature_to_categories(
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
