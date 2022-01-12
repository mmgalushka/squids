"""
A module for converting a data source to TFRecords.
"""
from __future__ import annotations


import glob

from pathlib import Path
from math import ceil

import tensorflow as tf
from tqdm import tqdm

from .feature import feature_to_item, KEY_FEATURE_MAP


def get_tfrecords_generator(
    tfrecords_path: Path,
    num_detecting_objects: int = 10,
    batch_size: int = 128,
    steps_per_epoch=0,
    verbose: bool = False,
):
    def record_parser(proto):
        parsed_features = tf.io.parse_single_example(proto, KEY_FEATURE_MAP)
        _, image, bboxes, segmentations, category_ids = feature_to_item(
            parsed_features, num_detecting_objects
        )

        X = image
        y = tf.concat([bboxes, segmentations, category_ids], axis=1)

        return X, (X, y)

    tfrecord_files = glob.glob(str(tfrecords_path / "part-*.tfrecord"))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(record_parser, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)

    if steps_per_epoch == 0:
        # If training steps per epoch have been set to 0, its value needs
        # to be computed

        if verbose:
            pbar = tqdm(total=len(tfrecord_files))
        records_count = 0
        for tfrecord_files in tfrecord_files:
            records_count += sum(
                1 for _ in tf.data.TFRecordDataset(tfrecord_files)
            )
            if verbose:
                pbar.update(1)

        steps_per_epoch = ceil(records_count / batch_size)

    return dataset, steps_per_epoch
