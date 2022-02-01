"""A module for converting a data source to TFRecords."""

import glob

from pathlib import Path
from math import ceil

import tensorflow as tf
from tqdm import tqdm

from .feature import features_to_items, FEATURE_KEYS_MAP
from ..config import OUTPUT_SCHEMA, NUM_DETECTING_OBJECTS, BATCH_SIZE


def load_tfrecords(
    tfrecords_dir: Path,
    output_schema: str = OUTPUT_SCHEMA,
    num_detecting_objects: int = NUM_DETECTING_OBJECTS,
    batch_size: int = BATCH_SIZE,
    steps_per_epoch=0,
    verbose: bool = False,
):
    """Returns TFRecord dataset and the number of steps per epoch.

    Args:
        tfrecords_dir (Path):
            The path to directory with TFRecords.
        output_schema (str):
            The output schema defines the format of output data (default is
            just categories for every object present on the input images).
            Read this function description to understand how to define the
            output schema.
        num_detecting_objects (int):
            The number of detecting objects per input image. It must greater
            than 1.
        batch_size (int):
            The batch size. It must greater than 0.
        steps_per_epoch (int):
            The number of steps per epoch. If the number of steps is set to
            0, the internal process will define it by scanning all TFrecords.
            This process might take some time, so it is advisable to set the
            `verbose` flag to `True`, to monitor the progress.
        verbose (bool):
            The flag to set a verbose mode.

    Returns:
        dataset (TFRecordDataset):
            The dataset to pass into tensorflow training function.
        steps_per_epoch (int):
            The number of steps per epoch provided or computed (if it was
            not specified as the function argument).

    Raises:
        ValueError:
            If the function argument values are incorrect.
    """

    if len(output_schema) == 0:
        raise ValueError("The output schema is empty.")
    if output_schema.count("I") > 1:
        raise ValueError(
            "The output schema contains multiple 'I' "
            "(image arrays definition)."
        )
    if output_schema.count("B") > 1:
        raise ValueError(
            "The output schema contains multiple 'B' "
            "(bounding boxes definition)."
        )
    if output_schema.count("M") > 1:
        raise ValueError(
            "The output schema contains multiple 'M' " "(masks definition)."
        )
    if output_schema.count("C") > 1:
        raise ValueError(
            "The output schema contains multiple 'C' "
            "(category ids definition)."
        )

    def record_parser(proto):
        parsed_features = tf.io.parse_single_example(proto, FEATURE_KEYS_MAP)
        _, image, bboxes, masks, category_ids = features_to_items(
            parsed_features, num_detecting_objects
        )

        buffer = [[]]
        for element in output_schema:
            if element == "I":
                buffer[-1].append(image)
            elif element == "B":
                buffer[-1].append(bboxes)
            elif element == "M":
                buffer[-1].append(masks)
            elif element == "C":
                buffer[-1].append(category_ids)
            elif element == ",":
                if len(buffer[-1]) > 0:
                    buffer.append([])
                else:
                    raise ValueError(
                        "The output schema contains two consequent commas."
                    )
            else:
                raise ValueError(
                    f"The output schema contains unknown element '{element}'."
                )

        X = image
        y = []
        for pack in buffer:
            if len(pack) > 1:
                y.append(tf.concat(pack, axis=1))
            else:
                y.append(pack[0])

        if len(y) > 1:
            return X, tuple(y)
        else:
            return X, *y

    tfrecord_files = glob.glob(str(tfrecords_dir / "part-*.tfrecord"))

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
