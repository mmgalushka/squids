"""
A module for converting a data source to TFRecords.
"""
from __future__ import annotations

import os
import glob
import cmd
from pathlib import Path
from math import ceil
from typing import Iterator

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from .dataset import (
    DATASET_DIR,
    CsvIterator,
    CocoIterator,
)
from .feature import item_to_feature, feature_to_item
from .image import IMAGE_WIDTH, IMAGE_HEIGHT
from .color import CATEGORY_COLORS


# -----------------------------------------------------------------------------
# TFRecords CSV/COCO Transformers
# -----------------------------------------------------------------------------


def items_to_tfrecords(
    output_dir: Path,
    instance_file: Path,
    items: Iterator,
    tfrecords_size: int,
    image_width: int,
    image_height: int,
    verbose: bool,
):
    def get_example(item):
        image_id = item["image"]["id"]
        img = item["image"]["data"]
        annotations = item["annotations"]
        categories = item["categories"]

        category_max_id = max(list(categories.keys()))

        bboxes = []
        segmentations = []
        category_ids = []
        for annotation in annotations:
            if annotation["iscrowd"] == 0:

                bboxes.append(annotation["bbox"])
                segmentations.append(annotation["segmentation"][0])
                category_ids.append(annotation["category_id"])

        feature = item_to_feature(
            image_id,
            img,
            image_width,
            image_height,
            bboxes,
            segmentations,
            category_ids,
            category_max_id,
        )
        return tf.train.Example(features=tf.train.Features(feature=feature))

    tfrecords_dir = output_dir / instance_file.stem
    tfrecords_dir.mkdir(exist_ok=True)

    # The TFRecords writer.
    writer = None
    # The index for the next TFRecords partition.
    part_index = -1
    # The count of how many records stored in the TFRecords files. It
    # is set here to maximum capacity (as a trick) to make the "if"
    # condition in the loop equals to True and start 0 - partition.
    part_count = tfrecords_size

    # Initializes the progress bar of verbose mode is on.
    if verbose:
        pbar = tqdm(total=len(items))

    for item in items:
        if item:
            if part_count >= tfrecords_size:
                # The current partition has been reached the maximum capacity,
                # so we need to start a new one.
                if writer is not None:
                    # Closes the existing TFRecords writer.
                    writer.close()
                part_index += 1
                writer = tf.io.TFRecordWriter(
                    str(tfrecords_dir / f"part-{part_index}.tfrecord")
                )
                part_count = 0

            example = get_example(item)
            if example:
                writer.write(example.SerializeToString())
        part_count += 1

        # Updates the progress bar of verbose mode is on.
        if verbose:
            pbar.update(1)

    # Closes the existing TFRecords writer after the last row.
    writer.close()


def is_csv_input(input_dir: Path) -> bool:
    return set(os.listdir(input_dir)) == set(
        [
            "images",
            "instances_train.csv",
            "instances_test.csv",
            "instances_val.csv",
            "categories.json",
        ]
    )


def is_coco_input(input_dir: Path) -> bool:
    root_artifacts = os.listdir(input_dir)
    if "annotations" in root_artifacts:
        annotations_artifacts = os.listdir(input_dir / "annotations")
        stems_artifacts = [
            Path(artifact).stem for artifact in annotations_artifacts
        ]
        return set(stems_artifacts).issubset(set(root_artifacts))
    return False


def create_tfrecords(
    dataset_dir: str = DATASET_DIR,
    selected_categories: list = [],
    tfrecords_dir: str = None,
    tfrecords_size: int = 256,
    tfrecords_image_width: int = IMAGE_WIDTH,
    tfrecords_image_height: int = IMAGE_HEIGHT,
    verbose: bool = False,
):

    input_dir = Path(dataset_dir)
    if not input_dir.exists():
        raise FileNotFoundError(str(input_dir))

    if tfrecords_dir is None:
        output_dir = input_dir.parent / (input_dir.name + "-tfrecords")
    else:
        output_dir = Path(tfrecords_dir)
    output_dir.mkdir(exist_ok=True)

    if is_csv_input(input_dir):
        for instance_file in input_dir.rglob("*.csv"):
            items_to_tfrecords(
                output_dir,
                instance_file,
                CsvIterator(instance_file, selected_categories),
                tfrecords_size,
                tfrecords_image_width,
                tfrecords_image_height,
                verbose,
            )
    elif is_coco_input(input_dir):
        for instance_file in (input_dir / "annotations").rglob("*.json"):
            items_to_tfrecords(
                output_dir,
                instance_file,
                CocoIterator(instance_file, selected_categories),
                tfrecords_size,
                tfrecords_image_width,
                tfrecords_image_height,
                verbose,
            )

    else:
        raise ValueError("invalid input data format.")


# -----------------------------------------------------------------------------
# TFRecords Generator/Explorer
# -----------------------------------------------------------------------------

KEY_FEATURE_MAP = {
    "image/id": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    "image/width": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    "image/height": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    "image/data": tf.io.FixedLenSequenceFeature(
        [], tf.string, allow_missing=True
    ),
    "annotations/number": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    "bboxes/data": tf.io.FixedLenSequenceFeature(
        [], tf.float32, allow_missing=True
    ),
    "segmentations/data": tf.io.FixedLenSequenceFeature(
        [], tf.string, allow_missing=True
    ),
    "category_ids/data": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    "category_ids/max": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
}


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


def get_tfrecords_explorer(
    tfrecords_path: Path,
):
    def record_parser(proto):
        parsed_features = tf.io.parse_single_example(proto, KEY_FEATURE_MAP)
        image_id, image, bboxes, segmentations, category_ids = feature_to_item(
            parsed_features
        )

        return image_id, image, bboxes, segmentations, category_ids

    tfrecord_files = glob.glob(str(tfrecords_path / "part-*.tfrecord"))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(record_parser, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(1)

    return dataset


# -----------------------------------------------------------------------------
# TFRecords Inspector/Viewer
# -----------------------------------------------------------------------------


def inspect_tfrecords(tfrecords_dir: str):
    cli = cmd.Cmd()
    input_path = Path(tfrecords_dir)
    if input_path.is_dir():
        tfrecord_files = glob.glob(str(input_path / "part-*.tfrecord"))
        if len(tfrecord_files) > 0:
            records = []
            batch = get_tfrecords_explorer(input_path)
            for image_id, _, _, _, category_onehots in batch:
                category_ids = [
                    np.argmax(category_onehot)
                    for category_onehot in category_onehots.numpy()[0]
                ]

                records.append(
                    str(image_id.numpy()[0])
                    + " ("
                    + ",".join(map(str, set(category_ids)))
                    + ")"
                )

            print(f"\n{tfrecords_dir} ({len(tfrecord_files)} parts)")
            cli.columnize(records)
            print(f"Total {len(records)} records")
        else:
            print(f"\n{tfrecords_dir} (no tfrecords found)")
            cli.columnize(os.listdir(input_path))
    else:
        raise FileNotFoundError(tfrecords_dir)


def inspect_tfrecord(
    tfrecords_dir: str,
    image_id: int,
    output_dir: str = ".",
    with_summary: bool = True,
    with_bboxes: bool = True,
    with_segmentations: bool = True,
):

    batch = get_tfrecords_explorer(Path(tfrecords_dir))
    for selected_image_id, image, bboxes, segmentations, category_ids in batch:
        if image_id == selected_image_id:
            output_image = Image.fromarray(
                tf.squeeze(image * 255, [0]).numpy().astype("uint8"), "RGB"
            )
            output_image_size = output_image.size
            output_image_shape = (
                output_image_size[0],
                output_image_size[1],
                3,
            )

            if with_summary:
                total_labeled_objects = len(category_ids.numpy()[0])
                available_categories_set = set(
                    [
                        np.argmax(category_onehot)
                        for category_onehot in category_ids.numpy()[0]
                    ]
                )

                summary = [
                    ["Image ID", image_id],
                    ["Image Type", "RGB"],
                    ["Image Shape", output_image_shape],
                    ["Total Labeled Objects", total_labeled_objects],
                    ["Available Categories Set", available_categories_set],
                ]
                print(tabulate(summary, headers=["Property", "Value"]))

            if with_segmentations:
                for mask, onehot in zip(
                    segmentations.numpy()[0],
                    category_ids.numpy()[0],
                ):

                    category_id = onehot.argmax()
                    category_color = CATEGORY_COLORS[
                        (category_id - 1) % len(CATEGORY_COLORS)
                    ]

                    blend_image = Image.new(
                        "RGBA", output_image_size, str(category_color)
                    )
                    mask_image = Image.fromarray(
                        (mask.reshape(output_image_shape)).astype("uint8"),
                        "RGB",
                    ).convert("L")
                    output_image = Image.blend(
                        output_image,
                        Image.composite(blend_image, output_image, mask_image),
                        0.7,
                    )

            if with_bboxes:
                draw = ImageDraw.Draw(output_image)
                for bbox, onehot in zip(
                    bboxes.numpy()[0],
                    category_ids.numpy()[0],
                ):

                    category_id = onehot.argmax()
                    category_color = CATEGORY_COLORS[
                        (category_id - 1) % len(CATEGORY_COLORS)
                    ]

                    draw.rectangle(
                        (
                            bbox[0],
                            bbox[1],
                            (bbox[0] + bbox[2]),
                            (bbox[1] + bbox[3]),
                        ),
                        outline=str(category_color),
                    )
                    draw.rectangle(
                        (bbox[0], bbox[1], (bbox[0] + 16), (bbox[1] - 10)),
                        outline=str(category_color),
                        fill=str(category_color),
                    )

                    draw.text(
                        (bbox[0] + 3, bbox[1] - 10),
                        str(onehot.argmax()),
                        "white",
                        font=ImageFont.load_default(),
                    )

            output_file = f"{output_dir}/{image_id}.png"
            output_image.save(output_file)
            print(f"Image saved to {output_file}")
        else:
            continue
