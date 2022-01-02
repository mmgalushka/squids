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
from tqdm import tqdm

from .dataset import CsvIterator, CocoIterator
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
        img = item["image"]["content"]
        bboxes = [anno["bbox"] for anno in item["annotations"]]
        segmentations = [
            anno["segmentation"][0] for anno in item["annotations"]
        ]
        category_ids = [anno["category_id"] for anno in item["annotations"]]

        feature = item_to_feature(
            image_id,
            img,
            image_width,
            image_height,
            bboxes,
            segmentations,
            category_ids,
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
    return set(os.listdir(input_dir)) == set(
        ["annotations", "train", "test", "val"]
    )


def create_tfrecords(
    dataset_dir: str,
    selected_categories: list = [],
    tfrecords_dir: str = None,
    tfrecords_size: int = 256,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    verbose: bool = False,
):
    # Gets input directory, containing dataset files that need to be
    # transformed to TFRecords.
    input_dir = Path(dataset_dir)
    # if not input_dir.exists():
    #     raise FileExistsError(f"Input directory not found at: {input_dir}")

    # Creates the output directory, where TFRecords should be stored.
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
                image_width,
                image_height,
                verbose,
            )
    elif is_coco_input(input_dir):
        for instance_file in (input_dir / "annotations").rglob("*.json"):
            items_to_tfrecords(
                output_dir,
                instance_file,
                CocoIterator(instance_file, selected_categories),
                tfrecords_size,
                image_width,
                image_height,
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
    batch = get_tfrecords_explorer(Path(tfrecords_dir))
    image_ids = []
    for image_id, _, _, _, _ in batch:
        image_ids.append(str(image_id.numpy()[0]))
    cli = cmd.Cmd()
    cli.columnize(image_ids)


def view_tfrecords(
    tfrecords_dir: str,
    image_id: int,
    with_bboxes: bool,
    with_segmentations: bool,
):
    batch = get_tfrecords_explorer(Path(tfrecords_dir))
    for selected_image_id, image, bboxes, segmentations, category_ids in batch:
        if image_id == selected_image_id:
            output_image = Image.fromarray(
                tf.squeeze(image * 255, [0]).numpy().astype("uint8"), "RGB"
            )

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
                        "RGBA", (64, 64), str(category_color)
                    )
                    mask_image = Image.fromarray(
                        (mask.reshape(64, 64, 3)).astype("uint8"), "RGB"
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

            output_image.save("my.png")
        else:
            continue
