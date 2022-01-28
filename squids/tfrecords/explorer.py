"""
A module for converting a data set to TFRecords.
"""
from __future__ import annotations

import os
import glob
import cmd
from pathlib import Path

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import tensorflow as tf
import numpy as np

from tabulate import tabulate

from .feature import features_to_items, FEATURE_KEYS_MAP
from .errors import (
    TFRecordsDirNotFoundError,
    TFRecordIdentifierNotFoundError,
    OutputDirNotFoundError,
)

MASK_HIGHLIGHTING_COLORS = [
    "#e6194b",  # red
    "#f58231",  # orange
    "#ffe119",  # yellow
    "#bfef45",  # lime
    "#3cb44b",  # green
    "#42d4f4",  # cyan
    "#4363d8",  # blue
    "#911eb4",  # purple
    "#f032e6",  # magenta
    "#a9a9a9",  # grey
]
"""A well-separated colors range for creating bounding boxes and masks."""


def get_tfrecords_dataset(tfrecords_path: Path):
    def record_parser(proto):
        parsed_features = tf.io.parse_single_example(proto, FEATURE_KEYS_MAP)
        (
            image_id,
            image,
            bboxes,
            segmentations,
            category_ids,
        ) = features_to_items(parsed_features)

        return image_id, image, bboxes, segmentations, category_ids

    tfrecord_files = glob.glob(str(tfrecords_path / "part-*.tfrecord"))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(record_parser, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(1)

    return dataset


def explore_tfrecords(
    tfrecords_dir: str,
    image_id: int = None,
    output_dir: str = ".",
    with_summary: bool = True,
    with_bboxes: bool = True,
    with_segmentations: bool = True,
    return_artifacts: bool = False,
):
    """Explores TFRecords.

    This function allows listing a summary of all TFRecords as well as
    viewing the content of individual records such as an image together
    with annotated bounding boxes, segments, and categories.

    Args:
        tfrecords_dir (str):
            The directory containing TFRecords parts.
        image_id (int):
            The image ID to view.
        output_dir (str):
            The output directory where to store the produced artifacts such as
            image content with superimposed binding boxes, masks, categories,
            etc.
        with_bboxes (bool):
            The flag to superimpose bounding boxes on an image, defined within
            a record. If the flag is `True` bounding boxes are superimposed
            to an image and `False` boxes are discarded.
        with_segmentations (bool):
            The flag to superimpose segmentation masks on an image, defined
            within a record. If the flag is `True` segmentation masks are
            superimposed to an image and `False` masks are discarded.
        return_artifacts (bool):
            The flag (`True` value) enforces the function to return generated
            artifacts instead of output them to a console or store them in a
            file. This flag would be useful if you for example working with
            Jupyter notebook and would like to present inspected results
            directly within a cell, or planning to perform its further
            processing.

    Returns:
        record_summaries (list):
            The list with summary information about each record
            (Returned if `image_id==None & return_artifacts==True` ).
        record_summary (dict):
            The list with summary information for the specified record
            (Returned if `image_id!=None & return_artifacts==True`).
        record_image (Image)
            The image, which is stored within the specified record with
            superimposed binding boxes, masks, and categories.
            Returned if `image_id!=None & return_artifacts==True`

    Raises:
        TFRecordsDirNotFoundError:
            If input TFRecords directory directory has not been found.
        OutputDirNotFoundError:
            If input output directory directory has not been found.
        TFRecordIdentifierNotFoundError:
            If the record with the specified image ID has not been found.
    """
    if image_id is None:
        record_summaries = list_tfrecords(Path(tfrecords_dir))
        if return_artifacts:
            return record_summaries

        cli = cmd.Cmd()
        print(f"\n{tfrecords_dir}")
        if len(record_summaries) > 0:
            cli.columnize(record_summaries)
            print(f"Total {len(record_summaries)} records")
        else:
            cli.columnize(os.listdir(tfrecords_dir))
            print("No tfrecords has found")
    else:
        record_summary, record_image = view_tfrecord(
            Path(tfrecords_dir),
            image_id,
            Path(output_dir),
            with_bboxes,
            with_segmentations,
        )

        if return_artifacts:
            return record_summary, record_image

        print(
            tabulate(
                [[k, v] for k, v in record_summary.items()],
                headers=["Property", "Value"],
            )
        )
        image_file = f"{output_dir}/{image_id}.png"
        record_image.save(image_file)
        print(f"Image saved to {image_file}")


def list_tfrecords(tfrecords_dir: Path):
    """List multiple TFRecords.

    Args:
        tfrecords_dir (Path):
            The directory containing TFRecords parts.

    Returns:
        record_summaries (list):
            The list with summary information about each record.

    Raises:
        TFRecordsDirNotFoundError:
            If input TFRecords directory directory has not been found.
    """
    if not tfrecords_dir.is_dir():
        raise TFRecordsDirNotFoundError(tfrecords_dir)

    record_summaries = []
    tfrecord_files = glob.glob(str(tfrecords_dir / "part-*.tfrecord"))
    if len(tfrecord_files) > 0:
        batch = get_tfrecords_dataset(tfrecords_dir)
        for image_id, _, _, _, category_onehots in batch:
            category_ids = [
                np.argmax(category_onehot)
                for category_onehot in category_onehots.numpy()[0]
            ]
            record_summaries.append(
                str(image_id.numpy()[0])
                + " ("
                + ",".join(map(str, set(category_ids)))
                + ")"
            )
    return record_summaries


def view_tfrecord(
    tfrecords_dir: Path,
    image_id: int,
    output_dir: Path,
    with_bboxes: bool,
    with_segmentations: bool,
):
    """Views  an individual TFRecord.

    Args:
        tfrecords_dir (Path):
            The directory containing TFRecords parts.
        image_id (int):
            The image ID to view.
        output_dir (Path):
            The output directory where to store the produced artifacts such as
            image content with superimposed binding boxes, masks, categories,
            etc.
        with_bboxes (bool):
            The flag to superimpose bounding boxes on an image.
        with_segmentations (bool):
            The flag to superimpose segmentation masks on an image.

    Returns:
        record_summary (dict):
            The list with summary information for the specified record.
        record_image (Image):
            The image, which is stored within the specified record with
            superimposed binding boxes, masks, and categories.

    Raises:
        TFRecordsDirNotFoundError:
            If input TFRecords directory directory has not been found.
        OutputDirNotFoundError:
            If input output directory directory has not been found.
        TFRecordIdentifierNotFoundError:
            If the record with the specified image ID has not been found.
    """
    if not tfrecords_dir.is_dir():
        raise TFRecordsDirNotFoundError(tfrecords_dir)
    if not output_dir.is_dir():
        raise OutputDirNotFoundError(output_dir)

    batch = get_tfrecords_dataset(tfrecords_dir)
    for record_id, image, bboxes, segmentations, category_ids in batch:
        if record_id == image_id:
            record_image = Image.fromarray(
                tf.squeeze(image * 255, [0]).numpy().astype("uint8"), "RGB"
            )
            record_image_size = record_image.size
            record_image_shape = (
                record_image_size[0],
                record_image_size[1],
                3,
            )

            total_labeled_objects = len(category_ids.numpy()[0])
            available_categories_set = set(
                [
                    np.argmax(category_onehot)
                    for category_onehot in category_ids.numpy()[0]
                ]
            )

            record_summary = {
                "image_id": image_id,
                "image_shape": record_image_shape,
                "number_labeled_objects": total_labeled_objects,
                "available_category_ids": available_categories_set,
            }

            if with_segmentations:
                for mask, onehot in zip(
                    segmentations.numpy()[0],
                    category_ids.numpy()[0],
                ):

                    category_id = onehot.argmax()
                    category_color = MASK_HIGHLIGHTING_COLORS[
                        (category_id - 1) % len(MASK_HIGHLIGHTING_COLORS)
                    ]

                    blend_image = Image.new(
                        "RGBA", record_image_size, str(category_color)
                    )
                    mask_image = Image.fromarray(
                        (mask.reshape(record_image_shape)).astype("uint8"),
                        "RGB",
                    ).convert("L")
                    record_image = Image.blend(
                        record_image,
                        Image.composite(blend_image, record_image, mask_image),
                        0.7,
                    )

            if with_bboxes:
                draw = ImageDraw.Draw(record_image)
                for bbox, onehot in zip(
                    bboxes.numpy()[0],
                    category_ids.numpy()[0],
                ):

                    category_id = onehot.argmax()
                    category_color = MASK_HIGHLIGHTING_COLORS[
                        (category_id - 1) % len(MASK_HIGHLIGHTING_COLORS)
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

            return record_summary, record_image
    raise TFRecordIdentifierNotFoundError(image_id, tfrecords_dir)
