"""A module for exploring TFRecords data set."""

import os
import glob
import cmd
from pathlib import Path
from collections import Counter

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import tensorflow as tf
import numpy as np

from tabulate import tabulate

from .feature import features_to_items, FEATURE_KEYS_MAP
from .errors import (
    DirNotFoundError,
    IdentifierNotFoundError,
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
    with_categories: bool = False,
    with_bboxes: bool = False,
    with_segmentations: bool = False,
    return_artifacts: bool = False,
):
    """Explores individual or multiple TFRecord(s).

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
            image content with overlaid binding boxes, masks, categories,
            etc.
        with_categories (bool):
            The flag to superimpose categories on an image, defined within
            a record. If the flag is `True` categories are overlaid
            to an image and `False` categories are discarded.
        with_bboxes (bool):
            The flag to superimpose bounding boxes on an image, defined within
            a record. If the flag is `True` bounding boxes are overlaid
            to an image and `False` boxes are discarded.
        with_segmentations (bool):
            The flag to superimpose segmentation masks on an image, defined
            within a record. If the flag is `True` segmentation masks are
            overlaid to an image and `False` masks are discarded.
        return_artifacts (bool):
            The flag (`True` value) enforces the function to return generated
            artifacts instead of output them to a console or store them in a
            file. This flag would be useful if you for example working with
            Jupyter notebook and would like to present inspected results
            directly within a cell, or planning to perform its further
            processing.

    Returns:
        (record_ids, record_summaries) | (record_image, record_summary):
        * These values are returned if `image_id==None & return_artifacts
            ==True`, where:
            - `record_ids` - a list<int> of records identifiers;
            - `record_summaries` - a list<Counter> of summary information
            about each record;
        * These values are returned if `image_id!=None &  return_artifacts
            ==True`, where:
            - `record_image` - a PIL image with overlays of binding boxes,
            masks, and categories;
            - `record_summary`- a dictionary with summary information about
            the selected record;

    Raises:
        TFRecordsDirNotFoundError:
            If input TFRecords directory directory has not been found.
        OutputDirNotFoundError:
            If input output directory directory has not been found.
        TFRecordIdentifierNotFoundError:
            If the record with the specified image ID has not been found.
    """
    if image_id is None:
        record_ids, record_summaries = list_tfrecords(Path(tfrecords_dir))
        if return_artifacts:
            return record_ids, record_summaries

        cli = cmd.Cmd()
        print(f"\n{tfrecords_dir}")
        if len(record_summaries) > 0:
            record_info = [
                f"{str(record_id)} {set(record_summary)}"
                for record_id, record_summary in zip(
                    record_ids, record_summaries
                )
            ]
            cli.columnize(record_info)
            print(f"Total {len(record_info)} records")
        else:
            cli.columnize(os.listdir(tfrecords_dir))
            print("No tfrecords has found")
    else:
        record_image, record_summary = view_tfrecord(
            Path(tfrecords_dir),
            image_id,
            Path(output_dir),
            with_categories,
            with_bboxes,
            with_segmentations,
        )

        if return_artifacts:
            return record_image, record_summary

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
        record_ids (list):
            The list<int> of record identifiers (same as image identifiers).
        record_summaries (list):
            The list<Counter> of record summaries.

    Raises:
        TFRecordsDirNotFoundError:
            If input TFRecords directory directory has not been found.
    """
    if not tfrecords_dir.is_dir():
        raise DirNotFoundError("exploring TFRecords", tfrecords_dir)

    record_ids = []
    record_summaries = []
    tfrecord_files = glob.glob(str(tfrecords_dir / "part-*.tfrecord"))
    if len(tfrecord_files) > 0:
        batch = get_tfrecords_dataset(tfrecords_dir)
        for record_id, _, _, _, onehots in batch:
            record_ids.append(record_id.numpy()[0])

            category_ids = Counter(
                [np.argmax(onehot) for onehot in onehots.numpy()[0]]
            )

            record_summaries.append(category_ids)
    return record_ids, record_summaries


def view_tfrecord(
    tfrecords_dir: Path,
    image_id: int,
    output_dir: Path,
    with_categories: bool,
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
            image content with overlaid binding boxes, masks, categories,
            etc.
        with_categories (bool):
            The flag to superimpose categories on an image.
        with_bboxes (bool):
            The flag to superimpose bounding boxes on an image.
        with_segmentations (bool):
            The flag to superimpose segmentation masks on an image.

    Returns:
        record_image (PIL.Image):
            The image, which is stored within the specified record with
            overlaid binding boxes, masks, and categories.
        record_summary (dict):
            The list with summary information for the specified record.

    Raises:
        TFRecordsDirNotFoundError:
            If input TFRecords directory directory has not been found.
        OutputDirNotFoundError:
            If input output directory directory has not been found.
        TFRecordIdentifierNotFoundError:
            If the record with the specified image ID has not been found.
    """
    if not tfrecords_dir.is_dir():
        raise DirNotFoundError("exploring TFRecords", tfrecords_dir)
    if not output_dir.is_dir():
        raise DirNotFoundError("output", output_dir)

    batch = get_tfrecords_dataset(tfrecords_dir)
    for record_id, image_data, bboxes, masks, onehots in batch:
        record_id = record_id.numpy()[0]

        if record_id == image_id:
            record_image = Image.fromarray(
                tf.squeeze(image_data * 255, [0]).numpy().astype("uint8"),
                "RGB",
            )
            bboxes = bboxes.numpy()[0]
            masks = masks.numpy()[0]
            onehots = onehots.numpy()[0]

            if with_segmentations:
                for mask, onehot in zip(masks, onehots):
                    category_id = onehot.argmax()
                    category_color = MASK_HIGHLIGHTING_COLORS[
                        (category_id - 1) % len(MASK_HIGHLIGHTING_COLORS)
                    ]

                    blend_image = Image.new(
                        "RGBA", record_image.size, str(category_color)
                    )

                    mask_image = Image.fromarray(
                        (mask.reshape(record_image.size) * 255.0).astype(
                            "uint8"
                        ),
                    ).convert("L")

                    record_image = Image.blend(
                        record_image,
                        Image.composite(blend_image, record_image, mask_image),
                        0.7,
                    )

            draw = ImageDraw.Draw(record_image)
            for bbox, onehot in zip(bboxes, onehots):
                category_id = onehot.argmax()
                category_color = MASK_HIGHLIGHTING_COLORS[
                    (category_id - 1) % len(MASK_HIGHLIGHTING_COLORS)
                ]

                if with_categories:
                    ax, ay, h = bbox[0], bbox[1], bbox[3]
                    if ay - 10 > 0:
                        bx, by = ax + 16, ay - 10
                        tx, ty = ax + 3, ay - 10
                    else:
                        ay += h
                        bx, by = ax + 16, ay + 10
                        tx, ty = ax + 3, ay - 1

                    draw.rectangle(
                        (ax, ay, bx, by),
                        outline=str(category_color),
                        fill=str(category_color),
                    )
                    draw.text(
                        (tx, ty),
                        str(onehot.argmax()),
                        "white",
                        font=ImageFont.load_default(),
                    )

                if with_bboxes:
                    draw.rectangle(
                        (
                            bbox[0],
                            bbox[1],
                            (bbox[0] + bbox[2]),
                            (bbox[1] + bbox[3]),
                        ),
                        outline=str(category_color),
                    )

            number_of_objects = len(onehots)
            available_categories = set(
                [np.argmax(onehot) for onehot in onehots]
            )

            record_summary = {
                "image_id": record_id,
                "image_size": record_image.size,
                "number_of_objects": number_of_objects,
                "available_categories": available_categories,
            }

            return record_image, record_summary
    raise IdentifierNotFoundError(image_id, tfrecords_dir)
