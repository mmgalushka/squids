"""
A module for converting a data source to TFRecords.
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

from .feature import feature_to_item, KEY_FEATURE_MAP


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
"""A well-separated colors range."""


def get_tfrecords_dataset(
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


def explore_tfrecords(tfrecords_dir: str):
    cli = cmd.Cmd()
    input_path = Path(tfrecords_dir)
    if input_path.is_dir():
        tfrecord_files = glob.glob(str(input_path / "part-*.tfrecord"))
        if len(tfrecord_files) > 0:
            records = []
            batch = get_tfrecords_dataset(input_path)
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


def explore_tfrecord(
    tfrecords_dir: str,
    image_id: int,
    output_dir: str = ".",
    with_summary: bool = True,
    with_bboxes: bool = True,
    with_segmentations: bool = True,
):

    batch = get_tfrecords_dataset(Path(tfrecords_dir))
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
                    category_color = MASK_HIGHLIGHTING_COLORS[
                        (category_id - 1) % len(MASK_HIGHLIGHTING_COLORS)
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

            output_file = f"{output_dir}/{image_id}.png"
            output_image.save(output_file)
            print(f"Image saved to {output_file}")
        else:
            continue
