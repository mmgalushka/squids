"""
A module for creating and transforming dataset to TFRecords.
"""

from __future__ import annotations

import copy
import json
import random
import pathlib
import datetime
from shutil import rmtree

from tqdm import tqdm

from .shape import Rectangle, Triangle
from .image import Palette, Background, create_synthetic_image

DATASET_DIR = "dataset/synthetic"
"""A default dataset directory."""

DATASET_SIZE = 10000
"""A number of generating synthetic images."""

DATASET_CATEGORIES = ["rectangle", "triangle"]
"""Shapes generating in image."""


def create_csv_dataset(
    dataset_dir: str,
    dataset_size: int,
    image_width: int,
    image_height: int,
    image_palette: Palette,
    image_background: Background,
    image_capacity: int,
    verbose: bool,
):
    """Generates CSV dataset.

    Args:
        dataset_dir (str): The directory where generated data will be stored.
        dataset_size (int): The number of generated samples.
        image_width (int): The image width in pixels.
        image_height (int): The image height in pixels.
        image_palette (Palette): The palette for generating images.
        image_background (Background): The palette for generating images.
        image_capacity (int): The number of geometrical shapes per image.
        verbose (bool): The flag to set verbose mode.
    """
    output_dir = pathlib.Path(dataset_dir)
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True)

    images_dir = output_dir / "images"
    images_dir.mkdir()

    train_file = pathlib.Path(output_dir / "instances_train.csv")
    val_file = pathlib.Path(output_dir / "instances_val.csv")
    test_file = pathlib.Path(output_dir / "instances_test.csv")
    categories_file = pathlib.Path(output_dir / "categories.json")

    with open(categories_file, "w") as categories_fp:
        json.dump(
            {
                "categories": [
                    {
                        "supercategory": "geometry",
                        "id": 0,
                        "name": "rectangle",
                    },
                    {"supercategory": "geometry", "id": 1, "name": "triangle"},
                ]
            },
            categories_fp,
        )

    with open(train_file, "w") as train, open(val_file, "w") as val, open(
        test_file, "w"
    ) as test:

        header = "image_id,file_name,bboxes,segments,categories\n"
        train.write(header)
        val.write(header)
        test.write(header)

        if verbose:
            pbar = tqdm(range(dataset_size))

        for image_id in range(dataset_size):
            image_file = f"image{image_id}.jpg"
            image, shapes = create_synthetic_image(
                image_width,
                image_height,
                image_palette,
                image_background,
                image_capacity,
            )
            path = images_dir / image_file
            image.save(path, "JPEG", quality=100, subsampling=0)

            bboxes = []
            segments = []
            categories = []
            for shape in shapes:
                bboxes.append(shape.bbox.flatten())
                segments.append(shape.polygon.flatten())
                categories.append(shape.category_id)

            record = ",".join(
                [
                    f"{image_id}",
                    f"images/{image_file}",
                    f'"{bboxes}"',
                    f'"{segments}"',
                    f'"{categories}"\n',
                ]
            )

            partition = random.uniform(0, 1)
            if partition <= 0.7:
                train.write(record)
            elif partition <= 0.9:
                val.write(record)
            else:
                test.write(record)

            if verbose:
                pbar.update(1)


def create_coco_dataset(
    dataset_dir: str,
    dataset_size: int,
    image_width: int,
    image_height: int,
    image_palette: Palette,
    image_background: Background,
    image_capacity: int,
    verbose: bool,
):
    """Generates COCO dataset.

    Args:
        dataset_dir (str): The directory where generated data will be stored.
        dataset_size (int): The number of generated samples.
        image_width (int): The image width in pixels.
        image_height (int): The image height in pixels.
        image_palette (Palette): The palette for generating images.
        image_background (Background): The palette for generating images.
        image_capacity (int): The number of geometrical shapes per image.
        verbose (bool): The flag to set verbose mode.
    """
    output_dir = pathlib.Path(dataset_dir)
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True)

    annotations_images_dir = output_dir / "annotations"
    annotations_images_dir.mkdir()
    train_images_dir = output_dir / "train"
    train_images_dir.mkdir()
    val_images_dir = output_dir / "val"
    val_images_dir.mkdir()
    test_images_dir = output_dir / "test"
    test_images_dir.mkdir()

    now = datetime.datetime.now()
    template = {
        "info": {
            "description": f"COCO {now.year} Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": int(now.year),
            "contributor": "COCO Consortium",
            "date_created": now.strftime("%Y-%m-%d"),
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 0,
                "name": "Attribution-NonCommercial-ShareAlike License",
            }
        ],
        "images": [],
        "categories": [
            {"supercategory": "geometry", "id": 0, "name": "rectangle"},
            {"supercategory": "geometry", "id": 1, "name": "triangle"},
        ],
        "annotations": [],
    }

    train = copy.deepcopy(template)
    val = copy.deepcopy(template)
    test = copy.deepcopy(template)

    if verbose:
        pbar = tqdm(range(dataset_size))

    annotation_id = 0
    categories = [Rectangle, Triangle]
    for image_id in range(dataset_size):
        image_file = f"image{image_id}.jpg"
        image, shapes = create_synthetic_image(
            image_width,
            image_height,
            image_palette,
            image_background,
            image_capacity,
        )

        partition = random.uniform(0, 1)
        if partition <= 0.7:
            file_name = train_images_dir / image_file
        elif partition <= 0.9:
            file_name = val_images_dir / image_file
        else:
            file_name = test_images_dir / image_file
        image.save(file_name, "JPEG", quality=100, subsampling=0)

        images_records = []
        annotations_records = []
        for shape in shapes:
            images_records.append(
                {
                    "file_name": str(file_name),
                    "coco_url": f"file:///{str(file_name)}",
                    "width": int(image_width),
                    "height": int(image_height),
                    "date_captured": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "id": image_id,
                }
            )
            annotations_records.append(
                {
                    "segmentation": [
                        [int(v) for v in shape.polygon.flatten()]
                    ],
                    "area": float(shape.get_area()),
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [int(v) for v in shape.bbox.flatten()],
                    "category_id": categories.index(type(shape)),
                    "id": annotation_id,
                }
            )
            annotation_id += 1

        if partition <= 0.7:
            train["images"].extend(images_records)
            train["annotations"].extend(annotations_records)
        elif partition <= 0.9:
            val["images"].extend(images_records)
            val["annotations"].extend(annotations_records)
        else:
            test["images"].extend(images_records)
            test["annotations"].extend(annotations_records)

        if verbose:
            pbar.update(1)

    train_file = pathlib.Path(annotations_images_dir / "instances_train.json")
    val_file = pathlib.Path(annotations_images_dir / "instances_val.json")
    test_file = pathlib.Path(annotations_images_dir / "instances_test.json")

    with open(train_file, "w") as fp:
        json.dump(train, fp, indent=4)
    with open(val_file, "w") as fp:
        json.dump(val, fp, indent=4)
    with open(test_file, "w") as fp:
        json.dump(test, fp, indent=4)
