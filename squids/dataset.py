"""
A module for creating and transforming dataset to TFRecords.
"""

from __future__ import annotations

import csv
import copy
import json
import random
import datetime
from pathlib import Path
from shutil import rmtree

import PIL.Image as Image
from tqdm import tqdm

from .image import (
    Palette,
    Background,
    create_synthetic_image,
    IMAGE_CAPACITY,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)
from .shape import SHAPES_CATEGORIES

DATASET_DIR = "dataset/synthetic"
"""A default dataset directory."""

DATASET_SIZE = 1000
"""A number of generating synthetic images."""

# ------------------------------------------------------------------------------
# CSV/COCO Datasets Creators
# ------------------------------------------------------------------------------


def create_csv_dataset(
    dataset_dir: str = DATASET_DIR,
    dataset_size: int = DATASET_SIZE,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    image_palette: Palette = Palette.COLOR,
    image_background: Background = Background.WHITE,
    image_capacity: int = IMAGE_CAPACITY,
    verbose: bool = False,
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
    output_dir = Path(dataset_dir)
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True)

    images_dir = output_dir / "images"
    images_dir.mkdir()

    train_file = Path(output_dir / "instances_train.csv")
    val_file = Path(output_dir / "instances_val.csv")
    test_file = Path(output_dir / "instances_test.csv")
    categories_file = Path(output_dir / "categories.json")

    with open(categories_file, "w") as categories_fp:

        json.dump(
            {"categories": SHAPES_CATEGORIES},
            categories_fp,
            sort_keys=True,
            indent=4,
        )

    with open(train_file, "w") as train, open(val_file, "w") as val, open(
        test_file, "w"
    ) as test:

        header = "image_id,file_name,bboxes,segmentations,category_ids\n"
        train.write(header)
        val.write(header)
        test.write(header)

        if verbose:
            pbar = tqdm(range(dataset_size))

        for image_id in range(dataset_size):
            file_name = f"image{image_id}.jpg"
            image, shapes = create_synthetic_image(
                image_width,
                image_height,
                image_palette,
                image_background,
                image_capacity,
            )
            image.save(
                images_dir / file_name, "JPEG", quality=100, subsampling=0
            )

            bboxes = []
            segmentations = []
            category_ids = []
            for shape in shapes:
                bboxes.append(shape.bbox.flatten())
                segmentations.append(shape.polygon.flatten())
                category_ids.append(shape.category_id)

            record = ",".join(
                [
                    f"{image_id}",
                    f"{file_name}",
                    f'"{bboxes}"',
                    f'"{segmentations}"',
                    f'"{category_ids}"\n',
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
    dataset_dir: str = DATASET_DIR,
    dataset_size: int = DATASET_SIZE,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    image_palette: Palette = Palette.COLOR,
    image_background: Background = Background.WHITE,
    image_capacity: int = IMAGE_CAPACITY,
    verbose: bool = False,
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
    output_dir = Path(dataset_dir)
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True)

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir()
    train_images_dir = output_dir / "instances_train"
    train_images_dir.mkdir()
    val_images_dir = output_dir / "instances_val"
    val_images_dir.mkdir()
    test_images_dir = output_dir / "instances_test"
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
        "categories": SHAPES_CATEGORIES,
        "annotations": [],
    }

    train = copy.deepcopy(template)
    val = copy.deepcopy(template)
    test = copy.deepcopy(template)

    if verbose:
        pbar = tqdm(range(dataset_size))

    annotation_id = 0
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

        image_record = {
            "file_name": str(image_file),
            "coco_url": f"file:///{str(file_name)}",
            "width": int(image_width),
            "height": int(image_height),
            "date_captured": now.strftime("%Y-%m-%d %H:%M:%S"),
            "id": image_id,
        }

        annotation_records = []
        for shape in shapes:
            annotation_records.append(
                {
                    "segmentation": [
                        [int(v) for v in shape.polygon.flatten()]
                    ],
                    "area": float(shape.get_area()),
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [int(v) for v in shape.bbox.flatten()],
                    "category_id": shape.category_id,
                    "id": annotation_id,
                }
            )
            annotation_id += 1

        if partition <= 0.7:
            train["images"].append(image_record)
            train["annotations"].extend(annotation_records)
        elif partition <= 0.9:
            val["images"].append(image_record)
            val["annotations"].extend(annotation_records)
        else:
            test["images"].append(image_record)
            test["annotations"].extend(annotation_records)

        if verbose:
            pbar.update(1)

    train_file = Path(annotations_dir / "instances_train.json")
    val_file = Path(annotations_dir / "instances_val.json")
    test_file = Path(annotations_dir / "instances_test.json")

    with open(train_file, "w") as fp:
        json.dump(train, fp, indent=4)
    with open(val_file, "w") as fp:
        json.dump(val, fp, indent=4)
    with open(test_file, "w") as fp:
        json.dump(test, fp, indent=4)


# ------------------------------------------------------------------------------
# CSV/COCO Datasets Iterators
# ------------------------------------------------------------------------------


class CategoriesMap:
    def __init__(self, selected_categories: list = []):
        self.__categories_mapping = {}
        if len(selected_categories) > 0:
            for new_category_id, old_category_id in enumerate(
                sorted(selected_categories)
            ):
                self.__categories_mapping[old_category_id] = (
                    new_category_id + 1
                )

    def __getitem__(self, category_id):
        if self.__categories_mapping:
            return self.__categories_mapping[category_id]
        else:
            return category_id

    def __contains__(self, category_id):
        if self.__categories_mapping:
            return category_id in self.__categories_mapping
        else:
            return True


class DatasetIterator:
    def __init__(self, records: list, image_dir: Path):
        self.__records = records
        self.__image_dir = image_dir
        self.__size = len(self.__records)
        self.__pointer = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.__size

    def __next__(self):
        if self.__pointer >= self.__size:
            raise StopIteration

        record = self.__records[self.__pointer]
        record["image"]["data"] = Image.open(
            self.__image_dir / record["image"]["file_name"]
        )
        self.__pointer += 1

        return record


class CsvIterator(DatasetIterator):
    def __init__(self, instance_file: Path, selected_categories: list):
        categories_map = CategoriesMap(selected_categories)

        categories = dict()
        with open(instance_file.parent / "categories.json") as fp:
            for category in json.load(fp)["categories"]:
                category_id = category["id"]
                if category_id in categories_map:
                    # Remaps ald category ID to the new one.
                    new_category = copy.deepcopy(category)
                    new_category["id"] = categories_map[category["id"]]
                    categories[new_category["id"]] = new_category

        records = []
        with open(instance_file, newline="\n") as csv_fp:
            csv_reader = csv.DictReader(csv_fp, delimiter=",", quotechar='"')
            for row in csv_reader:
                annotations = []
                for bbox, segmentation, category_id in zip(
                    json.loads(row["bboxes"]),
                    json.loads(row["segmentations"]),
                    json.loads(row["category_ids"]),
                ):
                    if category_id in categories_map:
                        annotations.append(
                            {
                                "bbox": bbox,
                                "iscrowd": 0,
                                "segmentation": [segmentation],
                                "category_id": categories_map[category_id],
                            }
                        )

                # Here we discard all images which do not have any
                # annotations for the selected categories.
                if len(annotations) > 0:
                    records.append(
                        {
                            "image": {
                                "id": int(row["image_id"]),
                                "file_name": row["file_name"],
                            },
                            "annotations": annotations,
                            "categories": categories,
                        }
                    )
        super().__init__(records, instance_file.parent / "images")


class CocoIterator(DatasetIterator):
    def __init__(self, instance_file: Path, selected_categories: list):
        categories_map = CategoriesMap(selected_categories)

        with open(instance_file) as f:
            content = json.load(f)

        annotations = dict()
        for annotation in content["annotations"]:
            category_id = annotation["category_id"]
            if category_id in categories_map:
                image_id = annotation["image_id"]
                if image_id not in annotations:
                    annotations[image_id] = []

                # Remaps ald category ID to the new one.
                new_annotation = copy.deepcopy(annotation)
                new_annotation["category_id"] = categories_map[category_id]
                annotations[image_id].append(new_annotation)

        categories = dict()
        for category in content["categories"]:
            category_id = category["id"]
            if category_id in categories_map:
                # Remaps ald category ID to the new one.
                new_category = copy.deepcopy(category)
                new_category["id"] = categories_map[category_id]
                categories[new_category["id"]] = new_category

        records = []
        for image in content["images"]:
            if image["id"] in annotations:
                records.append(
                    {
                        "image": image,
                        "annotations": annotations[image["id"]],
                        "categories": categories,
                    }
                )

        super().__init__(
            records, instance_file.parent.parent / instance_file.stem
        )
