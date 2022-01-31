"""A module for converting a data source to TFRecords."""

from __future__ import annotations

import os
import json
import copy
import csv
from pathlib import Path

import PIL.Image as Image
import tensorflow as tf
from tqdm import tqdm

from .feature import items_to_features
from .errors import DirNotFoundError, InvalidDatasetFormat

from ..config import IMAGE_WIDTH, IMAGE_HEIGHT, DATASET_DIR, TFRECORDS_SIZE

# ------------------------------------------------------------------------------
# CSV/COCO Dataset Detectors
# ------------------------------------------------------------------------------


def is_csv_input(input_dir: Path) -> bool:
    """
    Tests if the input directory represents CSV dataset format.

    Args:
        input_dir (Path):
            The input directory to test.

    Returns:
        status (bool):
            Returns `True` if the input directory represents CSV dataset
            format and `False` otherwise.
    """
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
    """
    Tests if the input directory represents COCO dataset format.

    Args:
        input_dir (Path):
            The input directory to test.

    Returns:
        status (bool):
            Returns `True` if the input directory represents COCO dataset
            format and `False` otherwise.
    """
    root_artifacts = os.listdir(input_dir)
    if "annotations" in root_artifacts:
        annotations_artifacts = os.listdir(input_dir / "annotations")
        stems_artifacts = [
            Path(artifact).stem for artifact in annotations_artifacts
        ]
        return set(stems_artifacts).issubset(set(root_artifacts))
    return False


# ------------------------------------------------------------------------------
# CSV/COCO Dataset Iterators
# ------------------------------------------------------------------------------


class CategoriesMap:
    """
    A dictionary-like object for intelligently mapping categories.

    The goal of this class is to remap user-specified categories for the
    compact one-hot encoding.  Let's review a simple example. Assume, the
    original data has images that include objects belonging to the two
    categories: 15 and 20. If we do not remap these categories, then we
    need to create one-hot with length `max(15,20) + 1 = 21` (plus one
    is to allow one additional category "no object"). This creates
    unnecessary overhead during the model training. The most intuitive
    solution would be to remap the original categories to the following
    `{15: 1, 20: 2}`. In this case, the one-hot encoding length would be
    `max(1,2) + 1 = 3` .

    To initiate remapping the `selected_categories` argument should be
    defined. All selected category IDs will be sorted in ascending order
    with the consequent re-assignment to the new IDs. For example, let's
    assume the specified selected categories are `[12, 5, 3,  23]`, after
    sorting this list will have the following  `[3, 5, 12, 23]` and the
    remapping `{3: 1, 5: 2, 12: 3, 23: 4}`.

    If the `selected_categories` argument is defined the following
    operation `map[ORIGINAL_ID]` returns `NEW_ID` (remapped category ID).
    If the `selected_categories` argument does not  defined the following
    operation `map[ORIGINAL_ID]` returns `ORIGINAL_ID` (in other words
    its remaps ID to itself).

    Args:
        selected_categories (list):
            The list of categories to map.
    """

    def __init__(self, selected_categories: list):
        self.__categories_mapping = {}
        if len(selected_categories) > 0:
            for new_category_id, old_category_id in enumerate(
                sorted(selected_categories)
            ):
                self.__categories_mapping[old_category_id] = (
                    new_category_id + 1
                )

    def __getitem__(self, category_id):
        """Returns the remapped category ID."""
        if self.__categories_mapping:
            return self.__categories_mapping[category_id]
        else:
            return category_id

    def __contains__(self, category_id):
        """Tests if the specified category ID in the map."""
        if self.__categories_mapping:
            return category_id in self.__categories_mapping
        else:
            return True


class DatasetIterator:
    """
    The base class for dataset records iterator.

    Args:
        records (list):
            The list with records to iterate.
        image_dir (Path):
            The base path for loading images.
    """

    def __init__(self, records: list, image_dir: Path):
        self.__records = records
        self.__image_dir = image_dir
        self.__size = len(self.__records)
        self.__pointer = 0

    def __iter__(self):
        """Returns the dataset records iterator."""
        return self

    def __len__(self):
        """Returns a number of records in the dataset."""
        return self.__size

    def __next__(self):
        """Returns the next record."""
        if self.__pointer >= self.__size:
            raise StopIteration

        record = self.__records[self.__pointer]
        record["image"]["data"] = Image.open(
            self.__image_dir / record["image"]["file_name"]
        )
        self.__pointer += 1

        return record


class CsvIterator(DatasetIterator):
    """
    The CSV dataset iterator.

    Args:
        instance_file (Path):
            The path to the `csv`-file with records to iterate.
        selected_categories (list):
            The list of category IDs on which iteration should take place.
            If an image within a record does not contain a selected category
            it's skipped. If the selected category IDs do not define, then
            iterate goes via all images.
    """

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
    """
    The COCO dataset iterator.

    Args:
        instance_file (Path):
            The path to the `json`-file with records to iterate.
        selected_categories (list):
            The list of category IDs on which iteration should take place.
            If an image within a record does not contain a selected category
            it's skipped. If the selected category IDs do not define, then
            iterate goes via all images.
    """

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


# ------------------------------------------------------------------------------
# Dataset to TFRecords Transformer
# ------------------------------------------------------------------------------


def instances_to_tfrecords(
    instance_file: Path,
    output_dir: Path,
    items: DatasetIterator,
    size: int,
    image_width: int,
    image_height: int,
    verbose: bool,
):
    """
    Converse instances to tfrecords.

    Args:
        instance_file (Path):
            The path to the instance file to read data from.
        output_dir (Path):
            The path to the output directory to save generated TFRecords.
        items (DatasetIterator):
            The CSV or COCO dataset iterator.
        size (int):
            The number of images per partion.
        image_width (int):
            The TFRecords image width resize to.
        image_height (int):
            The TFRecords image height resize to.
        verbose (bool):
            The flag to set verbose mode.
    """

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

        feature = items_to_features(
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
    part_count = size

    # Initializes the progress bar of verbose mode is on.
    if verbose:
        pbar = tqdm(total=len(items))

    for item in items:
        if item:
            if part_count >= size:
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


def create_tfrecords(
    dataset_dir: str = DATASET_DIR,
    tfrecords_dir: str = None,
    size: int = TFRECORDS_SIZE,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    selected_categories: list = [],
    verbose: bool = False,
):
    """
    This function transforms CSV or COCO dataset to TFRecords.

    Args:
        dataset_dir (str):
            The path to the data set directory to transform.
        tfrecords_dir (str):
            The path to the output directory to save generated TFRecords.
        size (int):
            The number of images per partion.
        image_width (int):
            The TFRecords image width resize to.
        image_height (int):
            The TFRecords image height resize to.
        selected_categories (list):
            The list of selected category IDs.
        verbose (bool):
            The flag to set verbose mode.

    Raises:
        DirNotFoundError:
            If input or output directories do not exist.
        InvalidDatasetFormat:
            If the input dataset has invalid CSV or COCO format.
    """
    input_dir = Path(dataset_dir)
    if not input_dir.exists():
        raise DirNotFoundError("input dataset", input_dir)

    if tfrecords_dir is None:
        output_dir = input_dir.parent / (input_dir.name + "-tfrecords")
    else:
        output_dir = Path(tfrecords_dir)
        if not output_dir.parent.exists():
            raise DirNotFoundError("parent (to output)", output_dir.parent)
    output_dir.mkdir(exist_ok=True)

    if is_csv_input(input_dir):
        for instance_file in input_dir.rglob("*.csv"):
            instances_to_tfrecords(
                instance_file,
                output_dir,
                CsvIterator(instance_file, selected_categories),
                size,
                image_width,
                image_height,
                verbose,
            )
    elif is_coco_input(input_dir):
        for instance_file in (input_dir / "annotations").rglob("*.json"):
            instances_to_tfrecords(
                instance_file,
                output_dir,
                CocoIterator(instance_file, selected_categories),
                size,
                image_width,
                image_height,
                verbose,
            )
    else:
        raise InvalidDatasetFormat()
