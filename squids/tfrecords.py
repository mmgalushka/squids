"""
A module for converting a data source to TFRecords.
"""
from __future__ import annotations

import os
import csv
import json
from pathlib import Path

from tqdm import tqdm
from PIL import Image

import tensorflow as tf

from .feature import (
    image_to_feature,
    bboxes_to_feature,
    segmentations_to_feature,
    category_ids_to_feature,
)


class ItemsIterator:
    def __init__(self, size):
        self._pointer = 0
        self.__size = size

    def __iter__(self):
        return self

    def __len__(self):
        return self.__size


class CsvIterator(ItemsIterator):
    def __init__(self, instance_file: Path):
        self.__instance_file = instance_file

        with open(instance_file.parent / "categories.json") as categories_fp:
            self.__categories = dict()
            for category in json.load(categories_fp)["categories"]:
                category_id = category["id"]
                if category_id not in self.__categories:
                    self.__categories[category_id] = []
                self.__categories[category_id].append(category)

        self.__annotations = []
        with open(instance_file, newline="\n") as csv_fp:
            csv_reader = csv.DictReader(csv_fp, delimiter=",", quotechar='"')
            for row in csv_reader:
                self.__annotations.append(
                    {
                        "image": {"file_name": row["file_name"]},
                        "annotations": [
                            {
                                "bbox": bbox,
                                "segmentation": [segmentation],
                                "category_id": category_id,
                            }
                            for bbox, segmentation, category_id in zip(
                                json.loads(row["bboxes"]),
                                json.loads(row["segmentations"]),
                                json.loads(row["category_ids"]),
                            )
                        ],
                    }
                )

            super().__init__(len(self.__annotations))

    def __next__(self):
        if self._pointer >= len(self):
            raise StopIteration

        item = self.__annotations[self._pointer]
        item["image"]["content"] = Image.open(
            self.__instance_file.parent / item["image"]["file_name"]
        )
        self._pointer += 1

        return item


class CocoIterator(ItemsIterator):
    def __init__(self, instance_file: Path):
        with open(instance_file) as f:
            self.content = json.load(f)

        self.__annotations = dict()
        for annotation in self.content["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.__annotations:
                self.__annotations[image_id] = []
            self.__annotations[image_id].append(annotation)

        self.__categories = dict()
        for category in self.content["categories"]:
            category_id = category["id"]
            if category_id not in self.__categories:
                self.__categories[category_id] = []
            self.__categories[category_id].append(category)

        super().__init__(len(self.content["images"]))

    def __next__(self):
        if self._pointer >= len(self):
            raise StopIteration

        image = self.content["images"][self._pointer]
        image["content"] = Image.open(image["file_name"])

        item = dict()
        item["image"] = image
        item["annotations"] = self.__annotations[image["id"]]

        self._pointer += 1

        return item


def items_to_tfrecords(
    output_dir: Path,
    instance_file: Path,
    items: ItemsIterator,
    tfrecords_size: int,
    image_width: int,
    image_height: int,
    verbose: bool,
):
    def get_example(item):
        img = item["image"]["content"]
        bboxes = [anno["bbox"] for anno in item["annotations"]]
        segmentations = [
            anno["segmentation"][0] for anno in item["annotations"]
        ]
        category_ids = [anno["category_id"] for anno in item["annotations"]]

        feature = {
            **image_to_feature(img, image_width, image_height),
            **bboxes_to_feature(bboxes),
            **segmentations_to_feature(segmentations),
            **category_ids_to_feature(category_ids),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    # Makes a directory where TFRecords files will be stored. For example
    #    output_dir -> /x/y/z
    #    instance_file   -> train.csv
    #
    # the TFRecords directory will be
    #    tfrecords_dir ->  /x/y/z/train
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
    tfrecords_dir: str = None,
    tfrecords_size: int = 256,
    image_width: int = None,
    image_height: int = None,
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
                CsvIterator(instance_file),
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
                CocoIterator(instance_file),
                tfrecords_size,
                image_width,
                image_height,
                verbose,
            )

    else:
        raise ValueError("invalid input data format.")
