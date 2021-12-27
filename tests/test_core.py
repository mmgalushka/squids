"""
Test functions for processing datasets form `squids/dataset.py` and
`squids/tfrecords.py`.
"""

import os
import argparse
import tempfile
from pathlib import Path

from squids.image import Palette, Background
from squids.dataset import create_csv_dataset, create_coco_dataset
from squids.tfrecords import create_tfrecords

from squids.actions import generate, transform


def validate_csv_generator(tmp_dataset_dir):
    assert set(os.listdir(tmp_dataset_dir)) == set(
        [
            "images",
            "instances_train.csv",
            "instances_test.csv",
            "instances_val.csv",
            "categories.json",
        ]
    )
    assert len(os.listdir(tmp_dataset_dir / "images")) == 100


def validate_csv_transformer(tmp_tfrecords_dir):
    assert set(os.listdir(tmp_tfrecords_dir)) == set(
        ["instances_train", "instances_test", "instances_val"]
    )
    assert len(os.listdir(tmp_tfrecords_dir / "instances_train")) > 0
    assert len(os.listdir(tmp_tfrecords_dir / "instances_val")) > 0
    assert len(os.listdir(tmp_tfrecords_dir / "instances_test")) > 0


def validate_coco_generator(tmp_dataset_dir):
    assert set(os.listdir(tmp_dataset_dir)) == set(
        [
            "annotations",
            "train",
            "val",
            "test",
        ]
    )
    assert set(os.listdir(tmp_dataset_dir / "annotations")) == set(
        [
            "instances_train.json",
            "instances_val.json",
            "instances_test.json",
        ]
    )
    assert len(os.listdir(tmp_dataset_dir / "train")) > 0
    assert len(os.listdir(tmp_dataset_dir / "val")) > 0
    assert len(os.listdir(tmp_dataset_dir / "test")) > 0


def validate_coco_transformer(tmp_tfrecords_dir):
    assert set(os.listdir(tmp_tfrecords_dir)) == set(
        ["instances_train", "instances_test", "instances_val"]
    )
    assert len(os.listdir(tmp_tfrecords_dir / "instances_train")) > 0
    assert len(os.listdir(tmp_tfrecords_dir / "instances_val")) > 0
    assert len(os.listdir(tmp_tfrecords_dir / "instances_test")) > 0


def test_test_csv_gen_trans_actions():
    """Tests the CSV data generate/transform actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        args = parser.parse_args(
            ["generate", "-o", str(tmp_dataset_dir), "-s", "100"]
        )
        args.func(args)

        validate_csv_generator(tmp_dataset_dir)

        # Transforms CSV dataset to the TFRecords
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")
        args = parser.parse_args(
            [
                "transform",
                "-i",
                str(tmp_dataset_dir),
                "-o",
                str(tmp_tfrecords_dir),
            ]
        )
        args.func(args)

        validate_csv_transformer(tmp_tfrecords_dir)


def test_test_coco_gen_trans_actions():
    """Tests the CSV data generate/transform actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks COCO dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        args = parser.parse_args(
            ["generate", "-o", str(tmp_dataset_dir), "-s", "100", "--coco"]
        )
        args.func(args)

        validate_coco_generator(tmp_dataset_dir)

        # Transforms COCO dataset to the TFRecords
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")
        args = parser.parse_args(
            [
                "transform",
                "-i",
                str(tmp_dataset_dir),
                "-o",
                str(tmp_tfrecords_dir),
            ]
        )
        args.func(args)

        validate_coco_transformer(tmp_tfrecords_dir)


def test_test_csv_gen_trans_functions():
    """Tests the CSV data generate/transform actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        create_csv_dataset(
            tmp_dataset_dir,
            100,
            64,
            64,
            Palette.COLOR,
            Background.WHITE,
            3,
            True,
        )

        validate_csv_generator(tmp_dataset_dir)

        # Transforms CSV dataset to the TFRecords
        create_tfrecords(tmp_dataset_dir, ["rectangle", "triangle"])
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")

        validate_csv_transformer(tmp_tfrecords_dir)


def test_test_coc_gen_trans_functions():
    """Tests the CSV data generate/transform actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks COCO dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        create_coco_dataset(
            tmp_dataset_dir,
            100,
            64,
            64,
            Palette.COLOR,
            Background.WHITE,
            3,
            True,
        )

        validate_coco_generator(tmp_dataset_dir)

        # Transforms COCO dataset to the TFRecords
        create_tfrecords(tmp_dataset_dir, ["rectangle", "triangle"])
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")

        validate_coco_transformer(tmp_tfrecords_dir)
