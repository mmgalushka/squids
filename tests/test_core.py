"""
Test functions for processing datasets form `squids/dataset.py` and
`squids/tfrecords.py`.
"""

import os
import argparse
import tempfile
from pathlib import Path

import pytest

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
    assert len(os.listdir(tmp_dataset_dir / "images")) == 1000


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


def test_test_csv_gen_tran_actions():
    """Tests the CSV data generate/transform actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        args = parser.parse_args(["generate", "-o", str(tmp_dataset_dir)])
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


def test_test_coco_gen_tran_actions():
    """Tests the COCO data generate/transform actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks COCO dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        args = parser.parse_args(
            ["generate", "-o", str(tmp_dataset_dir), "--coco"]
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


def test_test_csv_gen_tran_functions():
    """Tests the CSV data generate/transform actions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        create_csv_dataset(tmp_dataset_dir, verbose=True)

        validate_csv_generator(tmp_dataset_dir)

        # Transforms CSV dataset to the TFRecords
        create_tfrecords(tmp_dataset_dir, verbose=True)
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")

        validate_csv_transformer(tmp_tfrecords_dir)

    # Tests an action for removing the "/synthetic" directory.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        tmp_dataset_dir.mkdir()
        create_csv_dataset(tmp_dataset_dir)

    # Tests a reaction fon missing directory
    with pytest.raises(FileNotFoundError):
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        create_tfrecords(tmp_dataset_dir)


def test_test_coco_gen_tran_functions():
    """Tests the COCO data generate/transform actions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks COCO dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        create_coco_dataset(tmp_dataset_dir, verbose=True)

        validate_coco_generator(tmp_dataset_dir)

        # Transforms COCO dataset to the TFRecords
        create_tfrecords(tmp_dataset_dir, verbose=True)
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")

        validate_coco_transformer(tmp_tfrecords_dir)

    # This tests the action for removing the "/synthetic" directory.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        tmp_dataset_dir.mkdir()
        create_coco_dataset(tmp_dataset_dir)


def test_test_unknown_tran():
    """Tests transform of an unknown dataset type."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        tmp_dataset_dir.mkdir()

        with pytest.raises(ValueError):
            tmp_dataset_dir = Path(tmp_dir + "/synthetic")
            # The tmp_dataset_dir does not contain either CSV or COCO data.
            create_tfrecords(tmp_dataset_dir)
