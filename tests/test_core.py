"""
Test functions for processing datasets: generate, transform and explore;
"""

import os
import re
import argparse
import tempfile
from pathlib import Path

import pytest

from squids.dataset.maker import create_dataset
from squids.tfrecords.maker import create_tfrecords, CategoriesMap
from squids.tfrecords.explorer import explore_tfrecords, explore_tfrecord

from squids.actions import generate, transform, explore

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


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
            "instances_train",
            "instances_val",
            "instances_test",
        ]
    )
    assert set(os.listdir(tmp_dataset_dir / "annotations")) == set(
        [
            "instances_train.json",
            "instances_val.json",
            "instances_test.json",
        ]
    )
    assert len(os.listdir(tmp_dataset_dir / "instances_train")) > 0
    assert len(os.listdir(tmp_dataset_dir / "instances_val")) > 0
    assert len(os.listdir(tmp_dataset_dir / "instances_test")) > 0


def validate_coco_transformer(tmp_tfrecords_dir):
    assert set(os.listdir(tmp_tfrecords_dir)) == set(
        ["instances_train", "instances_test", "instances_val"]
    )
    assert len(os.listdir(tmp_tfrecords_dir / "instances_train")) > 0
    assert len(os.listdir(tmp_tfrecords_dir / "instances_val")) > 0
    assert len(os.listdir(tmp_tfrecords_dir / "instances_test")) > 0


def validate_tfrecords(stdout, kind):
    assert (
        len(
            re.findall(
                f".*instances_{kind}\\s\\(\\d+\\sparts\\)",
                stdout,
            )
        )
        == 1
    )
    assert (
        len(
            re.findall(
                "\\d+\\s\\((1|2|1,2)\\)",
                stdout,
            )
        )
        > 1
    )
    assert (
        len(
            re.findall(
                "Total\\s\\d+\\srecords",
                stdout,
            )
        )
        == 1
    )


def validate_tfrecord(stdout, image_id, image_output_dir):
    assert re.search("Property\\s+Value", stdout).group()
    assert re.search(f"Image ID\\s+{image_id}", stdout).group()
    assert re.search("Image Shape\\s+\\(\\d+, \\d+, \\d+\\)", stdout).group()
    assert re.search("Total Labeled Objects\\s+\\d+", stdout).group()
    assert re.search(
        "Available Categories Set\\s+\\{[\\d+,?]+\\}", stdout
    ).group()
    assert re.search(
        f"Image saved to {str(image_output_dir)}/{image_id}.png", stdout
    ).group()
    assert Path(image_output_dir / f"{image_id}.png").exists()


# ------------------------------------------------------------------------------
# Core Tests
# ------------------------------------------------------------------------------


def test_categories_map():
    categories_map = CategoriesMap()
    assert categories_map[1] == 1
    assert categories_map[2] == 2
    assert 2 in categories_map

    categories_map = CategoriesMap([3, 4])
    assert categories_map[3] == 1
    assert categories_map[4] == 2
    assert 4 in categories_map
    assert 5 not in categories_map


def test_csv_generator_transformer_functions(capsys):
    """Tests the CSV data generate/transform functions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset.
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        create_dataset(tmp_dataset_dir, verbose=True)

        validate_csv_generator(tmp_dataset_dir)

        # Transforms CSV dataset to the TFRecords.
        create_tfrecords(tmp_dataset_dir, verbose=True)
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")

        validate_csv_transformer(tmp_tfrecords_dir)

        # Explores TFRecords.
        image_id = None
        for kind in ["train", "val", "test"]:
            explore_tfrecords(
                Path(tmp_dir + f"/synthetic-tfrecords/instances_{kind}")
            )
            stdout, _ = capsys.readouterr()
            validate_tfrecords(stdout, kind)

            if kind == "train":
                match = re.findall(
                    "\\d+\\s\\([1|2|1,2]\\)",
                    stdout,
                )
                image_id = int(match[0].split(" ")[0])

        assert image_id is not None

        # Explores single TFRecord.
        explore_tfrecord(
            Path(tmp_dir + "/synthetic-tfrecords/instances_train"),
            image_id,
            tmp_tfrecords_dir,
        )
        stdout, _ = capsys.readouterr()
        validate_tfrecord(stdout, image_id, tmp_tfrecords_dir)

    # Tests an action for removing the "/synthetic" directory.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        tmp_dataset_dir.mkdir()
        create_dataset(tmp_dataset_dir)

    # Tests a reaction for missing directory
    with pytest.raises(FileNotFoundError):
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        create_tfrecords(tmp_dataset_dir)


def test_coco_generator_transformer_functions(capsys):
    """Tests the COCO data generate/transform actions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks COCO dataset.
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        create_dataset(tmp_dataset_dir, coco=True, verbose=True)

        validate_coco_generator(tmp_dataset_dir)

        # Transforms COCO dataset to the TFRecords.
        create_tfrecords(tmp_dataset_dir, verbose=True)
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")

        validate_coco_transformer(tmp_tfrecords_dir)

        # Explores TFRecords.
        for kind in ["train", "val", "test"]:
            explore_tfrecords(
                Path(tmp_dir + f"/synthetic-tfrecords/instances_{kind}")
            )
            stdout, _ = capsys.readouterr()
            validate_tfrecords(stdout, kind)

    # This tests the action for removing the "/synthetic" directory.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        tmp_dataset_dir.mkdir()
        create_dataset(tmp_dataset_dir, coco=True)

    # Tests a reaction for missing directory.
    with pytest.raises(FileNotFoundError):
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        create_tfrecords(tmp_dataset_dir)


def test_CSV_generator_transformer_actions(capsys):
    """Tests the CSV data generate/transform actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)
    explore(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset.
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        args = parser.parse_args(["generate", str(tmp_dataset_dir)])
        args.func(args)

        validate_csv_generator(tmp_dataset_dir)

        # Transforms CSV dataset to the TFRecords.
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")
        args = parser.parse_args(
            [
                "transform",
                str(tmp_dataset_dir),
                str(tmp_tfrecords_dir),
            ]
        )
        args.func(args)

        validate_csv_transformer(tmp_tfrecords_dir)

        # Explores TFRecords.
        for kind in ["train", "val", "test"]:
            args = parser.parse_args(
                [
                    "explore",
                    str(tmp_tfrecords_dir / f"instances_{kind}"),
                ]
            )

            args.func(args)

            stdout, _ = capsys.readouterr()
            validate_tfrecords(stdout, kind)


def test_coco_generator_transformer_actions(capsys):
    """Tests the COCO data generate/transform actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)
    explore(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks COCO dataset.
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        args = parser.parse_args(["generate", str(tmp_dataset_dir), "--coco"])
        args.func(args)

        validate_coco_generator(tmp_dataset_dir)

        # Transforms COCO dataset to the TFRecords.
        tmp_tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")
        args = parser.parse_args(
            [
                "transform",
                str(tmp_dataset_dir),
                str(tmp_tfrecords_dir),
            ]
        )
        args.func(args)

        validate_coco_transformer(tmp_tfrecords_dir)

        # Explores TFRecords.
        for kind in ["train", "val", "test"]:
            args = parser.parse_args(
                [
                    "explore",
                    str(tmp_tfrecords_dir / f"instances_{kind}"),
                ]
            )

            args.func(args)

            stdout, _ = capsys.readouterr()
            validate_tfrecords(stdout, kind)


def test_unknown_transformation():
    """Tests transform of an unknown dataset type."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset
        tmp_dataset_dir = Path(tmp_dir + "/synthetic")
        tmp_dataset_dir.mkdir()

        with pytest.raises(ValueError):
            tmp_dataset_dir = Path(tmp_dir + "/synthetic")
            # The tmp_dataset_dir does not contain either CSV or COCO data.
            create_tfrecords(tmp_dataset_dir)
