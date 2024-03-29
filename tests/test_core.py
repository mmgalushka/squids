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
from squids.tfrecords.maker import (
    create_tfrecords,
    CategoriesMap,
    InvalidDatasetFormat,
)
from squids.tfrecords.loader import load_tfrecords
from squids.tfrecords.explorer import explore_tfrecords
from squids.actions import generate, transform, explore
from squids.tfrecords.errors import DirNotFoundError, IdentifierNotFoundError
from squids.config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS,
    BATCH_SIZE,
    NUM_DETECTING_OBJECTS,
)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def validate_csv_generator(dataset_dir):
    assert set(os.listdir(dataset_dir)) == set(
        [
            "images",
            "instances_train.csv",
            "instances_test.csv",
            "instances_val.csv",
            "categories.json",
        ]
    )
    assert len(os.listdir(dataset_dir / "images")) == 1000


def validate_coco_generator(dataset_dir):
    assert set(os.listdir(dataset_dir)) == set(
        [
            "annotations",
            "instances_train",
            "instances_val",
            "instances_test",
        ]
    )
    assert set(os.listdir(dataset_dir / "annotations")) == set(
        [
            "instances_train.json",
            "instances_val.json",
            "instances_test.json",
        ]
    )
    assert len(os.listdir(dataset_dir / "instances_train")) > 0
    assert len(os.listdir(dataset_dir / "instances_val")) > 0
    assert len(os.listdir(dataset_dir / "instances_test")) > 0


def validate_transformer(tfrecords_dir):
    assert set(os.listdir(tfrecords_dir)) == set(
        ["instances_train", "instances_test", "instances_val"]
    )
    assert len(os.listdir(tfrecords_dir / "instances_train")) > 0
    assert len(os.listdir(tfrecords_dir / "instances_val")) > 0
    assert len(os.listdir(tfrecords_dir / "instances_test")) > 0


def validate_tfrecords_stdout(stdout, kind):
    assert re.search(
        f".*instances_{kind}",
        stdout,
    )

    assert (
        len(
            re.findall(
                "\\d+\\s\\{(1|2|3|1, 2|1, 3|2, 3|1, 2, 3)\\}",
                stdout,
            )
        )
        > 1
    )

    assert re.search(
        "Total\\s\\d+\\srecords",
        stdout,
    )


def validate_tfrecords_artifacts(record_ids, record_summaries):
    assert type(record_ids) == list
    assert type(record_summaries) == list

    assert len(record_ids) > 0
    assert len(record_summaries) > 0
    assert len(record_ids) == len(record_summaries)

    for record_id, record_summary in zip(record_ids, record_summaries):
        assert record_id >= 0
        assert len(record_summaries) > 0
        assert re.search(
            "\\{(1|2|3|1, 2|1, 3|2, 3|1, 2, 3)\\}",
            str(set(record_summary)),
        )


def validate_no_tfrecords(stdout):
    assert re.search(
        "No\\stfrecords\\shas\\sfound",
        stdout,
    )


def validate_tfrecord_stdout(stdout, image_id, image_output_dir):
    # Tests a record summary.
    assert re.search("Property\\s+Value", stdout)
    assert re.search(f"image_id\\s+{image_id}", stdout)
    assert re.search("image_size\\s+\\(\\d+, \\d+\\)", stdout)
    assert re.search("number_of_objects\\s+\\d+", stdout)
    assert re.search("available_categories\\s+\\{[\\d+(,\\s)?]+\\}", stdout)
    assert re.search(
        f"Image saved to {str(image_output_dir)}/{image_id}.png", stdout
    )

    # Tests a record image.
    assert Path(image_output_dir / f"{image_id}.png").exists()


def validate_tfrecord_artifacts(record_summary, record_image, image_id):
    # Tests a record summary.
    assert type(record_summary) == dict
    assert record_summary["image_id"] == image_id
    assert re.search("\\d+", str(record_summary["image_id"]))
    assert re.search("\\(\\d+, \\d+\\)", str(record_summary["image_size"]))
    assert re.search("\\d+", str(record_summary["number_of_objects"]))
    assert re.search(
        "\\{(\\d+(,\\s)?)+\\}", str(record_summary["available_categories"])
    )

    # Tests a record image.
    assert record_image is not None
    assert record_image.size == (64, 64)


# ------------------------------------------------------------------------------
# GTE Function Tests
# ------------------------------------------------------------------------------


def core_function_testscript(coco):
    """Tests generate/transform/explore functions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # -----------------------------
        # Generates and checks dataset.
        # -----------------------------
        dataset_dir = Path(tmp_dir + "/synthetic")
        dataset_dir.mkdir()  # this tests code which deletes old dataset;
        create_dataset(dataset_dir, coco=coco, random_state=42)

        if coco:
            validate_coco_generator(dataset_dir)
        else:
            validate_csv_generator(dataset_dir)

        # ------------------------------------
        # Transforms dataset to the TFRecords.
        # ------------------------------------
        tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")
        tfrecords_dir.mkdir()  # this tests code which deletes old TFRecords;
        create_tfrecords(dataset_dir)

        validate_transformer(tfrecords_dir)

        # -------------------
        # Explores TFRecords.
        # -------------------
        image_id = None
        for kind in ["train", "val", "test"]:
            record_ids, record_summaries = explore_tfrecords(
                Path(tmp_dir + f"/synthetic-tfrecords/instances_{kind}"),
                return_artifacts=True,
            )

            validate_tfrecords_artifacts(record_ids, record_summaries)

            # Grabs image ID for exploring individual records.
            if kind == "train":
                image_id = record_ids[0]

        with pytest.raises(DirNotFoundError):
            record_summaries = explore_tfrecords(
                Path(tmp_dir + "/synthetic-tfrecords/instances_xxx"),
                return_artifacts=True,
            )

        # -------------------------
        # Explores single TFRecord.
        # -------------------------
        assert image_id is not None
        record_image, record_summary = explore_tfrecords(
            Path(tmp_dir + "/synthetic-tfrecords/instances_train"),
            image_id,
            tfrecords_dir,
            return_artifacts=True,
        )

        validate_tfrecord_artifacts(record_summary, record_image, image_id)

        with pytest.raises(DirNotFoundError):
            explore_tfrecords(
                Path(tmp_dir + "/synthetic-tfrecords/instances_xxx"),
                image_id,
                tfrecords_dir,
                return_artifacts=True,
            )

        with pytest.raises(DirNotFoundError):
            explore_tfrecords(
                Path(tmp_dir + "/synthetic-tfrecords/instances_train"),
                image_id,
                tfrecords_dir / "xxx",
                return_artifacts=True,
            )

        with pytest.raises(IdentifierNotFoundError):
            explore_tfrecords(
                Path(tmp_dir + "/synthetic-tfrecords/instances_train"),
                999999,
                tfrecords_dir,
                return_artifacts=True,
            )

    # Tests an action for removing the "/synthetic" directory.
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_dir = Path(tmp_dir + "/synthetic")
        dataset_dir.mkdir()
        create_dataset(dataset_dir)

    # Tests a reaction for missing directory
    with pytest.raises(DirNotFoundError):
        dataset_dir = Path(tmp_dir + "/synthetic")
        create_tfrecords(dataset_dir)

    # Tests a reaction for missing directory
    with pytest.raises(DirNotFoundError):
        dataset_dir = Path(tmp_dir + "/synthetic")
        create_dataset(dataset_dir, coco=coco)
        tfrecords_dir = Path(tmp_dir + "/somewhere/synthetic")
        create_tfrecords(dataset_dir, tfrecords_dir)


def test_csv_generator_transformer_explore_functions():
    """Tests the CSV data generate/transform/explore functions."""
    core_function_testscript(coco=False)


def test_coco_generator_transformer_explore_functions():
    """Tests the CSV data generate/transform/explore functions."""
    core_function_testscript(coco=True)


# ------------------------------------------------------------------------------
# GTE Action Tests
# ------------------------------------------------------------------------------


def core_action_testscript(capsys, coco):
    """Tests generate/transform/explore actions."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)
    explore(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Defines output directory.
        output_dir = Path(tmp_dir + "/output")
        output_dir.mkdir()

        # -----------------------------
        # Generates and checks dataset.
        # -----------------------------
        dataset_dir = Path(tmp_dir + "/synthetic")
        if coco:
            args = parser.parse_args(
                ["generate", str(dataset_dir), "--coco", "-v"]
            )
            args.func(args)

            validate_coco_generator(dataset_dir)
        else:
            args = parser.parse_args(["generate", str(dataset_dir), "-v"])
            args.func(args)

            validate_csv_generator(dataset_dir)

        # ----------------------------------------
        # Transforms CSV dataset to the TFRecords.
        # ----------------------------------------
        tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")
        args = parser.parse_args(
            ["transform", str(dataset_dir), str(tfrecords_dir), "-v"]
        )
        args.func(args)

        validate_transformer(tfrecords_dir)

        # -------------------
        # Explores TFRecords.
        # -------------------
        for kind in ["train", "val", "test"]:
            args = parser.parse_args(
                [
                    "explore",
                    str(tfrecords_dir / f"instances_{kind}"),
                ]
            )
            args.func(args)
            stdout, _ = capsys.readouterr()

            validate_tfrecords_stdout(stdout, kind)

            # Grabs image ID for exploring individual records.
            if kind == "train":
                image_id = re.search(
                    "(\\d+)\\s\\{(1|2|1,2|1,3|2,3|1,2,3)\\}",
                    stdout,
                ).group(1)

        # -------------------------
        # Explores single TFRecord.
        # -------------------------
        assert image_id is not None
        args = parser.parse_args(
            [
                "explore",
                str(tfrecords_dir / "instances_train"),
                image_id,
                str(output_dir),
            ]
        )
        args.func(args)
        stdout, _ = capsys.readouterr()

        validate_tfrecord_stdout(stdout, image_id, output_dir)


def test_csv_generator_transformer_explore_actions(capsys):
    """Tests the CSV data generate/transform/explore functions."""
    core_action_testscript(capsys, coco=False)


def test_coco_generator_transformer_explore_actions(capsys):
    """Tests the CSV data generate/transform/explore functions."""
    core_action_testscript(capsys, coco=True)


# ------------------------------------------------------------------------------
# Other Tests
# ------------------------------------------------------------------------------


def test_categories_map():
    categories_map = CategoriesMap([])
    assert categories_map[1] == 1
    assert categories_map[2] == 2
    assert 2 in categories_map

    categories_map = CategoriesMap([3, 4])
    assert categories_map[3] == 1
    assert categories_map[4] == 2
    assert 4 in categories_map
    assert 5 not in categories_map


def test_no_tfrecords_found(capsys):
    """Tests no TFRecords found."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    explore(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tfrecords_dir = Path(tmp_dir)
        args = parser.parse_args(["explore", str(tfrecords_dir)])
        args.func(args)
        stdout, _ = capsys.readouterr()

        validate_no_tfrecords(stdout)


def test_unknown_transformation():
    """Tests transform of an unknown dataset type."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset
        dataset_dir = Path(tmp_dir + "/synthetic")
        dataset_dir.mkdir()

        with pytest.raises(InvalidDatasetFormat):
            dataset_dir = Path(tmp_dir + "/synthetic")
            # The dataset_dir does not contain either CSV or COCO data.
            create_tfrecords(dataset_dir)


def test_reproducibility():
    """Tests generate/transform/explore functions."""

    def read_data(dataset_dir, filename):
        with open(dataset_dir / filename, "r") as file:
            return file.read()

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_orig_dir = Path(tmp_dir + "/synthetic_orig")
        create_dataset(dataset_orig_dir, dataset_size=100, random_state=42)
        dataset_same_dir = Path(tmp_dir + "/synthetic_same")
        create_dataset(dataset_same_dir, dataset_size=100, random_state=42)
        dataset_diff_dir = Path(tmp_dir + "/synthetic_diff")
        create_dataset(dataset_diff_dir, dataset_size=100, random_state=None)

        for kind in ["train", "val", "test"]:
            expected = read_data(dataset_orig_dir, f"instances_{kind}.csv")
            actual_same = read_data(dataset_same_dir, f"instances_{kind}.csv")
            actual_diff = read_data(dataset_diff_dir, f"instances_{kind}.csv")

            assert actual_same == expected
            assert actual_diff != expected

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_orig_dir = Path(tmp_dir + "/synthetic_orig")
        create_dataset(
            dataset_orig_dir, dataset_size=100, random_state=42, coco=True
        )
        dataset_same_dir = Path(tmp_dir + "/synthetic_same")
        create_dataset(
            dataset_same_dir, dataset_size=100, random_state=42, coco=True
        )
        dataset_diff_dir = Path(tmp_dir + "/synthetic_diff")
        create_dataset(
            dataset_diff_dir, dataset_size=100, random_state=None, coco=True
        )

        for kind in ["train", "val", "test"]:
            expected = read_data(
                dataset_orig_dir, f"annotations/instances_{kind}.json"
            ).replace("_orig", "")
            actual_same = read_data(
                dataset_same_dir, f"annotations/instances_{kind}.json"
            ).replace("_same", "")
            actual_diff = read_data(
                dataset_diff_dir, f"annotations/instances_{kind}.json"
            ).replace("_diff", "")

            # Removes time stamps.
            expected = re.sub(
                r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}", "", expected
            )
            actual_same = re.sub(
                r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}", "", actual_same
            )
            actual_diff = re.sub(
                r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}", "", actual_diff
            )

            assert actual_same == expected
            assert actual_diff != expected


# ------------------------------------------------------------------------------
# Data Loader Tests
# ------------------------------------------------------------------------------


def test_data_loader(capsys):
    """Tests data loader for model for training and validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_dir = Path(tmp_dir + "/synthetic")
        tfrecords_dir = Path(tmp_dir + "/synthetic-tfrecords")

        create_dataset(dataset_dir)
        create_tfrecords(dataset_dir)

        # ------------------------------------
        # Tests checkers for loader arguments.
        # ------------------------------------
        with pytest.raises(ValueError, match="The output schema is empty."):
            load_tfrecords(tfrecords_dir, output_schema="")

        with pytest.raises(
            ValueError,
            match="The output schema contains multiple 'I'",
        ):
            load_tfrecords(tfrecords_dir, output_schema="II")

        with pytest.raises(
            ValueError,
            match="The output schema contains multiple 'B",
        ):
            load_tfrecords(tfrecords_dir, output_schema="BB")

        with pytest.raises(
            ValueError,
            match="The output schema contains multiple 'M",
        ):
            load_tfrecords(tfrecords_dir, output_schema="MM")

        with pytest.raises(
            ValueError,
            match="The output schema contains multiple 'C",
        ):
            load_tfrecords(tfrecords_dir, output_schema="CC")

        with pytest.raises(
            ValueError,
            match="The output schema contains two consequent commas.",
        ):
            load_tfrecords(tfrecords_dir, output_schema=",,")

        with pytest.raises(
            ValueError,
            match="The output schema contains unknown element 'X'.",
        ):
            load_tfrecords(tfrecords_dir, output_schema="X")

        # ----------------------------------
        # Tests data load for single output.
        # ----------------------------------
        dataset, steps_per_epoch = load_tfrecords(
            tfrecords_dir / "instances_train", output_schema="C", verbose=True
        )
        assert steps_per_epoch > 0
        for X, y in dataset:

            assert X.shape == (
                BATCH_SIZE,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                IMAGE_CHANNELS,
            )
            assert y.shape == (BATCH_SIZE, NUM_DETECTING_OBJECTS, 4)
            break

        # ----------------------------------------
        # Tests data load for concatinated output.
        # ----------------------------------------
        dataset, steps_per_epoch = load_tfrecords(
            tfrecords_dir / "instances_train", output_schema="BMC"
        )
        assert steps_per_epoch > 0
        for X, y in dataset:
            assert X.shape == (
                BATCH_SIZE,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                IMAGE_CHANNELS,
            )
            assert y.shape == (
                BATCH_SIZE,
                NUM_DETECTING_OBJECTS,
                IMAGE_WIDTH * IMAGE_HEIGHT + 4 + 4,
            )
            break

        # ----------------------------------------
        # Tests data load for concatinated output.
        # ----------------------------------------
        dataset, steps_per_epoch = load_tfrecords(
            tfrecords_dir / "instances_train", output_schema="I,BMC"
        )
        assert steps_per_epoch > 0
        for Xi, (Xo, y) in dataset:
            assert Xi.shape == (
                BATCH_SIZE,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                IMAGE_CHANNELS,
            )
            assert Xo.shape == (
                BATCH_SIZE,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                IMAGE_CHANNELS,
            )
            assert y.shape == (
                BATCH_SIZE,
                NUM_DETECTING_OBJECTS,
                IMAGE_WIDTH * IMAGE_HEIGHT + 4 + 4,
            )
            break

        dataset, steps_per_epoch = load_tfrecords(
            tfrecords_dir / "instances_train", output_schema="B,C"
        )
        assert steps_per_epoch > 0
        for Xi, (yb, yc) in dataset:
            assert X.shape == (
                BATCH_SIZE,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                IMAGE_CHANNELS,
            )
            assert yb.shape == (BATCH_SIZE, NUM_DETECTING_OBJECTS, 4)
            assert yc.shape == (BATCH_SIZE, NUM_DETECTING_OBJECTS, 4)
            break
