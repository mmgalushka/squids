"""
Test for action functions form `squids/actions.py`.
"""

import os
import argparse
import tempfile

from squids.actions import generate, transform


def test_generate():
    """Tests the `create_csv_dataset` function."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)
    transform(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset
        tmp_dataset_dir = tmp_dir + "/synthetic"
        args = parser.parse_args(
            ["generate", "-o", tmp_dataset_dir, "-s", "100"]
        )
        args.func(args)

        assert set(os.listdir(tmp_dataset_dir)) == set(
            ["train.csv", "images", "test.csv", "val.csv"]
        )
        assert len(os.listdir(tmp_dataset_dir + "/images")) == 100

        # Transforms CSV dataset to the TFRecords
        tmp_tfrecords_dir = tmp_dataset_dir + "-tfrecords"
        args = parser.parse_args(
            ["transform", "-i", tmp_dataset_dir, "-o", tmp_tfrecords_dir]
        )
        args.func(args)

        assert set(os.listdir(tmp_tfrecords_dir)) == set(
            ["train", "test", "val"]
        )
        assert len(os.listdir(tmp_tfrecords_dir + "/train")) > 0
        assert len(os.listdir(tmp_tfrecords_dir + "/val")) > 0
        assert len(os.listdir(tmp_tfrecords_dir + "/test")) > 0
