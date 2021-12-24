"""
Test functions for processing datasets form `squids/dataset.py` and
`squids/tfrecords.py`.
"""

import os
import tempfile

from squids.image import Palette, Background
from squids.dataset import create_csv_dataset
from squids.tfrecords import create_tfrecords


def test_csv_dataset():
    """Tests operations with CSV datasets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generates and checks CSV dataset
        tmp_dataset_dir = tmp_dir + "/synthetic"
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

        assert set(os.listdir(tmp_dataset_dir)) == set(
            ["train.csv", "images", "test.csv", "val.csv"]
        )
        assert len(os.listdir(tmp_dataset_dir + "/images")) == 100

        # Transforms CSV dataset to the TFRecords
        create_tfrecords(tmp_dataset_dir, ["rectangle", "triangle"])

        tmp_tfrecords_dir = tmp_dataset_dir + "-tfrecords"
        assert set(os.listdir(tmp_tfrecords_dir)) == set(
            ["train", "test", "val"]
        )
        assert len(os.listdir(tmp_tfrecords_dir + "/train")) > 0
        assert len(os.listdir(tmp_tfrecords_dir + "/val")) > 0
        assert len(os.listdir(tmp_tfrecords_dir + "/test")) > 0
