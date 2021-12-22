"""
Test for generator functions form `squids/dataset.py`.
"""

import os
import tempfile

from squids.image import Palette, Background
from squids.dataset import create_csv_dataset


def test_create_csv_dataset():
    """Tests the `create_csv_dataset` function."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_csv_dataset(
            tmp_dir, 100, 64, 64, Palette.COLOR, Background.WHITE, 3, True
        )

        # generate some random files in it
        assert set(os.listdir(tmp_dir)) == set(
            ["train.csv", "images", "test.csv", "val.csv"]
        )
        assert len(os.listdir(tmp_dir + "/images")) == 100
