"""
Test for action functions form `squids/actions.py`.
"""

import os
import argparse
import tempfile

from squids.actions import generate


def test_generate():
    """Tests the `create_csv_dataset` function."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate(subparsers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = parser.parse_args(["generate", "-o", tmp_dir, "-s", "100"])
        args.func(args)

        assert set(os.listdir(tmp_dir)) == set(
            ["train.csv", "images", "test.csv", "val.csv"]
        )
        assert len(os.listdir(tmp_dir + "/images")) == 100
