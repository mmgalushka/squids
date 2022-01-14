"""
A module for handling configuration.
"""

from pathlib import Path

import yaml

IMAGE_WIDTH = 64
"""The width of synthetic and TFRecords images."""

IMAGE_HEIGHT = 64
"""The height of synthetic and TFRecords images."""

IMAGE_CHANNELS = 3
"""The number of image channels."""

IMAGE_CAPACITY = 3
"""The maximum number of geometrical shapes per image (minimum is 1)."""

DATASET_DIR = "dataset/synthetic"
"""The default location for generating synthetic dataset."""

DATASET_SIZE = 1000
"""The number of generatign dataset records (images + annotations)."""


class Config(dict):
    """An experiment configuration.

    Args:
        source (dict): The source for creating an experiment configuration.
    """

    def __init__(self, source: dict = None):
        super().__init__({} or source)

    def __getitem__(self, key):
        path = key.split("/")
        branch = self.copy()

        for p in path:
            if isinstance(branch, list):
                try:
                    branch = branch[int(p)]
                except IndexError:
                    raise KeyError("%s in %s;" % (p, key))
            else:
                branch = branch[p]

        return branch

    def __setitem__(self, key, value):
        path = key.split("/")
        branch = self.copy()

        if len(path) == 1:
            super().__setitem__(key, value)
        elif len(path) > 1:
            for p in path[:-1]:
                if isinstance(branch, list):
                    try:
                        branch = branch[int(p)]
                    except IndexError:
                        raise KeyError("%s in %s;" % (p, key))
                else:
                    branch = branch[p]

            if isinstance(branch, list):
                try:
                    branch[int(path[-1])] = value
                except IndexError:
                    raise KeyError("%s in %s;" % (path[-1], key))
            else:
                branch[path[-1]] = value
        else:
            raise KeyError("empty key;")

    def __delitem__(self, key):
        path = key.split("/")
        branch = self.copy()

        if len(path) == 1:
            super().__delitem__(key)
        elif len(path) > 1:
            for p in path[:-1]:
                if isinstance(branch, list):
                    try:
                        branch = branch[int(p)]
                    except IndexError:
                        raise KeyError("%s in %s;" % (p, key))
                else:
                    branch = branch[p]

            if isinstance(branch, list):
                try:
                    del branch[int(path[-1])]
                except IndexError:
                    raise KeyError("%s in %s;" % (path[-1], key))
            else:
                del branch[path[-1]]
        else:
            raise KeyError("empty key;")

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False


def create_config(
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    image_channels: int = IMAGE_CHANNELS,
):
    source = {
        "input": {
            "image": {
                "width": image_width,
                "height": image_height,
                "channels": image_channels,
            },
        },
    }
    return Config(source)


def load_config(fp: Path) -> Config:
    with fp.open("r") as f:
        return Config(yaml.safe_load(f))


def save_config(fp: Path, config: Config):
    with fp.open("w") as f:
        yaml.dump(dict(config), f)
