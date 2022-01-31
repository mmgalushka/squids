"""A module for TFRecords related errors."""

from __future__ import annotations

from pathlib import Path


class TFRecordsError(Exception):
    pass


class DirNotFoundError(TFRecordsError):
    """
    Raises if a directory does not exist.

    Args:
        name (str):
            A name describing directory in the error message
            (it should be meaningful).
        path (path):
            A path to the directory causes a problem.
    """

    def __init__(self, name: str, path: Path):
        super().__init__(f"The {name} directory does not exist: {path};")


class InvalidDatasetFormat(TFRecordsError):
    """Raises if the input dataset has invalid CSV or COCO format."""

    def __init__(self):
        super().__init__("The input dataset has invalid CSV or COCO format;")


class IdentifierNotFoundError(TFRecordsError):
    """Raises if a record with the specified identifier has not been found."""

    def __init__(self, identifier: int, path: Path):

        super().__init__(
            f"The record with identifier :{identifier} "
            f"has not been found in: {path};"
        )
