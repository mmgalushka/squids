"""
A module for TFRecords related errors.
"""

from __future__ import annotations


class TFRecordsDirNotFoundError(FileNotFoundError):
    def __init__(self, tfrecords_dir: str):
        super().__init__(
            f"TFRecords directory does not existing: {tfrecords_dir}"
        )


class TFRecordIdentifierNotFoundError(ValueError):
    def __init__(self, image_id: int, tfrecords_dir: str):
        super().__init__(
            f"The TFRecords record (ID:{image_id}) "
            f"has not been found in {tfrecords_dir}."
        )


class OutputDirNotFoundError(FileNotFoundError):
    def __init__(self, output_dir: str):
        super().__init__(f"Output directory does not existing: {output_dir}")
