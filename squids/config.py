"""A module configuration constants."""

IMAGE_WIDTH = 64
"""The width of synthetic and TFRecords images."""

IMAGE_HEIGHT = 64
"""The height of synthetic and TFRecords images."""

IMAGE_CHANNELS = 3
"""The number of image channels."""

IMAGE_CAPACITY = 3
"""The maximum number of geometrical shapes per image (minimum is 1)."""

ADDING_SHAPES = "rte"
"""The shapes adding to generated images."""

NOISE_LEVEL = 0.0
"""The noise lebel."""

BLUR_LEVEL_X = 0.0
"""The blurring level on X-axis."""

BLUR_LEVEL_Y = 0.0
"""The blurring level on Y-axis."""

DATASET_DIR = "dataset/synthetic"
"""The default location for generating synthetic dataset."""

DATASET_SIZE = 1000
"""The number of generatign dataset records (images + annotations)."""

TFRECORDS_SIZE = 256
"""The number of TFRecords in a partition file."""

OUTPUT_SCHEMA = "B"
"""The output feature (`y`) format for data generator."""

NUM_DETECTING_OBJECTS = 10
"""The number of detecting objects per image."""

BATCH_SIZE = 32
"""The number of samples per batch."""
