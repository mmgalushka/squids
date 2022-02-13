"""A module for handling TFRecords features."""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from ..config import IMAGE_CHANNELS

FEATURE_KEYS_MAP = {
    "image/id": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    "image/size": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    "image/data": tf.io.FixedLenSequenceFeature(
        [], tf.string, allow_missing=True
    ),
    "annotations/number": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    "bboxes/data": tf.io.FixedLenSequenceFeature(
        [], tf.float32, allow_missing=True
    ),
    "masks/data": tf.io.FixedLenSequenceFeature(
        [], tf.string, allow_missing=True
    ),
    "category/ids": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    "category/max": tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
}
"""The mapping feature keys, for parsing a TFRecord."""


def items_to_features(
    image_id: int,
    image: Image,
    image_width: int,
    image_height: int,
    bboxes: list,
    segmentations: list,
    category_ids: list,
    category_max_id: int,
) -> dict:
    """Transforms image and annotation items TFRecord features.

    Args:
        image:
            The image object to transform.
        image_width:
            The target width to resize image.
        image_height:
            The target height to resize image.
        bboxes:
            The bounding boxes around annotated objects.
        segmentations:
            The polygons around annotated objects.
        category_ids:
            The list of categories for annotated objects.

    Returns:
        features (dict):
            The dictionary with features.
    """

    def scale_coordinates(coordinates):
        original_image_width, original_image_height = image.size

        scale_ratio_for_width = image_width / original_image_width
        scale_ratio_for_height = image_height / original_image_height

        return [
            scale_ratio_for_width * coordinate
            if (i % 2) == 0
            else scale_ratio_for_height * coordinate
            for i, coordinate in enumerate(coordinates)
        ]

    # Resize image to the target size which will be used durinn the model
    # training.
    image_size = (image_width, image_height)
    image_data = np.array(image.resize(image_size)).flatten()

    # Gets the total number annotations in the image.
    annotations_number = len(bboxes)

    # Gets bounding boxes and masks scaled according to the
    # target image size.
    bboxes_data = []
    masks_data = []
    for bbox, segmentation in zip(bboxes, segmentations):
        bboxes_data.extend(scale_coordinates(bbox))

        # Mask is a "binary" image with black background.
        mask = Image.new("1", image_size, 0)
        drawing = ImageDraw.Draw(mask)
        # The object segmentation is showing as white area.
        drawing.polygon(scale_coordinates(segmentation), fill=1, outline=1)
        masks_data.extend(np.array(mask).flatten())
    masks_data = np.array(masks_data)

    return {
        "image/id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[image_id])
        ),
        "image/size": tf.train.Feature(
            int64_list=tf.train.Int64List(value=image_size)
        ),
        "image/data": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_data.tostring()])
        ),
        "annotations/number": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[annotations_number])
        ),
        "bboxes/data": tf.train.Feature(
            float_list=tf.train.FloatList(value=bboxes_data)
        ),
        "masks/data": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[masks_data.tostring()])
        ),
        "category/ids": tf.train.Feature(
            int64_list=tf.train.Int64List(value=category_ids)
        ),
        "category/max": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[category_max_id])
        ),
    }


def features_to_items(
    features: dict, num_detecting_objects: int = None
) -> dict:
    """Transforms TFRecord features to image and annotation items.

    Args:
        features (dict):
            The dictionary with parsed feature.
        num_detecting_objects (int):
            The number of detecting objects.

    Returns:
        image_id (int):
            The unique image identifier.
        image_arr (Tensor):
            The image data.
        bboxes (Tensor):
            The bounding boxes around annotated objects.
        masks (Tensor):
            The masks around annotated objects.
        category_ids (Tensor):
            The list of categories for annotated objects.
    """
    # Gets an image.
    image_id = features["image/id"][0]
    image_width = features["image/size"][0]
    image_height = features["image/size"][1]
    image_shape = (image_width, image_height, IMAGE_CHANNELS)
    mask_size = image_width * image_height

    image = tf.io.decode_raw(features["image/data"], tf.uint8)
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    n = features["annotations/number"][0]

    # Gets bounding boxes;
    bboxes = features["bboxes/data"]
    bboxes = tf.reshape(bboxes, (-1, 4))

    # Gets segments masks;
    masks = tf.io.decode_raw(features["masks/data"], tf.uint8)
    masks = tf.reshape(masks, (-1, mask_size))

    # Gets category IDs;
    category_ids = features["category/ids"]
    category_max_id = features["category/max"][0]

    # Slices a pads data depending on number of detecting objects;
    if num_detecting_objects:
        if n < num_detecting_objects:
            # If the obtained number of record less than the detection
            # capacity, then the records must be padded.
            bboxes = tf.pad(
                bboxes,
                [[0, num_detecting_objects - n], [0, 0]],
                constant_values=0,
            )
            masks = tf.pad(
                masks,
                [[0, num_detecting_objects - n], [0, 0]],
                constant_values=0,
            )
            category_ids = tf.pad(
                category_ids,
                [[0, num_detecting_objects - n]],
                constant_values=0,
            )
        else:
            # If the obtained number of record binding boxes is greater
            # than the detection capacity, then the records must be sliced.
            bboxes = tf.slice(bboxes, [0, 0], [num_detecting_objects, 4])
            masks = tf.slice(masks, [0, 0], [num_detecting_objects, mask_size])
            category_ids = tf.slice(category_ids, [0], [num_detecting_objects])

    # The `num_detecting_objects==1` is a special case. Effective it
    # makes this task as simple image classification, single object
    # localization, or mask detection. In this case, the 2D output is
    # squeezed to 1D. For example, let assume the `num_detecting_objects==1`
    # when we predict object category. It makes this task as image
    # classification where one-hot encoding `[[0, 1, 0, 0]]` (2D tensor)
    # should be transformed to `[0, 1, 0, 0]` (1D tensor).
    if num_detecting_objects == 1:
        bboxes = tf.squeeze(bboxes)
        masks = tf.squeeze(masks)
        category_ids = tf.squeeze(category_ids)

    bboxes = tf.cast(bboxes, dtype=tf.float32)
    masks = tf.cast(masks, dtype=tf.float32)
    # +1: to the categories_number to allow "no object" category with ID == 0
    category_ids = tf.one_hot(category_ids, depth=int(category_max_id + 1))

    return image_id, image, bboxes, masks, category_ids
