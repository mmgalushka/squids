# """
# Test for the feature read/write functions form `squids/feature.py`.
# """

# import numpy as np
# import tensorflow as tf
# from PIL import Image, ImageDraw

# from squids.image import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS
# from squids.feature import (
#     image_to_feature,
#     feature_to_image,
#     bboxes_to_feature,
#     feature_to_bboxes,
#     segmentations_to_feature,
#     feature_to_segmentations,
#     category_ids_to_feature,
#     feature_to_category_ids,
# )
# from squids.color import BLACK_COLOR, WHITE_COLOR

# EPS = tf.constant(0.001, tf.float32)

# # -----------------------------------------------------------------------------
# # Tests: Image -> Feature -> Image transformars
# # -----------------------------------------------------------------------------


# def test_image_to_feature():
#     """Tests the `image_to_feature` function."""
#     actual_feature = image_to_feature(
#         Image.new(mode="RGB", size=(IMAGE_WIDTH, IMAGE_HEIGHT)),
#         IMAGE_WIDTH,
#         IMAGE_HEIGHT,
#     )
#     expected_feature = {
#         "image/shape": tf.train.Feature(
#             int64_list=tf.train.Int64List(
#                 value=[IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]
#             )
#         ),
#         "image/content": tf.train.Feature(
#             bytes_list=tf.train.BytesList(
#                 value=[
#                     np.array(
#                         Image.new(mode="RGB", size=(IMAGE_WIDTH, IMAGE_HEIGHT))
#                     ).tostring()
#                 ]
#             )
#         ),
#     }

#     assert actual_feature == expected_feature


# def test_feature_to_image():
#     """Tests the `feature_to_image` function."""
#     feature = {
#         "image/shape": tf.constant(
#             [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], dtype=tf.int64
#         ),
#         "image/content": tf.constant(
#             [
#                 np.array(
#                     Image.new(mode="RGB", size=(IMAGE_WIDTH, IMAGE_HEIGHT))
#                 ).tostring()
#             ],
#             dtype=tf.string,
#         ),
#     }

#     actual_image = feature_to_image(feature)
#     expected_image = Image.new(mode="RGB", size=(IMAGE_WIDTH, IMAGE_HEIGHT))
#     delta = tf.reduce_sum(tf.abs(actual_image - expected_image))
#     assert tf.math.less(delta, EPS)


# # -----------------------------------------------------------------------------
# # Tests: BBoxes -> Feature -> BBoxes transformars
# # -----------------------------------------------------------------------------


# def test_bboxes_to_feature():
#     """Tests the `bboxes_to_feature` function."""
#     actual_feature = bboxes_to_feature([[1, 2, 3, 4], [5, 6, 7, 8]])
#     expected_feature = {
#         "bboxes/number": tf.train.Feature(
#             int64_list=tf.train.Int64List(value=[2])
#         ),
#         "bboxes/data": tf.train.Feature(
#             int64_list=tf.train.Int64List(value=[1, 2, 3, 4, 5, 6, 7, 8])
#         ),
#     }
#     assert actual_feature == expected_feature


# def test_feature_to_bboxes():
#     """Tests the `feature_to_bboxes` function."""
#     feature = {
#         "bboxes/number": tf.constant([2], dtype=tf.int64),
#         "bboxes/data": tf.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.int64),
#     }

#     actual_bboxes = feature_to_bboxes(feature, 5)
#     expected_bboxes = tf.constant(
#         [
#             [1.0, 2.0, 3.0, 4.0],
#             [5.0, 6.0, 7.0, 8.0],
#             [0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0],
#         ],
#         dtype=tf.float32,
#     )
#     delta = tf.reduce_sum(tf.abs(actual_bboxes - expected_bboxes))
#     assert tf.math.less(delta, EPS)


# # -----------------------------------------------------------------------------
# # Tests: Segmentations -> Feature -> Segmentations transformars
# # -----------------------------------------------------------------------------


# def test_segmentations_to_feature():
#     """Tests the `segmentations_to_feature` function."""
#     actual_feature = segmentations_to_feature(
#         [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8]],
#         IMAGE_WIDTH,
#         IMAGE_HEIGHT,
#     )

#     image_1 = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), str(BLACK_COLOR))
#     drawing_1 = ImageDraw.Draw(image_1)
#     drawing_1.polygon(
#         [1, 2, 3, 4, 5, 6], fill=str(BLACK_COLOR), outline=str(WHITE_COLOR)
#     )

#     expected_feature = {
#         "segmentations/number": tf.train.Feature(
#             int64_list=tf.train.Int64List(value=[2])
#         ),
#         "segmentations/schema": tf.train.Feature(
#             int64_list=tf.train.Int64List(value=[6, 8])
#         ),
#         "segmentations/data": tf.train.Feature(
#             float_list=tf.train.FloatList(
#                 value=[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8]
#             )
#         ),
#     }
#     assert actual_feature == expected_feature


# # def test_feature_to_segmentations():
# #     """Tests the `feature_to_segmentations` function."""
# #     feature = {
# #         "segmentations/number": tf.constant([2], dtype=tf.int64),
# #         "segmentations/number": tf.constant([2, 3, 4], dtype=tf.int64),
# #         "segmentations/data": tf.constant(
# #             [1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.float32
# #         ),
# #     }

# #     actual_bboxes = feature_to_segmentations(feature, 5)
# #     expected_bboxes = tf.constant(
# #         [
# #             [1.0, 2.0, 3.0, 4.0],
# #             [5.0, 6.0, 7.0, 8.0],
# #             [0.0, 0.0, 0.0, 0.0],
# #             [0.0, 0.0, 0.0, 0.0],
# #             [0.0, 0.0, 0.0, 0.0],
# #         ],
# #         dtype=tf.float32,
# #     )
# #     delta = tf.reduce_sum(tf.abs(actual_bboxes - expected_bboxes))
# #     assert tf.math.less(delta, EPS)


# # -----------------------------------------------------------------------------
# # Tests: Category IDs -> Feature -> Category IDs transformars
# # -----------------------------------------------------------------------------


# def test_category_ids_to_feature():
#     """Tests the `category_ids_to_feature` function."""
#     actual_feature = category_ids_to_feature([1, 2, 3])
#     expected_feature = {
#         "categories/number": tf.train.Feature(
#             int64_list=tf.train.Int64List(value=[3])
#         ),
#         "categories/ids": tf.train.Feature(
#             int64_list=tf.train.Int64List(value=[1, 2, 3])
#         ),
#     }
#     assert actual_feature == expected_feature


# def test_feature_to_category_ids():
#     """Tests the `feature_to_category_ids` function."""
#     feature = {
#         "categories/number": tf.constant([3], dtype=tf.int64),
#         "categories/ids": tf.constant([1, 2, 3], dtype=tf.int64),
#     }
#     actual_category_ids = feature_to_category_ids(feature, 5)
#     expected_category_ids = tf.constant(
#         [
#             [0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0],
#             [0.0, 0.0, 0.0, 1.0],
#             [1.0, 0.0, 0.0, 0.0],
#             [1.0, 0.0, 0.0, 0.0],
#         ],
#         dtype=tf.float32,
#     )
#     delta = tf.reduce_sum(tf.abs(actual_category_ids - expected_category_ids))
#     assert tf.math.less(delta, EPS)
