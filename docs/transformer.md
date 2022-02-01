# Transform Dataset

The transformation process allows the conversion of a synthetic or real dataset in either CSV or COCO formats to [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord).

## Usage

The process of transforming a dataset can be initiated by the Python `create_tfrecords()` function  (see [PyDoc](#pydoc) for more information) or via a command line.

<!-- Usage tab (Python|Shell)  -->

=== "Python"
    ```py
    from squids import create_tfrecords
    create_tfrecords()

    ```
=== "Shell"
    ```shell
    ~$ python squids.main transform [-h] [-s NUMBER] [--image-width PIXELS] [--image-height PIXELS]
                                [--select-categories CATEGORY_ID [CATEGORY_ID ...]] [-v]
                                [DATASET_DIR] [TFRECORDS_DIR]

    positional arguments:
    DATASET_DIR           a source dataset directory, if not defined, it will be selected as the 'dataset/synthetic'
    TFRECORDS_DIR         a TFRecords directory, if not defined, it will be created in the <DATASET_DIR> parent under the name
                            '<DATASET_DIR>-tfrecords'

    optional arguments:
    -h, --help            show this help message and exit
    -s NUMBER, --size NUMBER
                            a number of images per partition (default=256)
    --image-width PIXELS  a TFRecords image width resize to (default=64)
    --image-height PIXELS
                            a TFRecords image height resize to (default=64)
    --select-categories CATEGORY_ID [CATEGORY_ID ...]
                            a list of selected category IDs
    -v, --verbose         a flag to set verbose mode
    ```

## Outcome

If this function runs with default arguments, it looks for the default dataset directory `dataset/synthetic`. Using the directory structure the transformation method determines a dataset format (CSV or COCO). Then it creates a new directory `dataset/synthetic-tfrecords` which stores the transformed records.

    ```text
    dataset/synthetic-tfrecords
        instances_train
            part-0.tfrecord
            part-1.tfrecord
            part-2.tfrecord
            ...
        instances_val
            part-0.tfrecord
            part-1.tfrecord
            part-2.tfrecord
            ...
        instances_test
            part-0.tfrecord
            part-1.tfrecord
            part-2.tfrecord
            ...
    ```

As you can see the top directory `dataset/synthetic-tfrecords` contains three subdirectories `instances_train`, `instances_val`, and `instances_test` containing file-parts with TFRecords for training, validation, and testing respectively. Each of these folders usually includes a different number of part-files, depending on the volume of the original dataset for each machine learning task. Each `part-*.tfrecord` file contains the specified number of records by the function argument `size` of the command line option `-s, --size`. By default, the size is 256 records.

## TFRecord Specification

Each TFRecord consists of the following properties:

| Key | Sequence of Type | Length | Description |
|-----|------------------|--------|-------------|
| image/id           | tf.int64   | 1 | The record identifier (the same as image ID) |
| image/size         | tf.int64   | 2 | The image size consisted of integer values for `WIDTH` and `HEIGHT`  |
| image/data         | tf.string  | WIDTH x HEIGHT x 3 | The list of flattened image pixels, the total length is a product of `WIDTH`, `HEIGHT`, and  3 (the number of channels) |
| annotations/number | tf.int64   | 1 | The number of annotated objects (`ANNO`) in the image |
| bboxes/data        | tf.float32 | ANNO x 4 | The list of bounding boxes coordinates which length is a product of `ANNO` and 4 (two coordinates `x`, `y` for the anchor point and two for the box `w`-width, `h`-heights )) |
| masks/data         | tf.string  | ANNO x WIDTH x HEIGHT | The list of bounding boxes coordinates which length is a product `WIDTH`, `HEIGHT`, and  `ANNO` (each annotation has an image size binary mask) |
| category/ids       | tf.int64   | ANNO | The list of annotated category identifiers of length `ANNO` |
| category/max       | tf.int64   | 1 | The maximum category identifier across all records used for creating one-hot categories encoding during records loading |

Use the TFRecords [explorer](explorer.md) for visualizing this information.

## PyDoc

::: squids.tfrecords.maker
    selection:
      members:
        - create_tfrecords
