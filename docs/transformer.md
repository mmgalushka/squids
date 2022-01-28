# Transform Dataset

The transformation process allows the conversion of a synthetic or real dataset in either CSV or COCO formats to TFRecords. The conversion function will automatically determine the format of the source dataset and use the appropriate mechanisms for its loading.

## Usage

You can initiate the process of transforming a dataset using the Python function or command line. For more information about transforming a dataset see [PyDoc](#pydoc)

<!-- Usage tab (Python|Shell)  -->

=== "Python"
    ```py
    from squids import create_tfrecords
    create_tfrecords()

    ```
=== "Shell"
    ```shell
        ~$ python squids.main transform [-h] [-s NUMBER] [--tfrecords-image-width PIXELS] [--tfrecords-image-height PIXELS] [--select-categories CATEGORY_ID [CATEGORY_ID ...]] [-v] [DATASET_DIR] [TFRECORDS_DIR]

    positional arguments:
        DATASET_DIR           a source dataset directory, if not defined, it will be selected as the './dataset/synthetic'
        TFRECORDS_DIR         a TFRecords directory, if not defined, it will be created in the <DATASET_DIR> parent under the name '<DATASET_DIR>-tfrecords

    optional arguments:
        -h, --help            show this help message and exit
        -s NUMBER, --tfrecords-size NUMBER
                            a number of images per partion (default=256)
        --tfrecords-image-width PIXELS
                            a TFRecords image width resize to (default=64)
        --tfrecords-image-height PIXELS
                            a TFRecords image height resize to (default=64)
        --select-categories CATEGORY_ID [CATEGORY_ID ...]
                            a list of selected category IDs
        -v, --verbose         a flag to set verbose mode
    ```

## Outcome

This function also can be run with only default arguments. In this case, it will look for the default dataset directory `dataset/synthetic`. Using the directory structure the transformation method determines the data format (CSV or COCO). Then it creates a new directory `dataset/synthetic-tfrecords` where stores the transformed records.

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

AS you can see the top directory `dataset/synthetic-tfrecords` contains three subdirectories `instances_train`, `instances_val`, and `instances_test` containing file-parts with TFRecords for training, validation, and testing respectively. Each of these folders usually includes a different number of part-files, depending on the volume of the original dataset for each machine learning task.

## PyDoc

::: squids.tfrecords.maker
    selection:
      members:
        - create_tfrecords
