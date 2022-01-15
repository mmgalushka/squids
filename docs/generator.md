# Generate Dataset

Synthetic datasets can help you quickly validate your ideas during building ML models and also allow help you to write unit tests for your ML model without exposing real data images.

## Usage

You can initiate the process of generating a synthetic dataset using the Python function or command line. For more information about generating a synthetic dataset see [PyDoc](#pydoc)

<!-- Usage tab (Python|Shell)  -->

=== "Python"
    ```py
    # To generate CSV data
    from squids import create_csv_dataset
    create_csv_dataset()

    # To generate COCO data
    from squids import create_coco_dataset
    create_coco_dataset()
    ```
=== "Shell"
    ```shell
    ~$ python squids.main generate [-h] [-s NUMBER] [--coco]
                                    [--image-width PIXELS] [--image-height PIXELS]
                                    [--image-palette {gray,color,binary}]
                                    [--image-background {white,black}]
                                    [--image-capacity NUMBER] [-v] [DATASET_DIR]

    positional arguments:
    DATASET_DIR           a generating dataset directory, (default './dataset/synthetic')

    optional arguments:
    -h, --help            show this help message and exit
    -s NUMBER, --dataset-size NUMBER
                            a number of generated data samples (default=1000)
    --coco                a flag to generate dataset in the COCO format
    --image-width PIXELS  a generated image width (default=64)
    --image-height PIXELS
                            a generated image height (default=64)
    --image-palette {gray,color,binary}
                            a generated palette (default='color')
    --image-background {white,black}
                            a generated background (default='white')
    --image-capacity NUMBER
                            a number of shapes per image (default=3)
    -v, --verbose         a flag to set verbose mode
    ```

## Outcome

This library allows the generation of synthetic data in CSV and COCO formats. Please note, that both formats will lead to the same outcome during the transformation of these data to tfrecords. You can choose which format suits you more.

It is important to understand the data structure for both formats. In the future, you may adopt the same formats for storing your real data, which significantly simplifies downstream processes such as transformation, exploration, and model training.

=== "CSV"
    This is probably the most intuitive format for generating synthetic (or storing real) CV data.

    If you leave all function arguments to default, it will create the following files structure (under `dataset/synthetic`) in the current folder (from which Python-code or command-line has been executed).

    ```text
    dataset/synthetic/
        images/
        instances_test.csv
        instances_val.csv
        instances_train.csv
        categories.json
    ```

    | Artifact            | Type   | Comment |
    |---------------------|--------|---------|
    | images              | Dir    | Contains all generated images |
    | instances_train.csv | File   | Contains training records including image and annotations |
    | instances_val.csv   | File   | Contains validation records including image and annotations |
    | instances_test.csv  | File   | Contains test records including image and annotations |
    | categories.json     | File   | Contains information about annotated categories |

    As an example let's review the structure of the `instances_train.csv` file. Please note, the structure of the other two CSV files is the same.

    | image_id | file_name | bboxes | segmentations | category_ids |
    |----------|-----------|--------|---------------|--------------|
    | 0 | 0 | image0.jpg | [[4, 11, 16, 32]] | [[10, 11, 20, 43, 4, 43]] | [2] |
    | 1 | 1 | image1.jpg | [[44, 17, 13, 23], [3, 2, 11, 9]] | [[48, 17, 57, 40, 44, 40], [3, 2, 14, 2, 14, 1... | [2, 1] |
    | 2 | 2 | image2.jpg | [[2, 46, 21, 8], [21, 6, 8, 26]] | [[2, 46, 23, 46, 23, 54, 2, 54], [29, 6, 29, 3... | [1, 2] |
    | 3 | 3 | image3.jpg | [[11, 31, 17, 21], [0, 31, 18, 31]] | [[24, 31, 28, 52, 11, 52], [0, 31, 18, 31, 18,... | [2, 1] |
    | 4 | 4 | image4.jpg | [[27, 21, 25, 7], [7, 32, 24, 27]] | [[38, 21, 52, 28, 27, 28], [7, 32, 31, 32, 31,... | [2, 1] |
    ... | ... | ... | ... | ... | ... |
    | 706 | 991 | image991.jpg | [[21, 5, 30, 26], [3, 12, 26, 27]] | [[21, 5, 51, 5, 51, 31, 21, 31], [20, 12, 29, ... | [1, 2] |
    | 707 | 993 | image993.jpg | [[37, 37, 18, 26]] | [[49, 37, 55, 63, 37, 63]] | [2] |
    | 708 | 994 | image994.jpg | [[5, 43, 9, 17]] | [[5, 43, 14, 43, 14, 60, 5, 60]] | [1] |
    | 709 | 995 | image995.jpg | [[1, 47, 27, 10]] | [[1, 47, 28, 47, 28, 57, 1, 57]] | [1] |
    | 710 | 997 | image997.jpg | [[52, 22, 7, 30], [10, 26, 17, 15]] | [[55, 22, 59, 52, 52, 52], [12, 26, 27, 41, 10... | [2, 2] |

    As you can see from the printout the CSV file has 5 columns:

    | Column Name   | Column Description |
    |---------------|--------------------|
    |image_id       | Defines an image ID which later will identifier this image in TFRecords |
    |file_name      | Defines an image file name (note the file will be automatically looked up in the images directory) |
    |bboxes         | Defines a list of bounding boxes, for all objects annotated in the corresponding image |
    |segmentations  | Defines a list of segmentations in the form of polygons, for all objects annotated in the corresponding image |
    |category_ids   | Defines a list of category IDs, for all objects annotated in the corresponding image |

    You can get more information about each category ID from the `categories.json` file.

    ```json
    {
        "categories": [
            {
                "id": 1,
                "name": "rectangle",
                "supercategory": "shape"
            },
            {
                "id": 2,
                "name": "triangle",
                "supercategory": "shape"
            }
        ]
    }
    ```

    For example, is a CSV file record has the following list of category IDs `[2,1]`. It means that the first annotated object is the `triangle` (since its ID is `2`), and the second is the `rectangle` (since its ID is `1` respectively).

=== "COCO"
    This is probably the most popular format for generating synthetic (or storing real) CV data. The synthetic data generated in this format will help you to debug your code and prepare it to use on real data.

    If you leave all function arguments to default, it will create the following files structure (under `dataset/synthetic`) in the current folder (from which Python-code or command-line has been executed).

    ```text
    dataset/synthetic/
        annotations/
            instances_test.json
            instances_train.json
            instances_val.json
        instances_test/
        instances_train/
        instances_val/
    ```

    | Artifact                         | Type   | Comment |
    |----------------------------------|--------|---------|
    | annotations                      | Dir    | Contains files describing annotations |
    | annotations/instances_train.json | File   | Contains training records including image and annotations |
    | annotations/instances_val.json   | File   | Contains validation records including image and annotations |
    | annotations/instances_test.json  | File   | Contains test records including image and annotations |
    | instances_train                  | Dir    | Contains all generated training images (annotated in instances_train.json) |
    | instances_val                    | Dir    | Contains all generated validation images (annotated in instances_val.json) |
    | instances_test                   | Dir    | Contains all generated test images (annotated in instances_test.json) |

    As an example let's review the structure of the `instances_train.json` file. Please note, the structure of the other two JSON files is the same. The top-level of the COCO JSON structure contains the five items: `info`, `licenses`, `images`, `categories`, `annotations`. The following three are the most important for the current topic: `images`, `annotations`, and `categories`. Let's review them based on an image record.

    `images` property contains information about images.

    ```json
    images: [
        ...,
        {
            "file_name": "image1.jpg",
            "coco_url": "file:///dataset/synthetic/instances_train/image1.jpg",
            "width": 64,
            "height": 64,
            "date_captured": "2022-01-13 15:00:33",
            "id": 1
        },
        ...
    ]
    ```

    The most relevant properties are `file_name` and `id`. The transformer will read this image file from the folder derived from the stem name of the JSON file. For example, if this file is `instances_train.json`, the image is expected to be located in the `.../instances_train` folder. The full path to the image is expected to be `dataset/synthetic/instances_train/image1.jpg`

    `annotations` property contains a list of bounding boxes, segmentations, and category IDs associated with images.

    ```json
    annotations: [
        ...,
        {
            "segmentation": [
                [
                    51,
                    52,
                    53,
                    58,
                    41,
                    58
                ]
            ],
            "area": 36.0,
            "iscrowd": 0,
            "image_id": 1,
            "bbox": [
                41,
                52,
                12,
                6
            ],
            "category_id": 2,
            "id": 2
        },
        ...
    ]
    ```

    An annotation record contains information about the image it belongs to via `image_id`, bounding box coordinates `bbox`, segmentation polygon coordinates `segmentation`, and `category_id` to define what type of an object is segmented. The identifier of the annotation itself is defined by the `id` property.

    `categories` property contains a list of available categories and information about them.

    ```json
    categories: [
        ...,
        {
            "supercategory": "shape",
            "id": 1,
            "name": "rectangle"
        },
        {
            "supercategory": "shape",
            "id": 2,
            "name": "triangle"
        },
        ...
    ]
    ```

    The category `id` is related to the `category_id` property defined in annotation and tide together annotation and category information.

## PyDoc

::: squids.dataset.maker
    selection:
      members:
        - create_csv_dataset
        - create_coco_dataset
