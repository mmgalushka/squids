"""A module for creating synthetic datasets."""

import copy
import json
import random
import datetime
from pathlib import Path
from shutil import rmtree

from tqdm import tqdm

from .image import create_synthetic_image
from .shape import Ellipse, Triangle, Rectangle
from .palette import Palette
from .background import Background

from ..config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CAPACITY,
    ADDING_SHAPES,
    NOISE_LEVEL,
    BLUR_LEVEL_X,
    BLUR_LEVEL_Y,
    DATASET_DIR,
    DATASET_SIZE,
)

SHAPES_CATEGORIES = [
    {
        "supercategory": "shape",
        "id": Ellipse.category_id,
        "name": Ellipse.category_name,
    },
    {
        "supercategory": "shape",
        "id": Triangle.category_id,
        "name": Triangle.category_name,
    },
    {
        "supercategory": "shape",
        "id": Rectangle.category_id,
        "name": Rectangle.category_name,
    },
]
"""The common shapes categories."""


def create_dataset(
    dataset_dir: str = DATASET_DIR,
    dataset_size: int = DATASET_SIZE,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    image_palette: Palette = Palette.default(),
    image_background: Background = Background.default(),
    image_capacity: int = IMAGE_CAPACITY,
    adding_shapes: str = ADDING_SHAPES,
    noise_level: float = NOISE_LEVEL,
    blur_level_x: float = BLUR_LEVEL_X,
    blur_level_y: float = BLUR_LEVEL_Y,
    coco: bool = False,
    random_state: int = None,
    verbose: bool = False,
):
    """
    This function generates dataset in CSV or COCO format.

    The input parameters for generating the dataset in CSV or COCO format
    are the same except for the `coco` flag. If the `coco` flag is `False`
    the dataset generating in CSV format, and if it is `True` the dataset
    generating in COCO format respectively.

    Args:
        dataset_dir (str):
            The directory where to store the generated data.
        dataset_size (int):
            The number of generated samples.
        image_width (int):
            The image width in pixels.
        image_height (int):
            The image height in pixels.
        image_palette (Palette):
            The palette for generating images.
        image_background (Background):
            The background for generating images.
        image_capacity (int):
            The number of geometrical shapes per image.
        adding_shapes (str):
            The collection of adding shapes to the generated images, it can
            include 'r' for rectangles, 't' for triangles, 'e' for ellipses
            and also a combination of all these parameters ex. 'rt' for
            generating rectangles and triangles.
        noise_level (float):
            The noise level defined in a range between 0 and 1. 0 - indicates
            no noise, any value between 0 and 1 define level of generated
            noise and value 1 - instruct randomly select noise level in range
            [0,1].
        coco (bool):
            The flag defines a type of generating dataset. If `False`
            to generate the CSV dataset, if `True` the COCO dataset
            respectively.
        random_state (int):
            The random state used for generating synthetic data. If it is
            `None`  then every time synthetic data will be generated with
            a new random value.
        verbose (bool):
            The flag to set verbose mode.
    """
    if coco:
        create_coco_dataset(
            dataset_dir,
            dataset_size,
            image_width,
            image_height,
            image_palette,
            image_background,
            image_capacity,
            adding_shapes,
            noise_level,
            blur_level_x,
            blur_level_y,
            random_state,
            verbose,
        )
    else:
        create_csv_dataset(
            dataset_dir,
            dataset_size,
            image_width,
            image_height,
            image_palette,
            image_background,
            image_capacity,
            adding_shapes,
            noise_level,
            blur_level_x,
            blur_level_y,
            random_state,
            verbose,
        )


def create_csv_dataset(
    dataset_dir: str,
    dataset_size: int,
    image_width: int,
    image_height: int,
    image_palette: Palette,
    image_background: Background,
    image_capacity: int,
    adding_shapes: str,
    noise_level: float,
    blur_level_x: float,
    blur_level_y: float,
    random_state: int,
    verbose: bool,
):
    """Generates CSV dataset.

    Args:
        dataset_dir (str):
            The directory where generated data are stored.
        dataset_size (int):
            The number of generated samples.
        image_width (int):
            The image width in pixels.
        image_height (int):
            The image height in pixels.
        image_palette (Palette):
            The palette for generating images.
        image_background (Background):
            The palette for generating images.
        image_capacity (int):
            The number of geometrical shapes per image.
        adding_shapes (str):
            The collection of adding shapes to the generated images, it can
            include 'r' for rectangles, 't' for triangles, 'e' for ellipses
            and also a combination of all these parameters ex. 'rt' for
            generating rectangles and triangles.
        noise_level (float):
            The noise level defined in a range between 0 and 1. 0 - indicates
            no noise, any value between 0 and 1 define level of generated
            noise and value 1 - instruct randomly select noise level in range
            [0,1].
        random_state (int):
            The random state used for generating synthetic data.
        verbose (bool):
            The flag to set verbose mode.
    """
    if random_state is not None:
        random.seed(random_state)

    output_dir = Path(dataset_dir)
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True)

    images_dir = output_dir / "images"
    images_dir.mkdir()

    train_file = Path(output_dir / "instances_train.csv")
    val_file = Path(output_dir / "instances_val.csv")
    test_file = Path(output_dir / "instances_test.csv")
    categories_file = Path(output_dir / "categories.json")

    with open(categories_file, "w") as categories_fp:

        json.dump(
            {"categories": SHAPES_CATEGORIES},
            categories_fp,
            sort_keys=True,
            indent=4,
        )

    with open(train_file, "w") as train, open(val_file, "w") as val, open(
        test_file, "w"
    ) as test:

        header = "image_id,file_name,bboxes,segmentations,category_ids\n"
        train.write(header)
        val.write(header)
        test.write(header)

        if verbose:
            pbar = tqdm(range(dataset_size))

        for image_id in range(dataset_size):
            file_name = f"image{image_id}.jpg"
            image, shapes = create_synthetic_image(
                image_width,
                image_height,
                image_palette,
                image_background,
                image_capacity,
                adding_shapes,
                noise_level,
                blur_level_x,
                blur_level_y,
            )
            image.save(
                images_dir / file_name, "JPEG", quality=100, subsampling=0
            )

            bboxes = []
            segmentations = []
            category_ids = []
            for shape in shapes:
                bboxes.append(shape.bbox.flatten())
                segmentations.append(shape.polygon.flatten())
                category_ids.append(shape.category_id)

            record = ",".join(
                [
                    f"{image_id}",
                    f"{file_name}",
                    f'"{bboxes}"',
                    f'"{segmentations}"',
                    f'"{category_ids}"\n',
                ]
            )

            partition = random.uniform(0, 1)
            if partition <= 0.7:
                train.write(record)
            elif partition <= 0.9:
                val.write(record)
            else:
                test.write(record)

            if verbose:
                pbar.update(1)


def create_coco_dataset(
    dataset_dir: str,
    dataset_size: int,
    image_width: int,
    image_height: int,
    image_palette: Palette,
    image_background: Background,
    image_capacity: int,
    adding_shapes: str,
    noise_level: float,
    blur_level_x: float,
    blur_level_y: float,
    random_state: int,
    verbose: bool,
):
    """Generates COCO dataset.

    Args:
        dataset_dir (str):
            The directory where generated data are stored.
        dataset_size (int):
            The number of generated samples.
        image_width (int):
            The image width in pixels.
        image_height (int):
            The image height in pixels.
        image_palette (Palette):
            The palette for generating images.
        image_background (Background):
            The palette for generating images.
        image_capacity (int):
            The number of geometrical shapes per image.
        adding_shapes (str):
            The collection of adding shapes to the generated images, it can
            include 'r' for rectangles, 't' for triangles, 'e' for ellipses
            and also a combination of all these parameters ex. 'rt' for
            generating rectangles and triangles.
        noise_level (float):
            The noise level defined in a range between 0 and 1. 0 - indicates
            no noise, any value between 0 and 1 define level of generated
            noise and value 1 - instruct randomly select noise level in range
            [0,1].
        blur_level_x (float):
            The level of blurring to add on X-axis in a range between 0 and 1.
            0 - indicates no blurring (all shapes with maximum sharpness), any
            value between 0 and 1 define level of added blurring and value
            1 - instruct randomly select blurring level in range [0,1].
        blur_level_y (float):
            The level of blurring to add on Y-axis in a range between 0 and 1.
            0 - indicates no blurring (all shapes with maximum sharpness), any
            value between 0 and 1 define level of added blurring and value
            1 - instruct randomly select blurring level in range [0,1].
        random_state (int):
            The random state used for generating synthetic data.
        verbose (bool):
            The flag to set verbose mode.
    """
    if random_state is not None:
        random.seed(random_state)

    output_dir = Path(dataset_dir)
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True)

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir()
    train_images_dir = output_dir / "instances_train"
    train_images_dir.mkdir()
    val_images_dir = output_dir / "instances_val"
    val_images_dir.mkdir()
    test_images_dir = output_dir / "instances_test"
    test_images_dir.mkdir()

    now = datetime.datetime.now()
    template = {
        "info": {
            "description": f"COCO {now.year} Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": int(now.year),
            "contributor": "COCO Consortium",
            "date_created": now.strftime("%Y-%m-%d"),
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 0,
                "name": "Attribution-NonCommercial-ShareAlike License",
            }
        ],
        "images": [],
        "categories": SHAPES_CATEGORIES,
        "annotations": [],
    }

    train = copy.deepcopy(template)
    val = copy.deepcopy(template)
    test = copy.deepcopy(template)

    if verbose:
        pbar = tqdm(range(dataset_size))

    annotation_id = 0
    for image_id in range(dataset_size):
        image_file = f"image{image_id}.jpg"
        image, shapes = create_synthetic_image(
            image_width,
            image_height,
            image_palette,
            image_background,
            image_capacity,
            adding_shapes,
            noise_level,
            blur_level_x,
            blur_level_y,
        )

        partition = random.uniform(0, 1)
        if partition <= 0.7:
            file_name = train_images_dir / image_file
        elif partition <= 0.9:
            file_name = val_images_dir / image_file
        else:
            file_name = test_images_dir / image_file
        image.save(file_name, "JPEG", quality=100, subsampling=0)

        image_record = {
            "file_name": str(image_file),
            "coco_url": f"file:///{str(file_name)}",
            "width": int(image_width),
            "height": int(image_height),
            "date_captured": now.strftime("%Y-%m-%d %H:%M:%S"),
            "id": image_id,
        }

        annotation_records = []
        for shape in shapes:
            annotation_records.append(
                {
                    "segmentation": [
                        [int(v) for v in shape.polygon.flatten()]
                    ],
                    "area": float(shape.get_area()),
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [int(v) for v in shape.bbox.flatten()],
                    "category_id": shape.category_id,
                    "id": annotation_id,
                }
            )
            annotation_id += 1

        if partition <= 0.7:
            train["images"].append(image_record)
            train["annotations"].extend(annotation_records)
        elif partition <= 0.9:
            val["images"].append(image_record)
            val["annotations"].extend(annotation_records)
        else:
            test["images"].append(image_record)
            test["annotations"].extend(annotation_records)

        if verbose:
            pbar.update(1)

    train_file = Path(annotations_dir / "instances_train.json")
    val_file = Path(annotations_dir / "instances_val.json")
    test_file = Path(annotations_dir / "instances_test.json")

    with open(train_file, "w") as fp:
        json.dump(train, fp, indent=4)
    with open(val_file, "w") as fp:
        json.dump(val, fp, indent=4)
    with open(test_file, "w") as fp:
        json.dump(test, fp, indent=4)
