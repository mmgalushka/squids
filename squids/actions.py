"""
A module for handling user actions.
"""

from .image import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CAPACITY,
    Palette,
    Background,
)
from .dataset import (
    DATASET_DIR,
    DATASET_SIZE,
    create_csv_dataset,
    create_coco_dataset,
)
from .tfrecords import create_tfrecords, inspect_tfrecords, view_tfrecords


def generate(subparsers):
    # Defines the command name.
    cmd = "generate"

    def run(args):
        if args.coco:
            print(f"\nGenerate COCO dataset in '{args.output}'...")
            create_coco_dataset(
                dataset_dir=args.output,
                dataset_size=args.size,
                image_width=args.image_width,
                image_height=args.image_height,
                image_palette=args.image_palette,
                image_background=args.image_background,
                image_capacity=args.image_capacity,
                verbose=args.verbose,
            )
        else:
            print(f"\nGenerate CSV dataset in '{args.output}'...")
            create_csv_dataset(
                dataset_dir=args.output,
                dataset_size=args.size,
                image_width=args.image_width,
                image_height=args.image_height,
                image_palette=args.image_palette,
                image_background=args.image_background,
                image_capacity=args.image_capacity,
                verbose=args.verbose,
            )

    # ---------------------------------
    # Sets "generate" command options
    # ---------------------------------
    parser = subparsers.add_parser(cmd)
    parser.set_defaults(func=run)

    # --- I/O options -----------------
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        type=str,
        default=DATASET_DIR,
        help=f'an output directory (default="{DATASET_DIR}")',
    )
    parser.add_argument(
        "-s",
        "--size",
        metavar="NUMBER",
        type=int,
        default=DATASET_SIZE,
        help=f"a number of generated data samples (default={DATASET_SIZE})",
    )
    parser.add_argument(
        "--coco",
        dest="coco",
        action="store_true",
        help="a flag to generate dataset using the COCO format.",
    )

    # --- image options ---------------
    parser.add_argument(
        "--image-width",
        metavar="PIXELS",
        type=int,
        default=IMAGE_WIDTH,
        help=f"a generated image width (default={IMAGE_WIDTH})",
    )
    parser.add_argument(
        "--image-height",
        metavar="PIXELS",
        type=int,
        default=IMAGE_HEIGHT,
        help=f"a generated image height (default={IMAGE_HEIGHT})",
    )
    parser.add_argument(
        "--image-palette",
        choices=Palette.values(),
        type=str,
        default=Palette.default(),
        help=f'an image palette (default="{Palette.default()}")',
    )
    parser.add_argument(
        "--image-background",
        choices=Background.values(),
        type=str,
        default=Background.default(),
        help=f'an image background (default="{Background.default()}")',
    )
    parser.add_argument(
        "--image-capacity",
        metavar="NUMBER",
        type=int,
        default=IMAGE_CAPACITY,
        help=f"a number of shapes per image (default={IMAGE_CAPACITY})",
    )

    # --- system options --------------
    parser.add_argument(
        "-v",
        "--verbose",
        help="the flag to set verbose mode",
        action="store_true",
    )


def transform(subparsers):
    # Defines the command name.
    cmd = "transform"

    def run(args):
        print(
            f"\nTransform dataset from '{args.input}' "
            f"to tfrecords in '{args.output}'..."
        )
        create_tfrecords(
            dataset_dir=args.input,
            selected_categories=args.select_categories,
            tfrecords_dir=args.output,
            tfrecords_size=args.size,
            image_width=args.image_width,
            image_height=args.image_height,
            verbose=args.verbose,
        )

    # ---------------------------------
    # Sets "transform" command options
    # ---------------------------------
    parser = subparsers.add_parser(cmd)
    parser.set_defaults(func=run)

    # --- input options ---------------
    parser.add_argument(
        "-i",
        "--input",
        metavar="DIR",
        type=str,
        default=DATASET_DIR,
        help=f'an input directory with source data (default="{DATASET_DIR}")',
    )

    # --- output options --------------
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        type=str,
        default=None,
        help="an output directory for TFRecords (default=None)",
    )
    parser.add_argument(
        "-s",
        "--size",
        metavar="NUMBER",
        type=int,
        default=256,
        help="a number of records per partion (default=256)",
    )

    # --- image options ---------------
    parser.add_argument(
        "--image-width",
        metavar="PIXELS",
        type=int,
        default=IMAGE_WIDTH,
        help="an image width resize to (default=None)",
    )
    parser.add_argument(
        "--image-height",
        metavar="PIXELS",
        type=int,
        default=IMAGE_HEIGHT,
        help="an image height resize to (default=None)",
    )

    # --- selection options -----------
    parser.add_argument(
        "--select-categories",
        metavar="CATEGORY_IDS",
        nargs="+",
        type=int,
        help="a list of selected category IDs",
        default=[],
    )

    # --- system options --------------
    parser.add_argument(
        "-v",
        "--verbose",
        help="the flag to set verbose mode",
        action="store_true",
    )


def inspect(subparsers):
    # Defines the command name.
    cmd = "inspect"

    def run(args):
        print(f"\nInspect tfrecords from '{args.input}'...")
        inspect_tfrecords(tfrecords_dir=args.input)

    # ---------------------------------
    # Sets "inspect" command options
    # ---------------------------------
    parser = subparsers.add_parser(cmd)
    parser.set_defaults(func=run)

    # --- input options ---------------
    parser.add_argument(
        "-i",
        "--input",
        metavar="DIR",
        type=str,
        help="an input directory with tfrecords",
    )


def view(subparsers):
    # Defines the command name.
    cmd = "view"

    def run(args):
        print(f"\nView '{args.image_id}' tfrecords from '{args.input}'...")
        view_tfrecords(
            tfrecords_dir=args.input,
            image_id=args.image_id,
            with_bboxes=args.no_bboxes,
            with_segmentations=args.no_segmentations,
        )

    # ---------------------------------
    # Sets "view" command options
    # ---------------------------------
    parser = subparsers.add_parser(cmd)
    parser.set_defaults(func=run)

    # --- input options ---------------
    parser.add_argument(
        "input",
        metavar="DIR",
        type=str,
        help="an input directory with tfrecords",
    )

    parser.add_argument(
        "image_id",
        metavar="IMAGE_ID",
        type=int,
        help="an image ID to view",
    )

    parser.add_argument(
        "--no-bboxes",
        help="turn off the showing of bounding boxes",
        action="store_false",
    )

    parser.add_argument(
        "--no-segmentations",
        help="turn off the showing of segmentations",
        action="store_false",
    )
