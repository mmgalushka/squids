"""
A module for handling user actions.
"""

from .config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CAPACITY,
    DATASET_DIR,
    DATASET_SIZE,
    TFRECORDS_SIZE,
)

from .dataset import (
    create_dataset,
    Background,
    Palette,
)
from .tfrecords import create_tfrecords, explore_tfrecords


def generate(subparsers):
    # Defines the command name.
    cmd = "generate"

    def run(args):
        create_dataset(
            dataset_dir=args.dataset_dir,
            dataset_size=args.dataset_size,
            image_width=args.image_width,
            image_height=args.image_height,
            image_palette=args.image_palette,
            image_background=args.image_background,
            image_capacity=args.image_capacity,
            coco=args.coco,
            verbose=args.verbose,
        )

    # ---------------------------------
    # Sets "generate" command options
    # ---------------------------------
    parser = subparsers.add_parser(cmd)
    parser.set_defaults(func=run)

    # --- I/O options -----------------
    parser.add_argument(
        "dataset_dir",
        metavar="DATASET_DIR",
        nargs="?",
        type=str,
        default=DATASET_DIR,
        help=(f"a generating dataset directory, (default '{DATASET_DIR}')"),
    )
    parser.add_argument(
        "-s",
        "--dataset-size",
        metavar="NUMBER",
        type=int,
        default=DATASET_SIZE,
        help=f"a number of generated data samples (default={DATASET_SIZE})",
    )
    parser.add_argument(
        "--coco",
        dest="coco",
        action="store_true",
        help="a flag to generate dataset in the COCO format",
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
        help=f"a used image palette (default='{Palette.default()}')",
    )
    parser.add_argument(
        "--image-background",
        choices=Background.values(),
        type=str,
        default=Background.default(),
        help=f"a used image background (default='{Background.default()}')",
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
        help="a flag to set verbose mode",
        action="store_true",
    )


def transform(subparsers):
    # Defines the command name.
    cmd = "transform"

    def run(args):
        create_tfrecords(
            dataset_dir=args.dataset_dir,
            tfrecords_dir=args.tfrecords_dir,
            size=args.size,
            image_width=args.image_width,
            image_height=args.image_height,
            selected_categories=args.select_categories,
            verbose=args.verbose,
        )

    # ---------------------------------
    # Sets "transform" command options
    # ---------------------------------
    parser = subparsers.add_parser(cmd)
    parser.set_defaults(func=run)

    # --- input options ---------------
    parser.add_argument(
        "dataset_dir",
        metavar="DATASET_DIR",
        nargs="?",
        type=str,
        default=DATASET_DIR,
        help=(
            "a source dataset directory, if not defined, "
            f"it will be selected as the '{DATASET_DIR}'"
        ),
    )

    # --- output options --------------
    parser.add_argument(
        "tfrecords_dir",
        metavar="TFRECORDS_DIR",
        nargs="?",
        type=str,
        default=None,
        help=(
            "a TFRecords directory, if not defined, "
            "it will be created in the <DATASET_DIR> parent "
            "under the name '<DATASET_DIR>-tfrecords'"
        ),
    )

    parser.add_argument(
        "-s",
        "--size",
        metavar="NUMBER",
        type=int,
        default=TFRECORDS_SIZE,
        help=f"a number of images per partition (default={TFRECORDS_SIZE})",
    )

    # --- image options ---------------
    parser.add_argument(
        "--image-width",
        metavar="PIXELS",
        type=int,
        default=IMAGE_WIDTH,
        help=f"a TFRecords image width resize to (default={IMAGE_WIDTH})",
    )
    parser.add_argument(
        "--image-height",
        metavar="PIXELS",
        type=int,
        default=IMAGE_HEIGHT,
        help=f"a TFRecords image height resize to (default={IMAGE_HEIGHT})",
    )

    # --- selection options -----------
    parser.add_argument(
        "--select-categories",
        metavar="CATEGORY_ID",
        nargs="+",
        type=int,
        help="a list of selected category IDs",
        default=[],
    )

    # --- system options --------------
    parser.add_argument(
        "-v",
        "--verbose",
        help="a flag to set verbose mode",
        action="store_true",
    )


def explore(subparsers):
    # Defines the command name.
    cmd = "explore"

    def run(args):
        try:
            explore_tfrecords(
                tfrecords_dir=args.tfrecords_dir,
                image_id=args.image_id,
                output_dir=args.output_dir,
                with_summary=not args.no_summary,
                with_bboxes=not args.no_bboxes,
                with_segmentations=not args.no_segmentations,
            )
        except FileNotFoundError as err:
            print("\nI/O Error:", err)

    # ---------------------------------
    # Sets "explore" command options
    # ---------------------------------
    parser = subparsers.add_parser(cmd)
    parser.set_defaults(func=run)

    # --- input options ---------------
    parser.add_argument(
        "tfrecords_dir",
        metavar="TFRECORDS_DIR",
        type=str,
        help="a TFRecords directory to explore",
    )

    group = parser.add_argument_group("A record exploration options")

    group.add_argument(
        "image_id",
        metavar="IMAGE_ID",
        nargs="?",
        type=int,
        help="an image ID to select",
        default=None,
    )

    group.add_argument(
        "output_dir",
        metavar="OUTPUT_DIR",
        nargs="?",
        type=str,
        help="an output directory to save rendered image",
        default=".",
    )

    group.add_argument(
        "--no-summary",
        help="turn off the showing of image summary",
        action="store_true",
    )

    group.add_argument(
        "--no-bboxes",
        help="turn off the showing of bounding boxes",
        action="store_true",
    )

    group.add_argument(
        "--no-segmentations",
        help="turn off the showing of segmentations",
        action="store_true",
    )
