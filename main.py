"""
The SquiDS main module.
"""

import sys
import argparse

import squids as sds

# ---------------------------------
# Initializes application commands
# ---------------------------------
parser = argparse.ArgumentParser(prog="./helper.sh", usage="%(prog)s")
subparsers = parser.add_subparsers()
sds.generate(subparsers)
sds.transform(subparsers)
sds.explore(subparsers)


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "help":
            parser.print_help()
        else:
            args = parser.parse_args(sys.argv[1:])
            try:
                args.func(args)
            except sds.TFRecordsError as err:
                print(f"Error: {err}")
                exit(1)

    else:
        parser.print_help()
    exit(0)


if __name__ == "__main__":
    main()
