"""
The SquiDS main module.
"""

import os

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


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "help":
            parser.print_help()
        else:
            args = parser.parse_args(sys.argv[1:])
            args.func(args)
    else:
        parser.print_help()
    exit(os.EX_OK)


if __name__ == "__main__":
    main()
