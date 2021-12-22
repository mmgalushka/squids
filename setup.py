"""
SquiDS setup script.
"""

import setuptools

from deeptrace import __version__


def get_long_description():
    """Reads the long project description from the 'README.md' file."""
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


setuptools.setup(
    name="squids",
    version=__version__,
    author="Mykola Galushka",
    author_email="mm.galushka@gmail.com",
    description="The synthetic dataset generator for Computer Vision tasks.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/mmgalushka/squids",
    project_urls={
        "Bug Tracker": "https://github.com/mmgalushka/squids/issues",
    },
    classifiers=[
        'Intended Audience :: Developers',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where=".", exclude=["tests"]),
    python_requires=">=3.6",
)
