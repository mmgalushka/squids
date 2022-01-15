# Welcome to SquiDS

The main goal of this project is to help data scientists working with Computer Vision (CV) domain better manage synthetic and real training data. This project will be useful in building machine learning (ML) models for:

* image classification;
* objects detection/localisation;
* objects segmentation.

## Installation

To install SquiDS, run the following command from the command line:

```bash
~$ pip install squids
```

Note that this PIP install is not yet available, since documentation is prepared ahead of time. When this PIP package gets deployed this note will be removed!

## Capability

This project gives you the following capabilities:

* [Generate synthetic dataset in CSV and COCO formats](generator.md);
* [Transform dataset in either format to TFRecords](transformer.md);
* [Explore content of generated TFRecords](explorer.md);
* [Load TFRecords for machine learning model training](loader.md).
