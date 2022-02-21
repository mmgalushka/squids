"""The helper functions for the notebook."""

import matplotlib.pyplot as plt


def plot_images(images, title=None):
    """Shows images in the 5x5 grid.

    Args:
        images (list):
            The list of images to show.
        title (str):
            The figure title.
    """
    assert (
        len(images) == 25
    ), "Invalid number of plotting images, it must be 25"
    fig, ax = plt.subplots(5, 5)
    fig.set_size_inches(18, 10)
    sample = 0
    for i in range(5):
        for j in range(5):
            ax[i, j].imshow(images[sample])
            sample += 1
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
