"""The helper functions for the notebook."""

import matplotlib.pyplot as plt

def plot_images(images, title=None):
    """Shows images in the 3x3 grid.
    
    Args:
        images (list):
            The list of images to show.
        title (str):
            The figure title.
    """
    fig, ax = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            ax[i,j].imshow(images[i * 2 +j])
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()