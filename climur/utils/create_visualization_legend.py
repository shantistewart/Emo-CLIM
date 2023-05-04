"""Script to create external legends for t-SNE visualizations."""


import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any
from climur.utils.constants import IMAGE2AUDIO_TAG_MAP
from climur.utils.constants import LABEL2COLOR


# plots directory:
PLOTS_DIR = "plots"


"""Adapted from https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib"""
def export_legend(legend: Any, save_path: str, expand: List = [-5, -5, 5, 5]):

    # export legend:
    legend.axes.axis("off")
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    # save to file:
    fig.savefig(save_path, dpi="figure", bbox_inches=bbox)


if __name__ == "__main__":
    print("\n")

    # create magical function:
    func = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]


    # IMAGE DATASET LEGEND:

    # get image dataset emotion labels and colors:
    image_emotion_labels = list(IMAGE2AUDIO_TAG_MAP.keys())
    colors = [LABEL2COLOR[IMAGE2AUDIO_TAG_MAP[label]] for label in list(IMAGE2AUDIO_TAG_MAP.keys())]

    # create legend:
    handles = [func(".", colors[i]) for i in range(len(colors))]
    legend = plt.legend(handles, image_emotion_labels, loc=3, framealpha=1, frameon=True)
    # export legend:
    export_legend(
        legend,
        save_path=os.path.join(PLOTS_DIR, "image_dataset_legend.png"),
        expand=[-2, -2, 2, 2]
    )


    # AUDIO DATASET LEGEND:

    # get audio dataset emotion labels and colors:
    audio_emotion_labels = list(IMAGE2AUDIO_TAG_MAP.values())
    colors = [LABEL2COLOR[label] for label in list(IMAGE2AUDIO_TAG_MAP.values())]

    # create legend:
    handles = [func("x", colors[i]) for i in range(len(colors))]
    legend = plt.legend(handles, audio_emotion_labels, loc=3, framealpha=1, frameon=True)
    # export legend:
    export_legend(
        legend,
        save_path=os.path.join(PLOTS_DIR, "audio_dataset_legend.png"),
        expand=[-2, -2, 2, 2]
    )

