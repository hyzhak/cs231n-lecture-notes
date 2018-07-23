# import tensorflow as tf
# import numpy as np
# import math
# import time
# import timeit
import matplotlib.pyplot as plt
from IPython.display import display, HTML

from nn_visualization.models import get_model_name


def show_graphs(models, ymax=5):
    """
    TODO:
    - show 2 graphs: training and validation
    - show accuracy
    """

    # plt.gcf().set_size_inches(15, 12)

    # training dynamic
    plt.grid(True)
    plt.title('Training Dynamics')
    plt.xlabel('minibatch iteration')
    plt.ylabel('minibatch loss')
    # plt.xlim(xmin=50)
    plt.ylim(ymax=ymax)

    for m in models:
        name = get_model_name(m)
        plt.plot(m['res']['training']['losses'], label=name)

    plt.legend()
    plt.show()
