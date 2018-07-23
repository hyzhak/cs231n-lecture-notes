#
# links for inspiration
# - https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
# simple snippet to train and visualize cnn layers of simple model
#

import math
import matplotlib.pyplot as plt
from nn_visualization.models import get_model_name


def draw_layers_of_models(models, **kwargs):
    img_idx = 0
    for m in models:
        img_idx = draw_layers(m, img_idx, **kwargs)


def draw_layers(m, img_idx):
    if 'params' not in m:
        return

    print(f'Model {get_model_name(m)}')
    for (layer_tf, layer) in m['params'].items():
        draw_layer(layer, str(layer_tf.name), img_idx)
        img_idx += layer.shape[3]

    return img_idx


def draw_layer(layer, title, img_idx):
    filters = layer.shape[3]
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(title, size=16)
    for idx in range(filters):
        ax = fig.add_subplot(n_rows, n_columns, idx + 1)
        ax.set_title(f'Filter {idx}')
        img = layer[:, :, :, idx]
        img_min = img.min()
        delta = img.max() - img_min
        img = (img - img_min) / delta
        plt.imshow(img, interpolation='nearest')
    plt.show()
