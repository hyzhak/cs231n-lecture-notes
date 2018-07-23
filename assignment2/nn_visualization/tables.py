import pandas as pd
from IPython.display import display, HTML

from nn_visualization.models import get_model_name

allowed = [
    'num_of_trainable',
    'time',
    'total_correct',
    'total_lost',
]

def show_tables(models):
    for m in models:
        print('model:', get_model_name(m))
        display(pd.DataFrame.from_dict(m['res']).filter(items=allowed, axis=0))
