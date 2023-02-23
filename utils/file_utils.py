import os
import json
import tensorflow as tf
from keras.models import save_model, load_model
from constants import (
    GENIE_NET_FNAME,
    GENIE_SCORE_FNAME
)


def save_genie(genie_model, loss, acc):
    save_model(
        genie_model,
        GENIE_NET_FNAME,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True
    )

    # save model loss, accuracy
    with open(GENIE_SCORE_FNAME, 'w') as score_file:
        net_score = {
            'accuracy': acc,
            'loss': loss
        }
        json.dump(net_score, score_file)


def load_genie():
    return load_model(GENIE_NET_FNAME)
