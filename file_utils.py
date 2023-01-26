from keras.models import save_model, load_model
from constants import (
    GENIE_NET_FNAME,
    GENIE_SCORE_FNAME
)

# TODO: convert to a tflite model
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
        score_file.write(f"genie net accuracy: {acc}\n"
                         f"genie net loss: {loss}")

def load_genie():
    return load_model(GENIE_NET_FNAME)
