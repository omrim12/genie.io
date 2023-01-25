import os
import sys

from termcolor import cprint
from CLI import CLI
from cnn_utils import cnn_train
from keras.models import save_model, load_model
from constants import (
    GENIE_NET_FNAME
)


def main(args):
    # Initialize CNN model if trained exists
    if os.path.exists(GENIE_NET_FNAME):
        if '--new-genie' in args:
            # Train a new CNN model
            genie_model = cnn_train()

            # Delete previous CNN model
            os.remove(GENIE_NET_FNAME)

            # Save new CNN model
            # Save CNN model locally
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
            # TODO: convert to a tflite model
        else:
            genie_model = load_model(GENIE_NET_FNAME)
    else:
        # Train a new CNN model
        genie_model = cnn_train()

        # Save CNN model locally
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
        # TODO: convert to a tflite model

    # Run a CLI session based on the trained model
    if '--cli-mode' in args:
        cli = CLI(genie_model)
        cli.session()


if __name__ == '__main__':
    main(sys.argv[1:])
