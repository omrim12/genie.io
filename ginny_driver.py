import os
import sys

from termcolor import cprint
from CLI import CLI
from cnn_utils import cnn_train
from keras.models import save_model, load_model
from constants import (
    GINNY_NET_FNAME
)


def main(args):
    # Initialize CNN model if trained exists
    if os.path.exists(GINNY_NET_FNAME):
        if len(args) > 0:
            if args[0] == 'new-ginny':
                # Train a new CNN model
                ginny_model = cnn_train()

                # Delete previous CNN model
                os.remove(GINNY_NET_FNAME)

                # Save new CNN model
                # Save CNN model locally
                save_model(
                    ginny_model,
                    GINNY_NET_FNAME,
                    overwrite=True,
                    include_optimizer=True,
                    save_format=None,
                    signatures=None,
                    options=None,
                    save_traces=True
                )
                # TODO: convert to a tflite model

            else:
                cprint('ERROR: invalid args given; aborting.', 'red')
                exit(1)
        else:
            ginny_model = load_model(GINNY_NET_FNAME)
    else:
        # Train a new CNN model
        ginny_model = cnn_train()

        # Save CNN model locally
        save_model(
            ginny_model,
            GINNY_NET_FNAME,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None,
            save_traces=True
        )
        # TODO: convert to a tflite model

    # Run a CLI session based on the trained model
    cli = CLI(ginny_model)
    cli.session()


if __name__ == '__main__':
    main(sys.argv[1:])
