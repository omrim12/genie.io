import os
from CLI import CLI
from cnn_utils import cnn_train
from keras.models import save_model, load_model
from constants import (
    GINNY_NET_FNAME
)


def main():
    # Initialize CNN model if trained exists
    if os.path.exists(GINNY_NET_FNAME):
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

    # Run a CLI session based on the trained model
    cli = CLI(ginny_model)
    cli.session()


if __name__ == '__main__':
    main()
