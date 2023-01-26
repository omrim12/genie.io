import os
import sys
from CLI import CLI
from cnn_utils import cnn_train
from file_utils import save_genie, load_genie
from constants import (
    GENIE_NET_FNAME,
)


def main(args):
    # Initialize CNN model if trained exists
    if os.path.exists(GENIE_NET_FNAME):
        if '--new-genie' in args:
            # Train a new CNN model
            genie_model, loss, acc = cnn_train()

            # Delete previous CNN model
            os.remove(GENIE_NET_FNAME)

            # Save trained CNN model
            save_genie(
                genie_model=genie_model,
                loss=loss,
                acc=acc
            )
        else:
            genie_model = load_genie()
    else:
        # Train a new CNN model
        genie_model, loss, acc = cnn_train()

        # Save trained CNN model
        save_genie(
            genie_model=genie_model,
            loss=loss,
            acc=acc
        )

    # Run a CLI session based on the trained model
    if '--cli-mode' in args:
        cli = CLI(genie_model)
        cli.session()


if __name__ == '__main__':
    main(sys.argv[1:])
