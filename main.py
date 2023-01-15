import os
import numpy as np
from CLI import CLI
from cnn_utils import cnn_train
from data_utils import load_food_101
from keras.models import save_model, load_model
from constants import (
    GINNY_NET_FNAME
)

# TODO: fix LOGGER.info output
# TODO: check why there are less tags in train than test dataset


def main():
    X_train, X_test, y_train, y_test = load_food_101()
    print(f'y_test = {np.unique(y_test)}')
    print(f'y_train = {np.unique(y_train)}')

    exit(1)

    """
    
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
    cli = CLI(cnn_model=ginny_model)
    cli.session()
    """


if __name__ == '__main__':
    main()
