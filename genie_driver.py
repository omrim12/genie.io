import logging

import os
import sys
from CLI import CLI
from flask import Flask
from flask_restful import Api
from utils.cnn_utils import cnn_train
from utils.file_utils import save_genie, load_genie
from utils.rest_api_utils import GenieModelResource
from constants import (
    GENIE_NET_FNAME,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
api = Api(app)


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
        if '--api-mode' in args:
            LOGGER.error("Cannot run both CLI and API modes.")
            exit(1)
        cli = CLI(genie_model)
        cli.session()

    elif '--api-mode' in args:
        api.add_resource(GenieModelResource, '/classify')
        app.run(debug=True)


if __name__ == '__main__':
    main(sys.argv[1:])
