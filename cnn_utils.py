import logging
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from data_utils import load_food_101
from constants import (
    IMAGE_SIZE
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def classify_client_input(image_array: np.array, cnn_model) -> str:
    # TODO: implement this
    return None


def train_by_type(X_train, X_test, y_train, y_test, conv_type):
    # TODO: implement this
    return None, None, None


def cnn_train():
    # load train + test datasets
    X_train, X_test, y_train, y_test = load_food_101()

    # a. Training CNN "same" using the fashion MNIST dataset
    LOGGER.info("--- Running CNN 'same' learning session ---")
    same_model, same_loss, same_acc = train_by_type(X_train, y_train,
                                                    X_test, y_test,
                                                    conv_type='same')

    # Training CNN "valid" using the fashion MNIST dataset
    LOGGER.info("--- Running CNN 'valid' learning session ---")
    valid_model, valid_loss, valid_acc = train_by_type(X_train, y_train,
                                                       X_test, y_test,
                                                       conv_type='valid')

    # Determine best model by model accuracy
    if valid_acc > same_acc:
        return valid_model

    return same_model
