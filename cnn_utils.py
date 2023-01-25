import logging
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from termcolor import cprint
from data_utils import load_food_101
from tag_utils import get_labels_list
from constants import (
    IMAGE_SIZE,
    FOOD_TYPES,
    DATASET_BATCH
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def classify_client_input(image_array: np.array, cnn_model) -> str:
    # pass image in cnn_model to predict compatible food item
    food_predict = cnn_model.predict(np.array([image_array]))

    # Return corresponding food item
    return get_labels_list()[np.argmax(food_predict)]


def train_by_type(
        X_train, y_train,
        X_test, y_test,
        conv_type,
        epoch=5,
        batch_size=64,
        layer_act='relu',
        output_act='softmax'
):
    # Init CNN inputs tensor
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    # Run convolution & max-pooling
    X = layers.Conv2D(filters=64, kernel_size=(4, 4), activation=layer_act, padding=conv_type)(inputs)
    X = layers.MaxPooling2D(pool_size=2)(X)
    X = layers.Conv2D(filters=128, kernel_size=(4, 4), activation=layer_act, padding=conv_type)(X)
    X = layers.MaxPooling2D(pool_size=2)(X)
    X = layers.Conv2D(filters=256, kernel_size=(4, 4), activation=layer_act, padding=conv_type)(X)

    # Flatten output layer (a KerasTensor) from CNN to pass output data
    # to NN output dense layer (knows only to receive a vector)
    X = layers.Flatten()(X)

    # Calculating outputs from NN dense layer
    outputs = layers.Dense(FOOD_TYPES, activation=output_act)(X)

    # Initialize model derived from CNN results
    # TODO: consider adding regularization param
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Summarize results from CNN
    model.summary()

    # Running NN simulation on trained model derived from CNN
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size)

    # Computing loss & accuracy over the entire test set
    test_loss, test_acc = model.evaluate(X_test, y_test)

    return model, test_loss, test_acc


def cnn_train():
    cprint('\nAbout to train a new genie!', "blue")

    # load train + test datasets
    X_train, X_test, y_train, y_test = load_food_101()

    # a. Training CNN "same" using the food 101 dataset
    LOGGER.info("--- Running CNN 'same' learning session ---")
    same_model, same_loss, same_acc = train_by_type(X_train, y_train,
                                                    X_test, y_test,
                                                    conv_type='same',
                                                    epoch=70)

    # Training CNN "valid" using the food 101 dataset
    LOGGER.info("--- Running CNN 'valid' learning session ---")
    valid_model, valid_loss, valid_acc = train_by_type(X_train, y_train,
                                                       X_test, y_test,
                                                       conv_type='valid')

    # Determine best model by model accuracy
    if valid_acc > same_acc:
        return valid_model

    return same_model
