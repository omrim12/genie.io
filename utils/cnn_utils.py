import logging
import numpy as np
from keras import layers
from termcolor import cprint
from tensorflow import keras
from utils.data_utils import load_food_101
from utils.tag_utils import get_labels_list
from constants import (
    IMAGE_SIZE,
    FOOD_TYPES,
    CONV_TYPE,
    EPOCH,
    KERNEL_SIZE,
    FILTERS,
    DROPOUT_PROB,
    ALPHA
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
        epoch=20,
        batch_size=128,
        layer_act='relu',
        output_act='softmax'
):
    # Init CNN inputs tensor
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    # Run convolution & max-pooling
    X = layers.Conv2D(
        filters=FILTERS,
        kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
        activation=layer_act, padding=conv_type)(inputs)
    X = layers.MaxPooling2D(pool_size=2)(X)
    X = layers.Conv2D(
        filters=FILTERS * 2,
        kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
        activation=layer_act, padding=conv_type)(X)
    X = layers.MaxPooling2D(pool_size=2)(X)
    X = layers.Conv2D(
        filters=FILTERS * 4,
        kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
        activation=layer_act, padding=conv_type)(X)

    # Flatten output layer (a KerasTensor) from CNN to pass output data
    # to NN output dense layer (knows only to receive a vector)
    X = layers.Flatten()(X)

    # Define a dropout layer to prevent over-fitting
    # X = layers.Dropout(DROPOUT_PROB)(X)

    # Calculating outputs from NN dense layer
    outputs = layers.Dense(
        FOOD_TYPES,
        activation=output_act
    )(X)

    # Initialize model from feature maps results
    # and other defined layers
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Summarize results from CNN
    model.summary()

    # Training CNN model based on optimization techniques
    opt = keras.optimizers.RMSprop(learning_rate=ALPHA)
    model.compile(optimizer=opt,
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

    # Training CNN using the food 101 dataset
    LOGGER.info(f"--- Running CNN '{CONV_TYPE}' learning session ---")
    model, model_loss, model_acc = train_by_type(X_train, y_train,
                                                 X_test, y_test,
                                                 conv_type=CONV_TYPE,
                                                 epoch=EPOCH)
    return model, model_loss, model_acc
