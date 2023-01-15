import logging
import numpy as np
from img_utils import convert_image
from concurrent.futures import ThreadPoolExecutor
from constants import (
    IMAGE_SIZE,
    IMAGES_META,
    DATASET_SIZE
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def load_food_101():
    LOGGER.warning("Loading Food 101 dataset...")
    # init datasets by types
    # running datasets init processes in seperate
    # threads to reduce runtime
    executor = ThreadPoolExecutor()
    res1 = executor.submit(load_dataset, 'train')
    res2 = executor.submit(load_dataset, 'test')
    X_train, y_train = res1.result()
    X_test, y_test = res2.result()

    return X_train, X_test, y_train, y_test


def load_dataset(ds_type: str):
    # init empty dataset
    dataset = np.empty([0, IMAGE_SIZE, IMAGE_SIZE], dtype=float)
    tags = np.empty([0, 1], dtype=int)

    # iterate through all image paths dedicated for
    # dataset type (train/test) and load their image to dataset
    dataset_file = open(f'{IMAGES_META}/{ds_type}.txt')

    # get images paths list
    image_paths_raw = dataset_file.read()
    images_paths_list = image_paths_raw.split('\n')

    for image_path in images_paths_list:
        # extract image food type and id from image_path
        if len(image_path.split('/')) == 1:
            continue  # last line
        food_type, image_id = image_path.split('/')[0], image_path.split('/')[1]

        # get image vector and tag based on given image and food type
        img_vect, img_tag = convert_image(image_id=image_id, image_tag=food_type)

        # add new example to dataset and tags
        dataset = np.concatenate((dataset, img_vect), axis=0)
        tags = np.concatenate((tags, img_tag), axis=0)

        # limiting dataset size
        if dataset.shape[0] == DATASET_SIZE:
            break
    dataset_file.close()
    return dataset, tags
