import logging
import numpy as np
from utils.img_utils import convert_image
from concurrent.futures import ThreadPoolExecutor
from constants import (
    IMAGE_SIZE,
    IMAGES_META,
    DATASET_BATCH
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_food_101():
    LOGGER.info("Loading Food 101 dataset")
    # init datasets by types
    # running datasets init processes in separate
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
    subset_size = 5000

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

        # sampling dataset size
        if ds_type == 'train' and dataset.shape[0] == 4500:
            break
        if ds_type == 'test' and dataset.shape[0] == 1500:
            break
        # if dataset.shape[0] % subset_size == 0:
        #     break
        #     LOGGER.info(f'{dataset.shape[0]} images loaded to {ds_type} dataset')
    dataset_file.close()
    return dataset, tags
