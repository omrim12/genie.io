import logging
import numpy as np
from PIL import Image
from utils.tag_utils import get_classes_list
from constants import (
    IMAGES_DIR,
    IMAGE_SIZE
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def analyze_food_img(food_img, as_path=True):
    try:
        # convert image to CNN input
        if as_path:
            food_img = Image.open(food_img)
        food_img = food_img.resize((IMAGE_SIZE, IMAGE_SIZE))

        # scale pixel intensities in range 0-1
        # and average RGB arrays
        if len(np.array(food_img).shape) == 2:
            food_img_arr = np.array([np.array(food_img) / 255.])
        else:
            food_img_arr = np.array([np.mean(np.array(food_img), axis=2) / 255.])
        food_img_arr = food_img_arr.reshape([IMAGE_SIZE, IMAGE_SIZE, 1])
        return food_img_arr
    except FileNotFoundError:
        return None


def convert_image(image_id, image_tag):
    food_img = Image.open(f'{IMAGES_DIR}/{image_tag}/{image_id}.jpg')

    # resize image to IMAGE_SIZE
    food_img = food_img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # scale pixel intensities in range 0-1
    # and average RGB arrays
    if len(np.array(food_img).shape) == 2:
        food_img_arr = np.array([np.array(food_img) / 255.])
    else:
        food_img_arr = np.array([np.mean(np.array(food_img), axis=2) / 255.])

    # extract food image class tag based on list index
    image_tag_id = get_classes_list().index(image_tag)
    image_tag_arr = np.array([[image_tag_id]], dtype=int)

    # close image object
    food_img.close()

    return food_img_arr, image_tag_arr
