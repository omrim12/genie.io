from py_edamam import Edamam
from constants import (
    EDAMAM_API
)


def edamam_api_recipes_client():
    return Edamam(
        recipes_appid=EDAMAM_API['recipes']['id'],
        recipes_appkey=EDAMAM_API['recipes']['key'],
    )


def edamam_api_food_client():
    return Edamam(
        food_appid=EDAMAM_API['food']['id'],
        food_appkey=EDAMAM_API['food']['key']
    )


def prompt_food_info(food_label):
    """
    :param food_label:
    :return:
    """
    # TODO: develop a nice output (:
    print(edamam_api_recipes_client().search_recipe(food_label))
