from termcolor import cprint
from prettytable import PrettyTable
from py_edamam import Edamam
from constants import (
    EDAMAM_API,
    TABLE_COLUMN_WIDTH
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


def parse_recipe_ingredients(recipe_raw):
    return "\n".join(recipe_raw['recipe']['ingredientLines'])


def parse_recipes_data(recipes_raw):
    # init recipes table with headers
    recipes_table = PrettyTable()
    recipes_table.field_names = [
        'Recipe name',
        # 'Recipe ingredients',
        'Cuisine',
        'Recipe source'
    ]

    # parse recipes from raw content
    for recipe in recipes_raw:
        # parse recipe info
        recipe_name = recipe['recipe']['label']
        # recipe_ingredients = parse_recipe_ingredients(recipe)
        recipe_cuisine = recipe['recipe']['cuisineType'][0]
        recipe_source = recipe['recipe']['url']

        # add to recipes table
        recipes_table.add_row([
            recipe_name,
            # recipe_ingredients,
            recipe_cuisine,
            recipe_source
        ])

    # adjust table columns width
    recipes_table._max_width = {
        f"Column {i+1}: {TABLE_COLUMN_WIDTH}"
        for i in range(len(recipes_raw))
    }

    return recipes_table


def parse_food_data(food_raw):
    # TODO: create food table for food callback
    return PrettyTable()


def get_recipes_info(food_label):
    """
    :param food_label:
    :return:
    """
    # init api client instance
    recipes_api_client = edamam_api_recipes_client()

    # query edamam api food and recipes according to given label
    recipes_callback = recipes_api_client.search_recipe(food_label)

    # create table
    recipes_table = parse_recipes_data(recipes_callback['hits'])
    return recipes_table


def get_food_info(food_label):
    """
    :param food_label:
    :return:
    """
    # init api client instances
    food_api_client = edamam_api_food_client()

    # query edamam api food and recipes according to given label
    food_callback = food_api_client.search_food(food_label)

    # create table
    food_table = parse_food_data(food_callback)
    return food_table
