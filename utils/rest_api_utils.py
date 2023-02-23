import os
from flask import jsonify
from flask_restful import Resource, request
from utils.cnn_utils import classify_client_input
from utils.img_utils import analyze_food_img
from utils.file_utils import load_genie
from constants import GENIE_FOOD_IMG
from PIL import Image


class GenieModelResource(Resource):
    def __init__(self):
        self.genie_model = load_genie()

    def post(self):
        # saving the given food image to project path
        food_img = request.files['food_image']
        food_img.save(GENIE_FOOD_IMG)

        return {'success': 'Image file uploaded successfully'}, 200

    def get(self):
        # open client food image
        food_image = Image.open(GENIE_FOOD_IMG)

        # classify food image type by genie
        image_array = analyze_food_img(food_img=food_image, as_path=False)
        food_type = classify_client_input(image_array=image_array, cnn_model=self.genie_model)

        # delete old food img from path TODO
        os.remove(GENIE_FOOD_IMG)

        return food_type, 200
