# images dataset
IMAGES_DIR = 'images/food-101/food-101/images'
IMAGES_META = 'images/food-101/food-101/meta'
IMAGE_SIZE = 56
FOOD_TYPES = 101

# genie classifier model files
GENIE_NET_FNAME = 'genie_net.h5'
GENIE_SCORE_FNAME = 'genie_res.h5'

# CNN variables
DATASET_BATCH = 10000
EPOCH=50
CONV_TYPE='same'

# Edamam API variables
EDAMAM_API = {
    'food': {
        'id': '3b32a5a9',
        'key': '6475aa8f96f37d9195f225b5cd009f5e'
    },
    'recipes': {
        'id': '05623be7',
        'key': '2a709f77459956e1d04330b98fcd90ee'
    }
}
TABLE_COLUMN_WIDTH = 10
