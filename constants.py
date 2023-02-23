# images dataset
IMAGES_DIR = 'images/food-101/food-101/images'
IMAGES_META = 'images/food-101/food-101/meta'
IMAGE_SIZE = 56
FOOD_TYPES = 101

# genie classifier model files
GENIE_NET_FNAME = 'genie_net.h5'
GENIE_SCORE_FNAME = 'genie_res.json'
GENIE_FOOD_IMG = 'food_img.jpg'

# CNN variables
DATASET_BATCH = 128
EPOCH = 50
CONV_TYPE = 'valid'
DROPOUT_PROB = 0.3
KERNEL_SIZE = 3
FILTERS = 64
ALPHA = 0.001

# Edamam API variables
TABLE_COLUMN_WIDTH = 10
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
