from constants import (
    IMAGES_META
)

labels_list = []
classes_list = []


def get_labels_list():
    global labels_list

    if not labels_list:
        labels_raw_file = open(f'{IMAGES_META}/labels.txt', 'r')
        labels_raw = labels_raw_file.read()
        labels_list = labels_raw.split('\n')
        labels_raw_file.close()
    return labels_list


def get_classes_list():
    global classes_list

    if not classes_list:
        classes_raw_file = open(f'{IMAGES_META}/classes.txt', 'r')
        classes_raw = classes_raw_file.read()
        classes_list = classes_raw.split('\n')
        classes_raw_file.close()
    return classes_list
