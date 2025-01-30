import os
import random
import shutil
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from configs import global_settings


detector_all_images_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_DETECTOR_FOLDER_NAME}/data/all/images'
)
detector_all_labels_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_DETECTOR_FOLDER_NAME}/data/all/labels'
)

detector_train_images_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_DETECTOR_FOLDER_NAME}/data/train/images'
)
detector_val_images_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_DETECTOR_FOLDER_NAME}/data/val/images'
)
detector_test_images_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_DETECTOR_FOLDER_NAME}/data/test/images'
)

labels = {
    'nroad': 0,           # Фон
    'road': 1,            # Дорога
    'plastic_barrel': 2,  # Препятствие
}

segmentor_all_images_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_SEGMENTOR_FOLDER_NAME}/data/all/images'
)
segmentor_all_masks_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_SEGMENTOR_FOLDER_NAME}/data/all/masks'
)
segmentor_all_annotations_file_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_SEGMENTOR_FOLDER_NAME}/data/all/annotations.xml'
)

segmentor_train_images_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_SEGMENTOR_FOLDER_NAME}/data/train/images'
)
segmentor_val_images_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_SEGMENTOR_FOLDER_NAME}/data/val/images'
)
segmentor_test_images_folder_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
    f'models/{global_settings.USED_SEGMENTOR_FOLDER_NAME}/data/test/images'
)


def parse_segmentor_annotations_and_create_masks(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    for image_node in root.findall('image'):
        image_name = image_node.get('name')
        image_width = int(image_node.get('width'))
        image_height = int(image_node.get('height'))

        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        polygon_nodes = sorted(image_node.findall('polygon'), key=lambda p: int(p.get('z_order', 0)))

        for polygon in polygon_nodes:
            label = polygon.get('label')
            points = polygon.get('points')

            points = [
                [float(coordinates) for coordinates in point.split(',')] for point in points.split(';')
            ]
            points = np.array(points, dtype=np.int32)

            class_color = labels[label]
            cv2.fillPoly(mask, [points], color=class_color)

            cv2.imwrite(os.path.join(segmentor_all_masks_folder_path, image_name), mask)


def move_images_with_labels(images_subset, destination_folder_path):
    for image_name in images_subset:
        shutil.move(os.path.join(detector_all_images_folder_path, image_name), os.path.join(destination_folder_path, image_name))
        shutil.move(os.path.join(detector_all_labels_folder_path, image_name[:-3] + 'txt'), os.path.join(destination_folder_path[:-6] + 'labels', image_name[:-3] + 'txt'))


def move_images_with_masks(images_subset, destination_folder_path):
    for image_name in images_subset:
        shutil.move(os.path.join(segmentor_all_images_folder_path, image_name), os.path.join(destination_folder_path, image_name))
        shutil.move(os.path.join(segmentor_all_masks_folder_path, image_name), os.path.join(destination_folder_path[:-6] + 'masks', image_name))


camera_images = {
    'cf': [],   # camera_front
    'cfb': [],  # camera_front_blind
    'cfl': [],  # camera_front_left
    'cfr': [],  # camera_front_right
    'cr': [],   # camera_rear
    # 'sv': [],     # surround_view
}

# parse_segmentor_annotations_and_create_masks(segmentor_all_annotations_file_path)

for image_name in os.listdir(detector_all_images_folder_path):  # segmentor_all_images_folder_path
    for camera_name_acronym in camera_images:
        if f'{camera_name_acronym}_' in image_name:
            camera_images[camera_name_acronym].append(image_name)

train_images, val_images, test_images = [], [], []

for camera_name_acronym in camera_images:
    images = camera_images[camera_name_acronym]
    random.shuffle(images)

    train_images.extend(images[:16])   # 80%
    val_images.extend(images[16:18])   # 10%
    test_images.extend(images[18:20])  # 10%

move_images_with_labels(train_images, detector_train_images_folder_path)
move_images_with_labels(val_images, detector_val_images_folder_path)
move_images_with_labels(test_images, detector_test_images_folder_path)

# move_images_with_masks(train_images, segmentor_train_images_folder_path)
# move_images_with_masks(val_images, segmentor_val_images_folder_path)
# move_images_with_masks(test_images, segmentor_test_images_folder_path)

print('\nThe shuffling has been successfully completed!\n')
print(f'train subset: {len(train_images)} images and labels or masks')
print(f'val subset: {len(val_images)} images and labels or masks')
print(f'test subset: {len(test_images)} images and labels or masks\n')
