import numpy as np
import cv2
import os
from configs import global_settings
import time


def image_bytes_to_numpy_array(image_bytes, image_shape, camera_name='', debug=False):
    image_array = np.frombuffer(image_bytes, np.uint8).reshape(image_shape)

    if debug:
        cv2.imwrite(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
            f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{camera_name}/debug/{time.strftime("%Y%m%d-%H%M%S")}.png'
        ), cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB))

    return image_array


def get_mask(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY)
    return mask


def get_overlap_region_mask(image_1, image_2):
    overlap_region = cv2.bitwise_and(image_1, image_2)

    mask = get_mask(overlap_region)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)

    return mask


def get_outmost_polygon_boundary(image):
    mask = get_mask(image)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)

    polygon = None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    if len(contours) > 0:
        C = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]
        polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)

    return polygon


def get_weight_mask_matrix(image_1, image_2, max_distance=5):
    overlap_region_mask = get_overlap_region_mask(image_1, image_2)
    overlap_region_mask_inverted = cv2.bitwise_not(overlap_region_mask)

    overlap_pixels_indices = np.where(overlap_region_mask == 255)

    image_1_unique_region = cv2.bitwise_and(image_1, image_1, mask=overlap_region_mask_inverted)
    image_2_unique_region = cv2.bitwise_and(image_2, image_2, mask=overlap_region_mask_inverted)

    G = get_mask(image_1).astype(np.float32) / 255.0

    image_1_polygon = get_outmost_polygon_boundary(image_1_unique_region)
    image_2_polygon = get_outmost_polygon_boundary(image_2_unique_region)

    distance_to_image_1_polygon = 0.0
    distance_to_image_2_polygon = 0.0

    for y, x in zip(*overlap_pixels_indices):
        xy = tuple([int(x), int(y)])

        if (image_2_polygon is not None) and (image_2_polygon.all() is not None):
            distance_to_image_2_polygon = cv2.pointPolygonTest(image_2_polygon, xy, True)

        if distance_to_image_2_polygon < max_distance:
            if (image_1_polygon is not None) and (image_1_polygon.all() is not None):
                distance_to_image_1_polygon = cv2.pointPolygonTest(image_1_polygon, xy, True)

            distance_to_image_2_polygon *= distance_to_image_2_polygon
            distance_to_image_1_polygon *= distance_to_image_1_polygon

            G[y, x] = distance_to_image_2_polygon / (distance_to_image_1_polygon + distance_to_image_2_polygon)

    return G, overlap_region_mask


def mean_luminance_ratio(image_1_gray, image_2_gray, mask):
    return np.sum(image_1_gray * mask) / np.sum(image_2_gray * mask)


def adjust_luminance(image_gray, factor):
    return np.minimum((image_gray * factor), 255).astype(np.uint8)


def make_white_balance(image):
    B, G, R = cv2.split(image)

    m1 = np.mean(B)
    m2 = np.mean(G)
    m3 = np.mean(R)

    K = (m1 + m2 + m3) / 3

    c1 = K / m1
    c2 = K / m2
    c3 = K / m3

    B = adjust_luminance(B, c1)
    G = adjust_luminance(G, c2)
    R = adjust_luminance(R, c3)

    return cv2.merge((B, G, R))
