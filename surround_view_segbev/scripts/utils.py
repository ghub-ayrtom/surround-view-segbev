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
