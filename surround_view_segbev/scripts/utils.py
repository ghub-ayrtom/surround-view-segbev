import numpy as np
import cv2
import os
from surround_view_segbev.configs import global_settings
import time
import math


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


def get_overlap_region_mask(image_1, image_2, images_part_name):
    cv2.destroyAllWindows()

    cv2.imshow(f'{images_part_name[0]}', cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.imshow(f'{images_part_name[1]}', cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    overlap_region = cv2.bitwise_and(image_1, image_2)

    cv2.imshow('overlap_region', cv2.cvtColor(overlap_region, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    mask = get_mask(overlap_region)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)

    cv2.imshow('overlap_region_mask', mask)
    cv2.waitKey(0)

    return mask


def get_outmost_polygon_boundary(image, image_part_name):
    mask = get_mask(image)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)

    cv2.imshow(f'{image_part_name}_unique_region_mask', mask)
    cv2.waitKey(0)

    polygon = None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    if len(contours) > 0:
        C = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]
        polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)

    return polygon


def get_weight_mask_matrix(image_1, image_2, images_part_name, max_distance=5):
    overlap_region_mask = get_overlap_region_mask(image_1, image_2, images_part_name)
    overlap_region_mask_inverted = cv2.bitwise_not(overlap_region_mask)

    cv2.imshow('overlap_region_mask_inverted', overlap_region_mask_inverted)
    cv2.waitKey(0)

    overlap_pixels_indices = np.where(overlap_region_mask == 255)

    image_1_unique_region = cv2.bitwise_and(image_1, image_1, mask=overlap_region_mask_inverted)
    image_2_unique_region = cv2.bitwise_and(image_2, image_2, mask=overlap_region_mask_inverted)

    cv2.imshow(f'{images_part_name[0]}_unique_region', cv2.cvtColor(image_1_unique_region, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.imshow(f'{images_part_name[1]}_unique_region', cv2.cvtColor(image_2_unique_region, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    G = get_mask(image_1).astype(np.float32) / 255.0

    image_1_polygon = get_outmost_polygon_boundary(image_1_unique_region, images_part_name[0])
    image_2_polygon = get_outmost_polygon_boundary(image_2_unique_region, images_part_name[1])

    cv2.imshow(
        f'{images_part_name[0]}_outmost_polygon_boundary', 
        cv2.drawContours(
            cv2.cvtColor(image_1_unique_region, cv2.COLOR_BGR2RGB), 
            [image_1_polygon], 
            -1, (255, 0, 0), 3, 
        )
    )

    cv2.waitKey(0)

    cv2.imshow(
        f'{images_part_name[1]}_outmost_polygon_boundary', 
        cv2.drawContours(
            cv2.cvtColor(image_2_unique_region, cv2.COLOR_BGR2RGB), 
            [image_2_polygon], 
            -1, (0, 255, 0), 3, 
        )
    )
    
    cv2.waitKey(0)

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
            distance_to_images_polygon = distance_to_image_1_polygon + distance_to_image_2_polygon

            if distance_to_images_polygon != 0:
                G[y, x] = distance_to_image_2_polygon / distance_to_images_polygon

    cv2.imshow('weight_matrix', G)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

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


def get_vectors_angle(vector_1, vector_2):
    # a * b = ax * bx + ay * by = |a| * |b| * cos(θ)
    scalar = vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1]

    # det(a, b) = ax * by - ay * bx = |a| * |b| * sin(θ)
    determinant = vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]  # det(a, b) > 0 - b CCW a, det(a, b) < 0 - b CW a

    return math.atan2(determinant, scalar)  # (-π, π]


def get_image_relative_coordinates(image_shape, x, y, scale_x=1.0, scale_y=1.0):
    return int(image_shape[1] / 2.0 + x * scale_x), int(image_shape[0] / 2.0 - y * scale_y)


def get_image_relative_route_point(route_point, ego_vehicle_yaw, ego_vehicle_position):
    route_point_relative = route_point.copy()

    dx = route_point[2] - ego_vehicle_position[1]
    dy = route_point[1] - ego_vehicle_position[0]

    route_point_relative[2] = dx * math.cos(-ego_vehicle_yaw) + dy * math.sin(-ego_vehicle_yaw)
    route_point_relative[1] = -dx * math.sin(-ego_vehicle_yaw) + dy * math.cos(-ego_vehicle_yaw)

    return route_point_relative


def get_global_coordinates_from_image_relative(point_relative, image_shape, ego_vehicle_yaw, ego_vehicle_position, scale_x, scale_y):
    # Извлекаем относительные координаты из центра изображения
    x_relative = int((point_relative[0] - image_shape[1] / 2.0) / scale_x)
    y_relative = int(((image_shape[0] / 2.0 - point_relative[1]) / scale_y) + 1.45)  # Убираем относительное смещение GPS в Webots

    # Преобразуем в глобальные относительные координаты с учётом поворота эго-автомобиля
    dx = x_relative * math.cos(ego_vehicle_yaw) - y_relative * math.sin(ego_vehicle_yaw)
    dy = x_relative * math.sin(ego_vehicle_yaw) + y_relative * math.cos(ego_vehicle_yaw)

    # Преобразуем в глобальные GPS-координаты
    latitude = ego_vehicle_position[0] + dy
    longitude = ego_vehicle_position[1] + dx

    return [latitude, longitude]


def draw_path_on_surround_view(
        surround_view_image, 
        ego_vehicle_vector_relative, 
        ego_vehicle_yaw, 
        ego_vehicle_position, 
        current_route_point_index, 
        route, 
    ):

    current_route_point_relative = None

    for i, point in enumerate(route):
        point_relative = get_image_relative_route_point(point, ego_vehicle_yaw, ego_vehicle_position)
        point_relative = get_image_relative_coordinates(
            surround_view_image.shape, 
            point_relative[2], 
            point_relative[1] - 1.45,  # 1.45 - относительное смещение GPS в Webots относительно центра эго-автомобиля (изображения)
            scale_x=30.0, 
            scale_y=27.5, 
        )

        point_color = (0, 255, 0) if point[0] else (0, 255, 255)

        # if point[0]:
        #     point_relative_temp = point_relative

        #     if point_relative[0] < 0:
        #         point_relative = 25, point_relative[1]
        #     elif point_relative[0] > surround_view_image.shape[1]:
        #         point_relative = surround_view_image.shape[1], point_relative[1]
        #     if point_relative[1] < 0:
        #         point_relative = point_relative[0], 25
        #     elif point_relative[1] > surround_view_image.shape[0]:
        #         point_relative = point_relative[0], surround_view_image.shape[0]

        #     if point_relative != point_relative_temp:
        #         route[i][0] = False

        #         point_global = get_global_coordinates_from_image_relative(
        #             point_relative, 
        #             surround_view_image.shape, 
        #             ego_vehicle_yaw, 
        #             ego_vehicle_position, 
        #             scale_x=30.0, 
        #             scale_y=27.5, 
        #         )

        #         if i != 0:
        #             route.insert(i, [False, point_global[0], point_global[1], 0.0])
        #             current_route_point_index = i
        #         else:
        #             route.insert(0, [False, point_global[0], point_global[1], 0.0])
        #             current_route_point_index = 0

        cv2.circle(
            surround_view_image, 
            point_relative, 
            5, 
            point_color, 
            -1, 
        )

        if i != 0 and point[0]:
            current_route_point_relative = point_relative

            previous_point_relative = get_image_relative_route_point(route[i - 1], ego_vehicle_yaw, ego_vehicle_position)
            previous_point_relative = get_image_relative_coordinates(
                surround_view_image.shape, 
                previous_point_relative[2], 
                previous_point_relative[1] - 1.45, 
                scale_x=30.0, 
                scale_y=27.5, 
            )

            cv2.arrowedLine(
                surround_view_image, 
                previous_point_relative, 
                point_relative, 
                (255, 0, 255), 
                3, 
            )

            cv2.putText(
                surround_view_image, 
                f'{(point[3] - 5.0):.1f}', 
                (
                    point_relative[0] + 10, 
                    point_relative[1], 
                ), 
                0, 
                0.75, 
                point_color, 
                2, 
            )

    cv2.circle(
        surround_view_image, 
        get_image_relative_coordinates(
            surround_view_image.shape, 
            ego_vehicle_position[1] - ego_vehicle_position[1], 
            # 40.0 - глобальное смещение GPS в Webots относительно центра эго-автомобиля (изображения)
            (ego_vehicle_position[0] - ego_vehicle_position[0]) - 40.0, 
        ), 
        5, 
        (0, 0, 255), 
        -1, 
    )

    cv2.arrowedLine(
        surround_view_image, 
        get_image_relative_coordinates(
            surround_view_image.shape, 
            ego_vehicle_position[1] - ego_vehicle_position[1], 
            (ego_vehicle_position[0] - ego_vehicle_position[0]) - 40.0, 
        ), 
        get_image_relative_coordinates(
            surround_view_image.shape, 
            ego_vehicle_vector_relative[1], 
            ego_vehicle_vector_relative[0], 
        ), 
        (0, 0, 255), 
        3, 
    )

    return surround_view_image, current_route_point_relative
