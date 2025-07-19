import numpy as np
from .utils import *
import cv2
import yaml
import traceback


# Координаты области изображения, занимаемой автомобилем: [(xl, yt), (xr, yb)]
xl = None  # Координата X левых углов
xr = None  # Координата X правых углов
yt = None  # Координата Y верхних углов
yb = None  # Координата Y нижних углов


def get_upper_part(image):
    return image[:yt, :]
def get_lower_part(image):
    return image[yb:, :]
def lr_get_central_part(image):
    return image[yt:yb, :]
def get_left_part(image):
    return image[:, :xl]
def get_right_part(image):
    return image[:, xr:]
def f_get_central_part_blind(image):
    return image[:, (xl - 135):(xr + 130)]
def f_get_central_part(image):
    return image[:295, (xl - 32):(xr + 30)]
def b_get_central_part(image):
    return image[:, xl:xr]


class BirdsEyeView():
    def __init__(self, node_logger, images=None, load_weights_and_masks=False):
        self.node_logger = node_logger
        self.frames = images

        self.bev_parameters = None

        self.near_shift_width = None   # Расстояние в пикселях между областью автомобиля (см. выше) и 
        self.near_shift_height = None  # угловыми шахматными досками размерностью 6x5 (см. сцену BEV.wbt)

        self.far_shift_width = None    # Расстояние в пикселях за пределами угловых шахматных досок 
        self.far_shift_height = None   # (чем больше эти значения, тем большую область покрывает круговой обзор)

        self.bev_total_width = None    # Итоговое разрешение единого сшитого изображения
        self.bev_total_height = None   #

        self.weights = None
        self.masks = None

        with open(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            'scripts/BEVFormer/bev_parameters.yaml', 
        )) as bev_parameters_yaml:
            try:
                self.bev_parameters = yaml.safe_load(bev_parameters_yaml)

                self.near_shift_width = self.bev_parameters['near_shift_width']
                self.near_shift_height = self.bev_parameters['near_shift_height']

                self.far_shift_width = self.bev_parameters['far_shift_width']
                self.far_shift_height = self.bev_parameters['far_shift_height']

                self.bev_total_width = self.bev_parameters['total_width_base'] + 2 * self.far_shift_width
                self.bev_total_height = self.bev_parameters['total_height_base'] + 2 * self.far_shift_height

                global xl, xr, yt, yb

                xl = self.far_shift_width + self.bev_parameters['vehicle_leftside_edges_x_inc'] + self.near_shift_width
                xr = self.bev_total_width - xl
                yt = self.far_shift_height + self.bev_parameters['vehicle_topside_edges_y_inc'] + self.near_shift_height
                yb = self.bev_total_height - yt

                bev_parameters_yaml.close()
            except yaml.YAMLError as e:
                if self.node_logger is not None:
                    self.node_logger.error(''.join(traceback.TracebackException.from_exception(e).format()))
                else:
                    print(e)

        self.image = np.zeros((self.bev_total_height, self.bev_total_width, 3), np.uint8)

        if load_weights_and_masks:
            self.load_weights_and_masks()

    def load_weights_and_masks(self):
        weights_file_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            'scripts/BEVFormer/weights.npy', 
        )
        masks_file_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            'scripts/BEVFormer/masks.npy', 
        )

        Gmat = np.load(weights_file_path)
        Mmat = np.load(masks_file_path)

        self.weights = [np.stack((Gmat[:, :, i], Gmat[:, :, i], Gmat[:, :, i]), axis=2) for i in range(Gmat.shape[2])]
        self.masks = [Mmat[:, :, i] for i in range(Mmat.shape[2])]

    def extract_frames(self):
        front_left = self.frames['camera_front_left'][0]
        front = self.frames['camera_front'][0]
        front_blind = self.frames['camera_front_blind'][0]
        front_right = self.frames['camera_front_right'][0]
        back = self.frames['camera_rear'][0]

        return front_left, front, front_blind, front_right, back
        # return front_left, front, front_right, back

    def get_weights_and_masks(self):
        if self.frames:
            left, front, front_blind, right, back = self.extract_frames()

            G0, M0 = get_weight_mask_matrix(get_left_part(front), get_upper_part(left))
            G1, M1 = get_weight_mask_matrix(f_get_central_part_blind(front), f_get_central_part_blind(front_blind))
            G2, M2 = get_weight_mask_matrix(get_right_part(front), get_upper_part(right))
            G3, M3 = get_weight_mask_matrix(get_left_part(back), get_lower_part(left))
            G4, M4 = get_weight_mask_matrix(get_right_part(back), get_lower_part(right))

            self.weights = [np.stack((G, G, G), axis=2) for G in (G0, G1, G2, G3, G4)]
            self.masks = [(M / 255.0).astype(int) for M in (M0, M1, M2, M3, M4)]

            return np.stack((G0, G1, G2, G3, G4), axis=2), np.stack((M0, M1, M2, M3, M4), axis=2)

            # left, front, right, back = self.extract_frames()

            # G0, M0 = get_weight_mask_matrix(get_left_part(front), get_upper_part(left), ['front_left_part', 'left_upper_part'])
            # G1, M1 = get_weight_mask_matrix(get_right_part(front), get_upper_part(right), ['front_right_part', 'right_upper_part'])
            # G2, M2 = get_weight_mask_matrix(get_left_part(back), get_lower_part(left), ['back_left_part', 'left_lower_part'])
            # G3, M3 = get_weight_mask_matrix(get_right_part(back), get_lower_part(right), ['back_right_part', 'right_lower_part'])

            # self.weights = [np.stack((G, G, G), axis=2) for G in (G0, G1, G2, G3)]
            # self.masks = [(M / 255.0).astype(int) for M in (M0, M1, M2, M3)]

            # return np.stack((G0, G1, G2, G3), axis=2), np.stack((M0, M1, M2, M3), axis=2)

    def luminance_balance(self):
        def tune(x):
            if x >= 1:
                return x * np.exp((1 - x) * 0.5)
            else:
                return x * np.exp((1 - x) * 0.8)

        if self.frames:
            left, front, front_blind, right, back = self.extract_frames()
            M0, M1, M2, M3, M4 = self.masks

            # left, front, right, back = self.extract_frames()
            # M0, M1, M2, M3 = self.masks

            left_B, left_G, left_R = cv2.split(left)
            front_B, front_G, front_R = cv2.split(front)
            front_blind_B, front_blind_G, front_blind_R = cv2.split(front_blind)
            right_B, right_G, right_R = cv2.split(right)
            back_B, back_G, back_R = cv2.split(back)

            a1 = mean_luminance_ratio(get_upper_part(right_B), get_right_part(front_B), M2)
            a2 = mean_luminance_ratio(get_upper_part(right_G), get_right_part(front_G), M2)
            a3 = mean_luminance_ratio(get_upper_part(right_R), get_right_part(front_R), M2)

            b1 = mean_luminance_ratio(get_right_part(back_B), get_lower_part(right_B), M4)
            b2 = mean_luminance_ratio(get_right_part(back_G), get_lower_part(right_G), M4)
            b3 = mean_luminance_ratio(get_right_part(back_R), get_lower_part(right_R), M4)

            c1 = mean_luminance_ratio(get_lower_part(left_B), get_left_part(back_B), M3)
            c2 = mean_luminance_ratio(get_lower_part(left_G), get_left_part(back_G), M3)
            c3 = mean_luminance_ratio(get_lower_part(left_R), get_left_part(back_R), M3)

            d1 = mean_luminance_ratio(get_left_part(front_B), get_upper_part(left_B), M0)
            d2 = mean_luminance_ratio(get_left_part(front_G), get_upper_part(left_G), M0)
            d3 = mean_luminance_ratio(get_left_part(front_R), get_upper_part(left_R), M0)

            e1 = mean_luminance_ratio(f_get_central_part_blind(front_B), f_get_central_part_blind(front_blind_B), M1)
            e2 = mean_luminance_ratio(f_get_central_part_blind(front_G), f_get_central_part_blind(front_blind_G), M1)
            e3 = mean_luminance_ratio(f_get_central_part_blind(front_R), f_get_central_part_blind(front_blind_R), M1)

            t1 = (a1 * b1 * c1 * d1 * e1)**0.25
            t2 = (a2 * b2 * c2 * d2 * e2)**0.25
            t3 = (a3 * b3 * c3 * d3 * e3)**0.25

            a1 = mean_luminance_ratio(get_upper_part(right_B), get_right_part(front_B), M1)
            a2 = mean_luminance_ratio(get_upper_part(right_G), get_right_part(front_G), M1)
            a3 = mean_luminance_ratio(get_upper_part(right_R), get_right_part(front_R), M1)

            b1 = mean_luminance_ratio(get_right_part(back_B), get_lower_part(right_B), M3)
            b2 = mean_luminance_ratio(get_right_part(back_G), get_lower_part(right_G), M3)
            b3 = mean_luminance_ratio(get_right_part(back_R), get_lower_part(right_R), M3)

            c1 = mean_luminance_ratio(get_lower_part(left_B), get_left_part(back_B), M2)
            c2 = mean_luminance_ratio(get_lower_part(left_G), get_left_part(back_G), M2)
            c3 = mean_luminance_ratio(get_lower_part(left_R), get_left_part(back_R), M2)

            d1 = mean_luminance_ratio(get_left_part(front_B), get_upper_part(left_B), M0)
            d2 = mean_luminance_ratio(get_left_part(front_G), get_upper_part(left_G), M0)
            d3 = mean_luminance_ratio(get_left_part(front_R), get_upper_part(left_R), M0)

            t1 = (a1 * b1 * c1 * d1)**0.25
            t2 = (a2 * b2 * c2 * d2)**0.25
            t3 = (a3 * b3 * c3 * d3)**0.25

            ###

            cd1 = t1 / (c1 / d1)**0.5
            cd2 = t2 / (c2 / d2)**0.5
            cd3 = t3 / (c3 / d3)**0.5

            cd1 = tune(cd1)
            cd2 = tune(cd2)
            cd3 = tune(cd3)

            left_B = adjust_luminance(left_B, cd1)
            left_G = adjust_luminance(left_G, cd2)
            left_R = adjust_luminance(left_R, cd3)

            ###

            da1 = t1 / (d1 / a1)**0.5
            da2 = t2 / (d2 / a2)**0.5
            da3 = t3 / (d3 / a3)**0.5

            da1 = tune(da1)
            da2 = tune(da2)
            da3 = tune(da3)

            front_B = adjust_luminance(front_B, da1)
            front_G = adjust_luminance(front_G, da2)
            front_R = adjust_luminance(front_R, da3)

            ###

            ec1 = t1 / (e1 / c1)**0.1
            ec2 = t2 / (e2 / c2)**0.1
            ec3 = t3 / (e3 / c3)**0.1

            ec1 = tune(ec1)
            ec2 = tune(ec2)
            ec3 = tune(ec3)

            front_blind_B = adjust_luminance(front_blind_B, ec1)
            front_blind_G = adjust_luminance(front_blind_G, ec2)
            front_blind_R = adjust_luminance(front_blind_R, ec3)

            ###

            ab1 = t1 / (a1 / b1)**0.5
            ab2 = t2 / (a2 / b2)**0.5
            ab3 = t3 / (a3 / b3)**0.5

            ab1 = tune(ab1)
            ab2 = tune(ab2)
            ab3 = tune(ab3)

            right_B = adjust_luminance(right_B, ab1)
            right_G = adjust_luminance(right_G, ab2)
            right_R = adjust_luminance(right_R, ab3)

            ###

            bc1 = t1 / (b1 / c1)**0.5
            bc2 = t2 / (b2 / c2)**0.5
            bc3 = t3 / (b3 / c3)**0.5

            bc1 = tune(bc1)
            bc2 = tune(bc2)
            bc3 = tune(bc3)

            back_B = adjust_luminance(back_B, bc1)
            back_G = adjust_luminance(back_G, bc2)
            back_R = adjust_luminance(back_R, bc3)

            ###

            self.frames['camera_front_left'][0] = cv2.merge((left_B, left_G, left_R))
            self.frames['camera_front'][0] = cv2.merge((front_B, front_G, front_R))
            self.frames['camera_front_blind'][0] = cv2.merge((front_blind_B, front_blind_G, front_blind_R))
            self.frames['camera_front_right'][0] = cv2.merge((right_B, right_G, right_R))
            self.frames['camera_rear'][0] = cv2.merge((back_B, back_G, back_R))

    @property
    def front_left(self): return self.image[:yt, :xl]
    @property
    def front_central(self): return self.image[:295, (xl - 32):(xr + 30)]
    @property
    def front_central_blind(self): return self.image[295:(yt - 13), (xl - 32):(xr + 30)]
    @property
    def front_right(self): return self.image[:yt, xr:]

    @property
    def back_left(self): return self.image[yb:, :xl]
    @property
    def back_central(self): return self.image[yb:, xl:xr]
    @property
    def back_right(self): return self.image[yb:, xr:]

    @property
    def left_central(self): return self.image[yt:yb, :xl]
    @property
    def central(self): return self.image[(yt - 10):(yb + 5), xl:xr]
    @property
    def right_central(self): return self.image[yt:yb, xr:]


    def merge(self, image_1, image_2, i):
        G = self.weights[i]
        return (image_1 * G + image_2 * (1 - G)).astype(np.uint8)

    def stitch(self):
        if self.frames:
            left, front, front_blind, right, back = self.extract_frames()
            # left, front, right, back = self.extract_frames()

            np.copyto(self.back_central, b_get_central_part(back))

            np.copyto(self.left_central, lr_get_central_part(left))
            np.copyto(self.right_central, lr_get_central_part(right))

            np.copyto(self.front_left, self.merge(get_left_part(front), get_upper_part(left), 0))
            np.copyto(self.front_right, self.merge(get_right_part(front), get_upper_part(right), 2))  # 1

            np.copyto(self.front_central, f_get_central_part(front))

            np.copyto(
                self.front_central_blind, 
                self.merge(
                    f_get_central_part_blind(front), 
                    f_get_central_part_blind(front_blind), 
                    1
                )[295 : - 13, 103 : - 100]
            )

            np.copyto(self.back_left, self.merge(get_left_part(back), get_lower_part(left), 3))       # 2
            np.copyto(self.back_right, self.merge(get_right_part(back), get_lower_part(right), 4))    # 3

    def white_balance(self):
        self.image = make_white_balance(self.image)
    
    def add_ego_vehicle_and_track_obstacles(self):
        blind_area_mask_image = cv2.imread(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
            'resource/images/blind_area_mask.png'
        ), cv2.IMREAD_UNCHANGED)  # RGBA
        ego_vehicle_image = cv2.imread(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
            'resource/images/ego_vehicle.png'
        ))

        if blind_area_mask_image.shape[2] == 4:
            mask_image_rgb = blind_area_mask_image[:, :, :3]
            mask_image_alpha = blind_area_mask_image[:, :, 3] / 255.0

            center_y, center_x = self.bev_total_height // 2, self.bev_total_width // 2

            start_y = center_y - blind_area_mask_image.shape[0] // 2
            end_y = start_y + blind_area_mask_image.shape[0]
            start_x = center_x - blind_area_mask_image.shape[1] // 2
            end_x = start_x + blind_area_mask_image.shape[1]

            blind_area = self.image[start_y:end_y, start_x:end_x]

            for channel in range(3):
                blind_area[:, :, channel] = (
                    mask_image_alpha * mask_image_rgb[:, :, channel] + (1 - mask_image_alpha) * blind_area[:, :, channel]
                )
            
            self.image[start_y:end_y, start_x:end_x] = blind_area

        np.copyto(self.central, cv2.resize(ego_vehicle_image, (xr - xl, (yb + 5) - (yt - 10))))

        if self.frames:
            for camera_name in self.frames:
                if len(self.frames[camera_name]) == 4:
                    obstacle_corners = self.frames[camera_name][1]
                    obstacle_centers = self.frames[camera_name][2]
                    obstacle_distances_m = self.frames[camera_name][3]

                    match camera_name:
                        case 'camera_front_left':
                            for i, bbox_corners in enumerate(obstacle_corners):
                                x1, y1, x2, y2, xy_corners_warped = bbox_corners

                                xy_corners_rotated = []

                                for x_warped, y_warped in xy_corners_warped:
                                    x_rotated = y_warped
                                    y_rotated = self.image.shape[0] - x_warped
                                    xy_corners_rotated.append([x_rotated, y_rotated])

                                xy_corners_warped = np.array(xy_corners_rotated).reshape((-1, 1, 2))
                                cv2.polylines(self.image, [xy_corners_warped], True, (255, 0, 0), 1)

                            for i, obstacle_center in enumerate(obstacle_centers):
                                center_y, center_x = self.image.shape[0] - obstacle_center[2], obstacle_center[3]
                                cv2.line(self.image, (362, 437), (center_x, center_y), (255, 0, 0), 1)
                                cv2.putText(
                                    self.image, 
                                    f'{obstacle_distances_m[i]:.1f}', 
                                    (center_x, center_y), 
                                    0, 
                                    0.5, 
                                    (255, 255, 255), 
                                    1, 
                                )

                        case 'camera_front':
                            for i, bbox_corners in enumerate(obstacle_corners):
                                x1, y1, x2, y2, xy_corners_warped = bbox_corners
                                xy_corners_warped = np.array(xy_corners_warped).reshape((-1, 1, 2))
                                cv2.polylines(self.image, [xy_corners_warped], True, (255, 0, 0), 1)

                            for i, obstacle_center in enumerate(obstacle_centers):
                                center_x, center_y = obstacle_center[2], obstacle_center[3]
                                cv2.line(self.image, (392, 375), (center_x, center_y), (255, 0, 0), 1)
                                cv2.putText(
                                    self.image, 
                                    f'{obstacle_distances_m[i]:.1f}', 
                                    (center_x, center_y), 
                                    0, 
                                    0.5, 
                                    (255, 255, 255), 
                                    1, 
                                )

                        case 'camera_front_blind':
                            for i, bbox_corners in enumerate(obstacle_corners):
                                x1, y1, x2, y2, xy_corners_warped = bbox_corners
                                xy_corners_warped = np.array(xy_corners_warped).reshape((-1, 1, 2))
                                cv2.polylines(self.image, [xy_corners_warped], True, (255, 0, 0), 1)

                            for i, obstacle_center in enumerate(obstacle_centers):
                                center_x, center_y = obstacle_center[2], obstacle_center[3]
                                cv2.line(self.image, (392, 332), (center_x, center_y), (255, 0, 0), 1)
                                cv2.putText(
                                    self.image, 
                                    f'{obstacle_distances_m[i]:.1f}', 
                                    (center_x, center_y), 
                                    0, 
                                    0.5, 
                                    (255, 255, 255), 
                                    1, 
                                )

                        case 'camera_front_right':
                            for i, bbox_corners in enumerate(obstacle_corners):
                                x1, y1, x2, y2, xy_corners_warped = bbox_corners

                                xy_corners_rotated = []

                                for x_warped, y_warped in xy_corners_warped:
                                    x_rotated = self.image.shape[1] - y_warped
                                    y_rotated = x_warped
                                    xy_corners_rotated.append([x_rotated, y_rotated])

                                xy_corners_warped = np.array(xy_corners_rotated).reshape((-1, 1, 2))
                                cv2.polylines(self.image, [xy_corners_warped], True, (255, 0, 0), 1)

                            for i, obstacle_center in enumerate(obstacle_centers):
                                center_y, center_x = obstacle_center[2], self.image.shape[1] - obstacle_center[3]
                                cv2.line(self.image, (421, 437), (center_x, center_y), (255, 0, 0), 1)
                                cv2.putText(
                                    self.image, 
                                    f'{obstacle_distances_m[i]:.1f}', 
                                    (center_x, center_y), 
                                    0, 
                                    0.5, 
                                    (255, 255, 255), 
                                    1, 
                                )

                        case 'camera_rear':
                            for i, bbox_corners in enumerate(obstacle_corners):
                                x1, y1, x2, y2, xy_corners_warped = bbox_corners

                                xy_corners_rotated = []

                                for x_warped, y_warped in xy_corners_warped:
                                    x_rotated = self.image.shape[1] - x_warped
                                    y_rotated = self.image.shape[0] - y_warped
                                    xy_corners_rotated.append([x_rotated, y_rotated])

                                xy_corners_warped = np.array(xy_corners_rotated).reshape((-1, 1, 2))
                                cv2.polylines(self.image, [xy_corners_warped], True, (255, 0, 0), 1)

                            for i, obstacle_center in enumerate(obstacle_centers):
                                center_x, center_y = self.image.shape[1] - obstacle_center[2], self.image.shape[0] - obstacle_center[3]
                                cv2.line(self.image, (392, 531), (center_x, center_y), (255, 0, 0), 1)
                                cv2.putText(
                                    self.image, 
                                    f'{obstacle_distances_m[i]:.1f}', 
                                    (center_x, center_y), 
                                    0, 
                                    0.5, 
                                    (255, 255, 255), 
                                    1, 
                                )
