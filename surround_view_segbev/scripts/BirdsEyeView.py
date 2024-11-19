import numpy as np
from .utils import *
import cv2
from configs.BEV import bev_parameters


xl = bev_parameters.vehicle_leftside_edges_x
xr = bev_parameters.vehicle_rightside_edges_x
yt = bev_parameters.vehicle_topside_edges_y
yb = bev_parameters.vehicle_bottomside_edges_y


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


def fb_get_central_part(image):
    return image[:, xl:xr]


class BirdsEyeView():


    def __init__(self, node_logger, images=None, load_weights_and_masks=False):
        self.node_logger = node_logger
        self.frames = images

        self.weights = None
        self.masks = None

        self.image = np.zeros((bev_parameters.total_height, bev_parameters.total_width, 3), np.uint8)

        if load_weights_and_masks:
            self.load_weights_and_masks()


    def load_weights_and_masks(self):
        weights_file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 'configs/BEV/weights.npy')
        masks_file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 'configs/BEV/masks.npy')

        Gmat = np.load(weights_file_path)
        Mmat = np.load(masks_file_path)

        self.weights = [np.stack((Gmat[:, :, i], Gmat[:, :, i], Gmat[:, :, i]), axis=2) for i in range(Gmat.shape[2])]
        self.masks = [Mmat[:, :, i] for i in range(Mmat.shape[2])]


    def extract_frames(self):
        front_left = self.frames['camera_front_left']
        front = self.frames['camera_front']
        front_blind = self.frames['camera_front_blind']
        front_right = self.frames['camera_front_right']

        # back_left = self.frames['camera_rear_left']
        back = self.frames['camera_rear']
        # back_right = self.frames['camera_rear_right']

        return front_left, front, front_blind, front_right, back


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


    def luminance_balance(self):


        def tune(x):
            if x >= 1:
                return x * np.exp((1 - x) * 0.5)
            else:
                return x * np.exp((1 - x) * 0.8)

        if self.frames:
            left, front, front_blind, right, back = self.extract_frames()
            M0, M1, M2, M3, M4 = self.masks

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

            self.frames = [
                cv2.merge((left_B, left_G, left_R)), 
                cv2.merge((front_B, front_G, front_R)), 
                cv2.merge((front_blind_B, front_blind_G, front_blind_R)), 
                cv2.merge((right_B, right_G, right_R)), 
                cv2.merge((back_B, back_G, back_R)), 
            ]


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

            np.copyto(self.back_central, fb_get_central_part(back))

            np.copyto(self.left_central, lr_get_central_part(left))
            np.copyto(self.right_central, lr_get_central_part(right))

            np.copyto(self.front_left, self.merge(get_left_part(front), get_upper_part(left), 0))
            np.copyto(self.front_right, self.merge(get_right_part(front), get_upper_part(right), 2))

            np.copyto(self.front_central, f_get_central_part(front))
            np.copyto(self.front_central_blind, self.merge(f_get_central_part_blind(front), f_get_central_part_blind(front_blind), 1)[295 : - 13, 103 : - 100])

            np.copyto(self.back_left, self.merge(get_left_part(back), get_lower_part(left), 3))
            np.copyto(self.back_right, self.merge(get_right_part(back), get_lower_part(right), 4))


    def white_balance(self):
        self.image = make_white_balance(self.image)

    
    def add_ego_vehicle(self):
        ego_vehicle_image = cv2.imread(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
            'resource/images/ego_vehicle.png'
        ))

        ego_vehicle_image = cv2.cvtColor(ego_vehicle_image, cv2.COLOR_BGR2RGB)
        np.copyto(self.central, cv2.resize(ego_vehicle_image, (xr - xl, (yb + 5) - (yt - 10))))
