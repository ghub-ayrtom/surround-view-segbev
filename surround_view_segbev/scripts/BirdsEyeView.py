import numpy as np
from .utils import *
import cv2
from configs import bev_parameters


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


def fb_get_central_part(image):
    return image[:, xl:xr]


class BirdsEyeView():


    def __init__(self, images, node_logger):
        self.node_logger = node_logger

        self.frames = images
        self.weights = None
        self.masks = None

        self.image = np.zeros((bev_parameters.total_height, bev_parameters.total_width, 3), np.uint8)


    def get_weights_and_masks(self):
        left, front, right, back = self.frames

        G0, M0 = get_weight_mask_matrix(get_left_part(front), get_upper_part(left))
        G1, M1 = get_weight_mask_matrix(get_right_part(front), get_upper_part(right))
        G2, M2 = get_weight_mask_matrix(get_left_part(back), get_lower_part(left))
        G3, M3 = get_weight_mask_matrix(get_right_part(back), get_lower_part(right))

        self.weights = [np.stack((G, G, G), axis=2) for G in (G0, G1, G2, G3)]
        self.masks = [(M / 255.0).astype(int) for M in (M0, M1, M2, M3)]

        return np.stack((G0, G1, G2, G3), axis=2), np.stack((M0, M1, M2, M3), axis=2)


    def luminance_balance(self):


        def tune(x):
            if x >= 1:
                return x * np.exp((1 - x) * 0.5)
            else:
                return x * np.exp((1 - x) * 0.8)

        left, front, right, back = self.frames
        M0, M1, M2, M3 = self.masks

        left_B, left_G, left_R = cv2.split(left)
        front_B, front_G, front_R = cv2.split(front)
        right_B, right_G, right_R = cv2.split(right)
        back_B, back_G, back_R = cv2.split(back)

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

        x1 = t1 / (d1 / a1)**0.5
        x2 = t2 / (d2 / a2)**0.5
        x3 = t3 / (d3 / a3)**0.5

        x1 = tune(x1)
        x2 = tune(x2)
        x3 = tune(x3)

        front_B = adjust_luminance(front_B, x1)
        front_G = adjust_luminance(front_G, x2)
        front_R = adjust_luminance(front_R, x3)

        ###

        y1 = t1 / (b1 / c1)**0.5
        y2 = t2 / (b2 / c2)**0.5
        y3 = t3 / (b3 / c3)**0.5

        y1 = tune(y1)
        y2 = tune(y2)
        y3 = tune(y3)

        back_B = adjust_luminance(back_B, y1)
        back_G = adjust_luminance(back_G, y2)
        back_R = adjust_luminance(back_R, y3)

        ###

        z1 = t1 / (c1 / d1)**0.5
        z2 = t2 / (c2 / d2)**0.5
        z3 = t3 / (c3 / d3)**0.5

        z1 = tune(z1)
        z2 = tune(z2)
        z3 = tune(z3)

        left_B = adjust_luminance(left_B, z1)
        left_G = adjust_luminance(left_G, z2)
        left_R = adjust_luminance(left_R, z3)

        ###

        w1 = t1 / (a1 / b1)**0.5
        w2 = t2 / (a2 / b2)**0.5
        w3 = t3 / (a3 / b3)**0.5

        w1 = tune(w1)
        w2 = tune(w2)
        w3 = tune(w3)

        right_B = adjust_luminance(right_B, w1)
        right_G = adjust_luminance(right_G, w2)
        right_R = adjust_luminance(right_R, w3)

        self.frames = [
            cv2.merge((left_B, left_G, left_R)), 
            cv2.merge((front_B, front_G, front_R)), 
            cv2.merge((right_B, right_G, right_R)), 
            cv2.merge((back_B, back_G, back_R)), 
        ]


    @property
    def front_left(self): return self.image[:yt, :xl]
    @property
    def front_central(self): return self.image[:yt, xl:xr]
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
    def central(self): return self.image[yt:yb, xl:xr]
    @property
    def right_central(self): return self.image[yt:yb, xr:]


    def merge(self, image_1, image_2, i):
        G = self.weights[i]
        return (image_1 * G + image_2 * (1 - G)).astype(np.uint8)


    def stitch(self):
        left, front, right, back = self.frames

        np.copyto(self.front_central, fb_get_central_part(front))
        np.copyto(self.back_central, fb_get_central_part(back))
        np.copyto(self.left_central, lr_get_central_part(left))
        np.copyto(self.right_central, lr_get_central_part(right))

        np.copyto(self.front_left, self.merge(get_left_part(front), get_upper_part(left), 0))
        np.copyto(self.front_right, self.merge(get_right_part(front), get_upper_part(right), 1))
        np.copyto(self.back_left, self.merge(get_left_part(back), get_lower_part(left), 2))
        np.copyto(self.back_right, self.merge(get_right_part(back), get_lower_part(right), 3))


    def white_balance(self):
        self.image = make_white_balance(self.image)

    
    def add_ego_vehicle(self):
        ego_vehicle_image = cv2.imread(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
            'resource/images/ego_vehicle.png'
        ))

        ego_vehicle_image = cv2.cvtColor(ego_vehicle_image, cv2.COLOR_BGR2RGB)
        np.copyto(self.central, cv2.resize(ego_vehicle_image, (xr - xl, yb - yt)))
