import cv2
import numpy as np
import os
import yaml
from configs import global_settings
import traceback


def display_and_tune_projected_image(camera_name, src, image, fixed_main_bev_parameters):

    def do_nothing(x): pass

    def flip(image):
        match camera_name:
            case 'camera_front_left':
                return cv2.transpose(image)[::-1]
            case 'camera_front':
                return image.copy()
            case 'camera_front_right':
                return np.flip(cv2.transpose(image), 1)
            case 'camera_rear':
                return image.copy()[::-1, ::-1, :]
    
    window_title = f'{camera_name} (BEV)'
    
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE, 0)
    cv2.resizeWindow(window_title, 1920, 1080)  # Подставьте разрешение экрана вашего устройства

    bev_parameters = None

    with open(os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
        'surround_view_segbev/scripts/BEVFormer/bev_parameters.yaml', 
    )) as bev_parameters_yaml:
        try:
            bev_parameters = yaml.safe_load(bev_parameters_yaml)
            bev_parameters_yaml.close()
        except yaml.YAMLError as e:
            print(e)

    cv2.createTrackbar('near_shift_width', window_title, 0, 1000, do_nothing)
    cv2.setTrackbarPos('near_shift_width', window_title, bev_parameters['near_shift_width'])
    cv2.createTrackbar('near_shift_height', window_title, 0, 1000, do_nothing)
    cv2.setTrackbarPos('near_shift_height', window_title, bev_parameters['near_shift_height'])

    cv2.createTrackbar('far_shift_width', window_title, 0, 1000, do_nothing)
    cv2.setTrackbarPos('far_shift_width', window_title, bev_parameters['far_shift_width'])
    cv2.createTrackbar('far_shift_height', window_title, 0, 1000, do_nothing)
    cv2.setTrackbarPos('far_shift_height', window_title, bev_parameters['far_shift_height'])

    cv2.createTrackbar('total_width_base', window_title, 0, 1000, do_nothing)
    cv2.setTrackbarPos('total_width_base', window_title, bev_parameters['total_width_base'])
    cv2.createTrackbar('total_height_base', window_title, 0, 1000, do_nothing)
    cv2.setTrackbarPos('total_height_base', window_title, bev_parameters['total_height_base'])

    cv2.createTrackbar('vehicle_leftside_edges_x_inc', window_title, 0, 500, do_nothing)
    cv2.setTrackbarPos('vehicle_leftside_edges_x_inc', window_title, bev_parameters['vehicle_leftside_edges_x_inc'])
    cv2.createTrackbar('vehicle_topside_edges_y_inc', window_title, 0, 500, do_nothing)
    cv2.setTrackbarPos('vehicle_topside_edges_y_inc', window_title, bev_parameters['vehicle_topside_edges_y_inc'])

    cv2.createTrackbar('dst_points_13_x_inc', window_title, 0, 500, do_nothing)
    cv2.setTrackbarPos('dst_points_13_x_inc', window_title, 250)

    cv2.createTrackbar('dst_points_12_y_inc', window_title, 0, 500, do_nothing)
    cv2.setTrackbarPos('dst_points_12_y_inc', window_title, 0)

    cv2.createTrackbar('dst_points_24_x_inc', window_title, 0, 500, do_nothing)
    cv2.setTrackbarPos('dst_points_24_x_inc', window_title, 500)

    cv2.createTrackbar('dst_points_34_y_inc', window_title, 0, 500, do_nothing)
    cv2.setTrackbarPos('dst_points_34_y_inc', window_title, 250)

    while True:
        near_shift_width = cv2.getTrackbarPos('near_shift_width', window_title)
        near_shift_height = cv2.getTrackbarPos('near_shift_height', window_title)

        far_shift_width = cv2.getTrackbarPos('far_shift_width', window_title)
        far_shift_height = cv2.getTrackbarPos('far_shift_height', window_title)

        total_width = cv2.getTrackbarPos('total_width_base', window_title) + 2 * far_shift_width
        total_height = cv2.getTrackbarPos('total_height_base', window_title) + 2 * far_shift_height

        vehicle_leftside_edges_x = far_shift_width + cv2.getTrackbarPos('vehicle_leftside_edges_x_inc', window_title) + near_shift_width
        vehicle_topside_edges_y = far_shift_height + cv2.getTrackbarPos('vehicle_topside_edges_y_inc', window_title) + near_shift_height

        dst_points_13_x_inc = cv2.getTrackbarPos('dst_points_13_x_inc', window_title)
        dst_points_12_y_inc = cv2.getTrackbarPos('dst_points_12_y_inc', window_title)
        dst_points_24_x_inc = cv2.getTrackbarPos('dst_points_24_x_inc', window_title)
        dst_points_34_y_inc = cv2.getTrackbarPos('dst_points_34_y_inc', window_title)

        if fixed_main_bev_parameters:
            cv2.setTrackbarPos('near_shift_width', window_title, bev_parameters['near_shift_width'])
            cv2.setTrackbarPos('near_shift_height', window_title, bev_parameters['near_shift_height'])

            cv2.setTrackbarPos('far_shift_width', window_title, bev_parameters['far_shift_width'])
            cv2.setTrackbarPos('far_shift_height', window_title, bev_parameters['far_shift_height'])

            cv2.setTrackbarPos('total_width_base', window_title, bev_parameters['total_width_base'])
            cv2.setTrackbarPos('total_height_base', window_title, bev_parameters['total_height_base'])

            cv2.setTrackbarPos('vehicle_leftside_edges_x_inc', window_title, bev_parameters['vehicle_leftside_edges_x_inc'])
            cv2.setTrackbarPos('vehicle_topside_edges_y_inc', window_title, bev_parameters['vehicle_topside_edges_y_inc'])

        if dst_points_13_x_inc >= dst_points_24_x_inc:
            if dst_points_13_x_inc != 0 and dst_points_13_x_inc != 1:
                dst_points_13_x_inc -= 1
                cv2.setTrackbarPos('dst_points_13_x_inc', window_title, dst_points_13_x_inc)
            else:
                dst_points_24_x_inc = 2
                cv2.setTrackbarPos('dst_points_24_x_inc', window_title, dst_points_24_x_inc)
        elif dst_points_12_y_inc >= dst_points_34_y_inc:
            if dst_points_12_y_inc != 0 and dst_points_12_y_inc != 1:
                dst_points_12_y_inc -= 1
                cv2.setTrackbarPos('dst_points_12_y_inc', window_title, dst_points_12_y_inc)
            else:
                dst_points_34_y_inc = 2
                cv2.setTrackbarPos('dst_points_34_y_inc', window_title, dst_points_34_y_inc)
        else:
            # Координаты четырёх точек в порядке их проставления, но образующие собой как бы фигуру 
            # прямоугольника - вид сверху на отмечаемую ранее через утилиту PointSelector жёлтую область
            dst = np.float32([
                (far_shift_height + dst_points_13_x_inc, far_shift_width + dst_points_12_y_inc), 
                (far_shift_height + dst_points_24_x_inc, far_shift_width + dst_points_12_y_inc), 
                (far_shift_height + dst_points_13_x_inc, far_shift_width + dst_points_34_y_inc), 
                (far_shift_height + dst_points_24_x_inc, far_shift_width + dst_points_34_y_inc), 
            ])

            # Разрешение BEV-изображения для текущей видеокамеры 
            # (см. итоговый результат работы PointSelector)
            projection_shape = ()

            match camera_name:
                case 'camera_front_left':
                    projection_shape = (total_height, vehicle_leftside_edges_x)
                case 'camera_front':
                    projection_shape = (total_width, vehicle_topside_edges_y)
                case 'camera_front_right':
                    projection_shape = (total_height, vehicle_leftside_edges_x)
                case 'camera_rear':
                    projection_shape = (total_width, vehicle_topside_edges_y)

            projection_matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC)  # cv2.getPerspectiveTransform(src, dst)
            image_projected = flip(cv2.warpPerspective(image, projection_matrix, projection_shape))

            cv2.imshow(window_title, image_projected)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                return None
            if key == 13:  # Enter
                if not fixed_main_bev_parameters:
                    bev_parameters_yaml_path = os.path.join(
                        os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
                        'surround_view_segbev/scripts/BEVFormer/bev_parameters.yaml', 
                    )

                    bev_parameters['near_shift_width'] = near_shift_width
                    bev_parameters['near_shift_height'] = near_shift_height

                    bev_parameters['far_shift_width'] = far_shift_width
                    bev_parameters['far_shift_height'] = far_shift_height

                    bev_parameters['total_width_base'] = cv2.getTrackbarPos('total_width_base', window_title)
                    bev_parameters['total_height_base'] = cv2.getTrackbarPos('total_height_base', window_title)
                    
                    bev_parameters['vehicle_leftside_edges_x_inc'] = cv2.getTrackbarPos('vehicle_leftside_edges_x_inc', window_title)
                    bev_parameters['vehicle_topside_edges_y_inc'] = cv2.getTrackbarPos('vehicle_topside_edges_y_inc', window_title)

                    with open(bev_parameters_yaml_path, 'w') as bev_parameters_yaml:
                        yaml.dump(bev_parameters, bev_parameters_yaml, sort_keys=False)

                camera_name_short = ''
                i = len(camera_name) - 1

                while camera_name[i] != '_':
                    camera_name_short += camera_name[i]
                    i -= 1

                camera_parameters_file_path = os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
                    f'configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/parameters/{camera_name}.yaml', 
                )

                if os.path.isfile(camera_parameters_file_path):
                    try:
                        camera_parameters_file_read = cv2.FileStorage(camera_parameters_file_path, cv2.FILE_STORAGE_READ)
                        camera_parameters_file_write = cv2.FileStorage(camera_parameters_file_path, cv2.FILE_STORAGE_WRITE)

                        if camera_parameters_file_read.isOpened() and camera_parameters_file_write.isOpened():
                            image_shape = np.array(camera_parameters_file_read.getNode('image_resolution').mat(), dtype=int).flatten()
                            K = np.array(camera_parameters_file_read.getNode('camera_matrix').mat())
                            D = np.array(camera_parameters_file_read.getNode('distortion_coefficients').mat())

                            camera_parameters_file_write.write('image_resolution', image_shape)
                            camera_parameters_file_write.write('camera_matrix', K)
                            camera_parameters_file_write.write('distortion_coefficients', D)
                            camera_parameters_file_write.write('projection_matrix', projection_matrix)

                            camera_parameters_file_read.release()
                            camera_parameters_file_write.release()
                    except Exception as e:
                        print(''.join(traceback.TracebackException.from_exception(e).format()))
                return [cv2.cvtColor(image_projected, cv2.COLOR_RGB2BGR)]


class PointSelector(object):

    POINT_COLOR = (0, 0, 255)
    FILL_COLOR = (0, 255, 255)


    def __init__(self, image, title="PointSelector"):
        self.image = image
        self.title = title
        self.keypoints = []


    def draw(self):
        image_copy = self.image.copy()

        for i, p in enumerate(self.keypoints):
            cv2.circle(image_copy, p, 5, self.POINT_COLOR, -1)
            cv2.putText(image_copy, str(i), (p[0], p[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.POINT_COLOR, 2)

        if len(self.keypoints) == 2:
            point_1, point_2 = self.keypoints
            cv2.line(image_copy, point_1, point_2, self.POINT_COLOR, 2)

        if len(self.keypoints) > 2:
            mask = self.create_mask_from_pixels(self.keypoints, self.image.shape)
            image_copy = self.draw_mask(image_copy, mask)

        cv2.imshow(self.title, image_copy)


    def onclick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.keypoints.append((x, y))
            self.draw()


    def loop(self):
        cv2.namedWindow(self.title)
        cv2.setMouseCallback(self.title, self.onclick)
        cv2.imshow(self.title, self.image)

        while True:
            click = cv2.getWindowProperty(self.title, cv2.WND_PROP_AUTOSIZE)

            if click < 0:
                return 0
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return 0
            if key == 13:
                return 1
            if key == ord("d"):
                if len(self.keypoints) > 0:
                    self.keypoints.pop()
                    self.draw()


    def create_mask_from_pixels(self, pixels, image_shape):
        mask = np.zeros(image_shape[:2], np.int8)

        pixels = np.int32(pixels).reshape(-1, 2)
        hull = cv2.convexHull(pixels)

        cv2.fillConvexPoly(mask, hull, 1, lineType=8, shift=0)

        mask = mask.astype(bool)
        return mask


    def draw_mask(self, image, mask):
        image_copy = np.zeros_like(image)
        image_copy[:, :] = self.FILL_COLOR

        mask = np.array(mask, dtype=np.uint8)
        mask_new = cv2.bitwise_and(image_copy, image_copy, mask=mask)
        cv2.addWeighted(image, 1.0, mask_new, 0.5, 0.0, image)

        return image
