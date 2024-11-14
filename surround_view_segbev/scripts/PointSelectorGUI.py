import cv2
import numpy as np


def display_image(window_title, image):
    cv2.imshow(window_title, image)

    while True:
        click = cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE)

        if click < 0:
            return -1
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return -1
        if key == 13:  # Enter
            return 1


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
                return False
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return False
            if key == 13:
                return True
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
