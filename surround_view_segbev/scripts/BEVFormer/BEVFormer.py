import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from surround_view_segbev.scripts.PointSelectorGUI import PointSelector, display_and_tune_projected_image
import numpy as np
import cv2
from surround_view_segbev.scripts.BirdsEyeView import BirdsEyeView
import os
import yaml


class BEVFormer:
    def __init__(self, root):
        self.root = root
        self.root.title('BEVFormer')

        self.frames = {}

        self.images = [None, None, None, None]
        self.images_projected = {}

        self.src = []

        self.checkbox_value = tk.BooleanVar(value=True)
        self.load_src_keypoints = self.checkbox_value.get()

        self.src_keypoints_checkbox = tk.Checkbutton(
            root, 
            command=self.on_checkbox_toggle, 
            text='Load src keypoints', 
            variable=self.checkbox_value, 
        )

        self.done_icon = ImageTk.PhotoImage(Image.open('../../../resource/images/done.png').resize((32, 32)))
        self.fixed_main_bev_parameters = False

        self.bev_parameters = None
        self.bev_parameters_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
            'scripts/BEVFormer/bev_parameters.yaml', 
        )

        with open(self.bev_parameters_path, 'r') as bev_parameters_yaml:
            try:
                self.bev_parameters = yaml.safe_load(bev_parameters_yaml)
                bev_parameters_yaml.close()
            except yaml.YAMLError as e:
                print(e)

        self.setup_gui()

    def setup_gui(self):
        grid = tk.Frame(self.root)
        grid.pack(padx=15, pady=15)

        ego_vehicle_image = ImageTk.PhotoImage(Image.open('../../../resource/images/ego_vehicle.png'))

        ego_vehicle_panel = tk.Label(grid)
        ego_vehicle_panel.configure(image=ego_vehicle_image)
        ego_vehicle_panel.image = ego_vehicle_image
        ego_vehicle_panel.grid(row=1, column=1, pady=15)

        self.add_image_button(grid, index=0, column=0, row=1)
        self.add_image_button(grid, index=1, column=1, row=0)
        self.add_image_button(grid, index=2, column=2, row=1)
        self.add_image_button(grid, index=3, column=1, row=2)

        self.src_keypoints_checkbox.pack(anchor='w', padx=7, pady=12)

    def add_image_button(self, parent, index, column, row):
        frame = tk.Frame(parent)
        frame.grid(column=column, row=row)

        self.frames[index] = frame

        image_button = tk.Button(frame, text='Add image...', command=lambda idx=index: self.load_image(idx, image_button))
        image_button.pack()

    def load_image(self, index, image_button):
        image_path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp')])

        if not image_path:
            return
        try:
            image = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_BGR2RGB)
            self.images[index] = image

            image_button.config(text='Select points...', command=lambda idx=index: self.get_projection_matrix(idx, image_button))
            self.get_projection_matrix(index, image_button)
        except Exception as e:
            print(e)

    def on_checkbox_toggle(self):
        self.load_src_keypoints = self.checkbox_value.get()

    def form_bev_and_save_npys(self, images):
        bev = BirdsEyeView(None, images=images)

        Gmat, Mmat = bev.get_weights_and_masks()
        bev.luminance_balance()
        bev.stitch()
        bev.white_balance()
        bev.add_ego_vehicle_and_track_obstacles()

        cv2.imshow("Surround Bird's Eye View", cv2.cvtColor(bev.image, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            root.quit()
            cv2.destroyAllWindows()
        if key == 13:  # Enter
            result_files_save_path = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                'BEVFormer/', 
            )

            np.save(result_files_save_path + 'weights.npy', Gmat)
            np.save(result_files_save_path + 'masks.npy', Mmat)

            bev.frames.clear()

            root.quit()
            cv2.destroyAllWindows()

    def get_projection_matrix(self, index, image_button):
        try:
            camera_name = ''
            image = self.images[index]

            match index:
                case 0:
                    camera_name = 'camera_front_left'
                case 1:
                    camera_name = 'camera_front'
                case 2:
                    camera_name = 'camera_front_right'
                case 3:
                    camera_name = 'camera_rear'

            self.src = None

            if not self.load_src_keypoints:
                gui = PointSelector(image, title=camera_name)
                choice = gui.loop()

                if choice > 0:
                    # Список координат четырёх точек, отмечаемых на изображении с каждой из видеокамер через 
                    # инструмент PointSelector, строго в их следующем порядке: верхняя левая, верхняя правая, нижняя левая, нижняя правая
                    self.src = np.float32(gui.keypoints)
                    self.bev_parameters['src_keypoints'][camera_name] = self.src.tolist()

                    with open(self.bev_parameters_path, 'w') as bev_parameters_yaml:
                        yaml.dump(self.bev_parameters, bev_parameters_yaml, sort_keys=False)
            else:
                self.src = np.float32(self.bev_parameters['src_keypoints'][camera_name])

            if self.src is not None:
                self.images_projected[camera_name] = display_and_tune_projected_image(camera_name, self.src, image, self.fixed_main_bev_parameters)

                if self.images_projected[camera_name] is not None:
                    image_button.destroy()

                    image_label = tk.Label(self.frames[index], image=self.done_icon)
                    image_label.pack()

                    if camera_name == 'camera_front_left':
                        if 'camera_front' in self.images_projected.keys():
                            self.src_keypoints_checkbox.pack(anchor='w', padx=9, pady=14)
                        else:
                            self.src_keypoints_checkbox.pack(anchor='w', padx=9, pady=12)
                    elif camera_name == 'camera_front':
                        if 'camera_front_left' in self.images_projected.keys():
                            self.src_keypoints_checkbox.pack(anchor='w', padx=9, pady=14)
                        else:
                            self.src_keypoints_checkbox.pack(anchor='w', padx=7, pady=14)

                    if len(self.images_projected) == 1:
                        self.fixed_main_bev_parameters = True

                        with open(self.bev_parameters_path, 'r') as bev_parameters_yaml:
                            try:
                                self.bev_parameters = yaml.safe_load(bev_parameters_yaml)
                                bev_parameters_yaml.close()
                            except yaml.YAMLError as e:
                                print(e)
                    elif len(self.images_projected) == 4:
                        self.root.update()
                        self.form_bev_and_save_npys(self.images_projected)

            cv2.destroyAllWindows()
        except Exception as e:
            print(e)

if __name__ == '__main__':
    root = tk.Tk()
    app = BEVFormer(root)
    root.mainloop()
