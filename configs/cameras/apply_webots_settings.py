import sys
import os
from configs import global_settings
import yaml
import traceback
import math
import shutil
import time


cameras_webots_settings = None

webots_world_description_data = None
description_data_buffer = []


def buffering_description_data(line_index):
    global webots_world_description_data, description_data_buffer

    if len(description_data_buffer) != 0:
        description_data_buffer.clear()
    for new_line_index in range(line_index, len(webots_world_description_data)):
        description_data_buffer.append(webots_world_description_data[new_line_index])


def unbuffering_description_data(line_index):
    global webots_world_description_data, description_data_buffer

    webots_world_description_data = webots_world_description_data[:line_index]
    webots_world_description_data.extend(description_data_buffer)


def main(args=None):
    if len(args) > 1:
        with open(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
            f'resource/worlds/{args[1]}'
        ), 'r+') as webots_world_wbt_file:
            try:
                global webots_world_description_data
                webots_world_description_data = webots_world_wbt_file.readlines()

                with open(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/webots_settings.yaml'
                )) as webots_settings_yaml_file:
                    try:
                        cameras_webots_settings = yaml.safe_load(webots_settings_yaml_file)
                        webots_settings_yaml_file.close()

                        horizontal_fov = math.radians(cameras_webots_settings['horizontal_fov'])
                        vertical_fov = math.radians(cameras_webots_settings['vertical_fov'])

                        cameras_image_width = cameras_webots_settings['image_width']
                        cameras_image_height = cameras_webots_settings['image_height']

                        focal_length = cameras_webots_settings['focal_length']

                        distortion_center = cameras_webots_settings["distortion_center"]
                        radial_distortion = cameras_webots_settings["radial_distortion"]
                        tangential_distortion = cameras_webots_settings["tangential_distortion"]

                        min_range = cameras_webots_settings['min_range']
                        max_range = cameras_webots_settings['max_range']

                        if cameras_webots_settings['use_fov_values']:
                            cameras_image_height = round(cameras_image_width * (math.tan(vertical_fov / 2) / math.tan(horizontal_fov / 2)))
                            focal_length = round(cameras_image_width / (2 * math.tan(horizontal_fov / 2)))

                            with open(os.path.join(
                                os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                                f'cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/webots_settings.yaml'
                            ), 'r+') as webots_settings_yaml_file:
                                try:
                                    webots_settings_data = webots_settings_yaml_file.readlines()

                                    for line_index in range(0, len(webots_settings_data)):
                                        line = webots_settings_data[line_index]

                                        if 'image_height' in line:
                                            webots_settings_data[line_index] = line.replace(line.split()[1], str(cameras_image_height))
                                        elif 'focal_length' in line:
                                            webots_settings_data[line_index] = line.replace(line.split()[1], str(focal_length))
                                    
                                    webots_settings_yaml_file.seek(0)
                                    webots_settings_yaml_file.writelines(webots_settings_data)
                                    webots_settings_yaml_file.truncate()

                                    webots_settings_data.clear()
                                    webots_settings_yaml_file.close()
                                except Exception as e:
                                    print(''.join(traceback.TracebackException.from_exception(e).format()))

                        found_rangefinder_name, found_camera_name = False, False
                        line_index = 0

                        # Пока не достигли конца файла с описанием Webots-мира
                        while line_index != len(webots_world_description_data):
                            line = webots_world_description_data[line_index]  # Считываем из него очередную строку

                            # Если в ней было найдено ключевое слово RangeFinder
                            if found_rangefinder_name:
                                if 'fieldOfView' in line:
                                    # Перезаписываем текущее значение на то, что было указано в конфигурационном файле webots_settings.yaml
                                    webots_world_description_data[line_index] = line.replace(line.split()[1], str(horizontal_fov))
                                elif 'width' in line:
                                    webots_world_description_data[line_index] = line.replace(line.split()[1], str(cameras_image_width))
                                elif 'height' in line:
                                    webots_world_description_data[line_index] = line.replace(line.split()[1], str(cameras_image_height))
                                elif 'minRange' in line:
                                    webots_world_description_data[line_index] = line.replace(line.split()[1], str(min_range))
                                elif 'maxRange' in line:
                                    webots_world_description_data[line_index] = line.replace(line.split()[1], str(max_range))
                                elif 'lens' in line:
                                    lens_node_set_parameters_count = 0
                                    indentation_level = line.count(' ')  # Уровень табуляции в текущей строке

                                    # Пропускаем строки без интересующих нас параметров
                                    while '}' not in line:
                                        lens_node_set_parameters_count += 1  # Считаем количество уже заданных параметров для узла линзы
                                        line_index += 1
                                        line = webots_world_description_data[line_index]

                                    buffering_description_data(line_index)  # Буферизуем все строки ниже line_index

                                    if lens_node_set_parameters_count > 0:
                                        line_index -= lens_node_set_parameters_count - 1

                                    # Дописываем или перезаписываем параметры для узла линзы
                                    webots_world_description_data[line_index] = f'{" " * indentation_level}center {distortion_center[0]} {distortion_center[1]}\n'
                                    line_index += 1

                                    webots_world_description_data[line_index] = f'{" " * indentation_level}radialCoefficients {radial_distortion[0]} {radial_distortion[1]}\n'
                                    line_index += 1

                                    webots_world_description_data[line_index] = f'{" " * indentation_level}tangentialCoefficients {tangential_distortion[0]} {tangential_distortion[1]}\n'
                                    line_index += 1

                                    unbuffering_description_data(line_index)  # Возвращаем из буфера оставшуюся часть описания
                                else:
                                    found_rangefinder_name = False
                            elif found_camera_name:
                                if 'fieldOfView' in line:
                                    webots_world_description_data[line_index] = line.replace(line.split()[1], str(horizontal_fov))
                                elif 'width' in line:
                                    webots_world_description_data[line_index] = line.replace(line.split()[1], str(cameras_image_width))
                                elif 'height' in line:
                                    webots_world_description_data[line_index] = line.replace(line.split()[1], str(cameras_image_height))
                                elif 'lens' in line:
                                    lens_node_set_parameters_count = 0
                                    indentation_level = line.count(' ')

                                    while '}' not in line:
                                        lens_node_set_parameters_count += 1
                                        line_index += 1
                                        line = webots_world_description_data[line_index]

                                    buffering_description_data(line_index)

                                    if lens_node_set_parameters_count > 0:
                                        line_index -= lens_node_set_parameters_count - 1

                                    webots_world_description_data[line_index] = f'{" " * indentation_level}center {distortion_center[0]} {distortion_center[1]}\n'
                                    line_index += 1

                                    webots_world_description_data[line_index] = f'{" " * indentation_level}radialCoefficients {radial_distortion[0]} {radial_distortion[1]}\n'
                                    line_index += 1

                                    webots_world_description_data[line_index] = f'{" " * indentation_level}tangentialCoefficients {tangential_distortion[0]} {tangential_distortion[1]}\n'
                                    line_index += 1

                                    unbuffering_description_data(line_index)
                                elif 'focus' in line:
                                    focus_node_set_parameters_count = 0
                                    indentation_level = line.count(' ')

                                    while '}' not in line:
                                        focus_node_set_parameters_count += 1
                                        line_index += 1
                                        line = webots_world_description_data[line_index]

                                    buffering_description_data(line_index)

                                    if focus_node_set_parameters_count > 0:
                                        line_index -= focus_node_set_parameters_count - 1

                                    webots_world_description_data[line_index] = f'{" " * indentation_level}focalLength {focal_length}\n'
                                    line_index += 1

                                    unbuffering_description_data(line_index)
                                else:
                                    found_camera_name = False
                            else:
                                for device_name in cameras_webots_settings['devices_name']:
                                    if found_rangefinder_name or found_camera_name:
                                        break
                                    elif f'name "{device_name}"' in line:
                                        line_index -= 1
                                        line = webots_world_description_data[line_index]

                                        if 'RangeFinder' in line:
                                            found_rangefinder_name = True
                                            line_index += 1

                                            continue

                                        found_camera_name = True
                                        line_index += 1

                            line_index += 1

                        try:
                            shutil.copy2(
                                os.path.join(
                                    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
                                    f'resource/worlds/{args[1]}'
                                ), 
                                os.path.join(
                                    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
                                    f'resource/worlds/backups/{time.strftime("%Y%m%d-%H%M%S")}_{args[1]}'
                                )
                            )

                            webots_world_wbt_file.seek(0)
                            webots_world_wbt_file.writelines(webots_world_description_data)
                            webots_world_wbt_file.truncate()

                            webots_world_description_data.clear()
                            webots_world_wbt_file.close()
                        except Exception as e:
                            print(''.join(traceback.TracebackException.from_exception(e).format()))
                    except yaml.YAMLError as e:
                        print(''.join(traceback.TracebackException.from_exception(e).format()))
            except Exception as e:
                print(''.join(traceback.TracebackException.from_exception(e).format()))
    else:
        print('Please specify the Webots .wbt world file as the second argument')

if __name__ == "__main__":
    main(args=sys.argv)
