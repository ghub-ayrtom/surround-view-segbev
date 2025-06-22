SIMULATION_TIME_STEP = 10  # Шаг (скорость) симуляции в Webots

# Название папки из ".../surround_view_segbev/configs/cameras/" с характеристиками используемой модели видеокамеры
USED_CAMERA_MODEL_FOLDER_NAME = 'ZED_2'

EGO_VEHICLE_MAX_SPEED = 7.0            # 3.0 / Км/ч
EGO_VEHICLE_MAX_STEERING_ANGLE = 35.0  # Градусы

'''
    'Manual' - ручное управление с клавиатуры
    'Auto'   - автономное управление алгоритмами
'''
EGO_VEHICLE_CONTROL_MODE = 'Auto'

# Название папок из ".../surround_view_segbev/models/" с весами моделей нейронных сетей глубокого обучения
USED_DETECTOR_FOLDER_NAME = 'YOLO11'

'''
    'FastSeg'    - обучена на перспективных изображениях с каждой из 5-ти камер
    'FastSegBEV' - обучена на единых изображениях локальной карты с видом сверху
'''
USED_SEGMENTOR_FOLDER_NAME = 'FastSegBEV'
