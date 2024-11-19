import os
from setuptools import setup


PACKAGE_NAME = 'surround_view_segbev'


def generate_data_files(share_path, data_files_path):
    data_files_temp = []

    for path, _, files in os.walk(data_files_path):
        data_file = (
            os.path.dirname(os.path.dirname(share_path)) + '/' + path, [
                os.path.join(path, file) for file in files if not file.startswith('.')
            ]
        )

        data_files_temp.append(data_file)

    return data_files_temp


data_files = []

data_files.append(('share/ament_index/resource_index/packages', ['resource/' + PACKAGE_NAME]))
data_files.append(('share/' + PACKAGE_NAME, ['package.xml']))

data_files += generate_data_files('share/' + PACKAGE_NAME + '/configs/', 'configs/')
data_files += generate_data_files('share/' + PACKAGE_NAME + '/launch/', 'launch/')
data_files += generate_data_files('share/' + PACKAGE_NAME + '/resource/', 'resource/')


setup(
    name=PACKAGE_NAME, 
    version='0.0.1', 
    packages=[PACKAGE_NAME], 
    data_files=data_files, 
    install_requires=['setuptools'], 
    zip_safe=True, 
    maintainer='Molchanov Artyom', 
    maintainer_email='molchanovlive@gmail.com', 
    description='Segmented local map for self-driving vehicles path planning based on a two-dimensional (2D) surround view system', 
    license='No license', 
    tests_require=['pytest'], 
    entry_points={
        'console_scripts': [
            'ackermann_keyboard_teleop_node = surround_view_segbev.ackermann_keyboard_teleop_node:main', 
            'chessboards_controller_node = surround_view_segbev.chessboards_controller_node:main', 
            'projection_weight_matrices_node = surround_view_segbev.projection_weight_matrices_node:main', 
            'surround_view_node = surround_view_segbev.surround_view_node:main', 
        ], 
    }, 
)
