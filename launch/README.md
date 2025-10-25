`cameras_calibration_launch.py` — launches the Webots world named ![waltz.wbt](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/resource/worlds/waltz.wbt), after which the chessboards automatically start moving along the specified trajectories, and the cameras take pictures of them. During this time, information about the calibration process can be found in the console:<br>

https://github.com/user-attachments/assets/a4011fa5-7f74-4248-8625-eb723cf11ecd

![](https://github.com/user-attachments/assets/7da9f2d7-845a-4b32-b098-46a7281a4e48)

![](https://github.com/user-attachments/assets/0749f95d-9a3a-4beb-8e62-3e6d810c890f)
<div align="center">Distorted (top) and resulting (bottom) images from the front camera</div>
<br>

```yaml
%YAML:1.0
---
image_resolution: !!opencv-matrix
...
data: [ 640, 1305, 4 ] # image_height, image_width, channels (depth)
camera_matrix: !!opencv-matrix
...
# fx, 0., cx, 0., fy, cy, ...
data: [ 456.88983054030831, 0., 652., 0., 456.88983054030831, 319.5,
0., 0., 1. ]
distortion_coefficients: !!opencv-matrix
...
# k1, k2, p1, p2, k3
data: [ 0.09343090353266352, -0.030199853852743441,
-0.0027583853029400809, 0.0047953056439332372, 0. ]
```
<div align="center">Rear camera parameters matched by OpenCV algorithms</div>
<br>

`mapping_launch.py` — launches the Webots world named ![main.wbt](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/resource/worlds/main.wbt), after which SLAM Toolbox starts building its map in OccupancyGrid format, which can be then saved via RViz for further reuse:<br>

https://github.com/user-attachments/assets/4b33e4ab-ae5c-40d3-afa1-d5e75aaf52a4

For manual WASD ego vehicle control during global mapping, you also need to switch `EGO_VEHICLE_CONTROL_MODE` in ![global_settings.py](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/surround_view_segbev/configs/global_settings.py) to `Manual` and, if necessary, change the speed and angle via `EGO_VEHICLE_MAX_SPEED` and `EGO_VEHICLE_MAX_STEERING_ANGLE` respectively. After that, in a separate active console run the control node as follows:

```bash
ros2 run surround_view_segbev ackermann_keyboard_teleop_node
```

If you want to use your robot, you may also need to replace the following stuff:

1. The full path to your robot description file, frame and topic names, and the description itself (it's better to create a new file) in ![EgoVehicle.urdf](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/resource/descriptions/EgoVehicle.urdf)
2. Frame and topic names in ![async_pointcloud_merge_node.cpp](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/pointcloud_preprocessing/src/async_pointcloud_merge_node.cpp) (This node merges point clouds from two lidars into one and publishes it in a separate topic) and ![ego_vehicle_odometry_node.py](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/surround_view_segbev/ego_vehicle_odometry_node.py) (GPS must be installed on your robot)
3. When using only one lidar, comment out the launch of `async_pointcloud_merge_node` in ![mapping_launch.py](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/launch/mapping_launch.py) and publish your point cloud to the `/cloud_in` topic, as required by the ![pointcloud_to_laserscan](https://github.com/ros-perception/pointcloud_to_laserscan/tree/humble) package

`simulation_launch.py` — launches the Webots world named ![main.wbt](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/resource/worlds/main.wbt), after which AMCL or SLAM Toolbox starts to localize on a pre-built global map (see `mapping_launch.py` description above), while the Nav2 stack loads it from a set of files saved via RViz, converts the map to Costmap 2D format and performs autonomous navigation based on it:<br>

https://github.com/user-attachments/assets/2337a577-0540-4e89-933c-3c676f22bc5e

https://github.com/user-attachments/assets/a9b2b8df-4dcf-4481-9f45-f23e54c0f1aa

> [!TIP]
> When mapping, it's recommended to save all your maps in `.../surround_view_segbev/configs/slam_toolbox/maps/<YOUR_MAP_FOLDER_NAME>`. In this case, you will need to replace the following paths:

![simulation_launch.py](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/launch/simulation_launch.py)

```python
map_yaml = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, f'{PACKAGE_NAME}/configs/slam_toolbox/maps/<YOUR_MAP_FOLDER_NAME>/<YOUR_MAP_NAME>.yaml'))
)
```

![nav2_params.yaml](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/surround_view_segbev/configs/nav2/nav2_params.yaml) (Don't forget about frame names in this file and in the adjacent ![nav_through_poses_bt.xml](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/surround_view_segbev/configs/nav2/nav_through_poses_bt.xml), if you have different ones)

```yaml
map_server:
  ros__parameters:
    yaml_filename: '.../surround_view_segbev/surround_view_segbev/configs/slam_toolbox/maps/<YOUR_MAP_FOLDER_NAME>/<YOUR_MAP_NAME>.yaml'
```

![mapper_params_online_async.yaml](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/surround_view_segbev/configs/slam_toolbox/mapper_params_online_async.yaml)

```yaml
slam_toolbox:
  ros__parameters:
    ...

    mode: mapping # localization ← Swap this and uncomment three lines below when using localization from SLAM Toolbox

    # map_file_name: '.../surround_view_segbev/configs/slam_toolbox/maps/<YOUR_MAP_FOLDER_NAME>/<YOUR_MAP_NAME>'
    # map_start_pose: [0.0, 5.75, -1.57]
    # map_start_at_dock: true

    ...
```

> [!IMPORTANT]
> 1. In ![nav2_path_planning_node.py](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/surround_view_segbev/nav2_path_planning_node.py) you can manually fill the route in map local coordinates that ego vehicle will follow while avoiding obstacles or comment out the launch of this node in ![simulation_launch.py](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/launch/simulation_launch.py), using only 2D Pose Estimate and 2D Goal Pose in RViz
> 2. When using localization from SLAM Toolbox, also do the following in `simulation_launch.py` and the opposite when using localization from AMCL (including rolling back the changes in `mapper_params_online_async.yaml` above):

```python
...

lifecycle_nodes = [
    'map_server', 
    # 'amcl', ← Comment this
    'controller_server', 
    'smoother_server', 
    'planner_server', 
    'behavior_server', 
    'bt_navigator', 
    'waypoint_follower', 
    'velocity_smoother', 
]

...

# ↓ Comment this

# Node(
#     executable='amcl', 
#     package='nav2_amcl', 
#     name='amcl', 
#     parameters=[configured_params], 
#     remappings=remappings, 
#     arguments=['--ros-args', '--log-level', log_level], 
#     respawn=use_respawn, 
#     respawn_delay=2.0, 
#     output='screen', 
# ), 

...

# ↓ Uncomment this

LifecycleNode(
    executable='localization_slam_toolbox_node', 
    package='slam_toolbox', 
    name='slam_toolbox', 
    namespace='', 
    parameters=[
        mapper_params_online_async_yaml, 
        {
            'use_sim_time': USE_SIM_TIME, 
            'use_lifecycle_manager': False, 
        }, 
    ], 
    output='screen', 
), 

...

# ↓ Comment this

# ComposableNode(
#     package='nav2_amcl', 
#     plugin='nav2_amcl::AmclNode', 
#     name='amcl', 
#     parameters=[configured_params], 
#     remappings=remappings, 
# ), 

...
```

`surround_view_launch.py` — ...
