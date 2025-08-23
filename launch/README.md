`cameras_calibration_launch.py` — launches the Webots world called ![waltz.wbt](https://github.com/ghub-ayrtom/surround-view-segbev/blob/main/resource/worlds/waltz.wbt), after which the chessboards automatically start moving along the specified trajectories, and the cameras take pictures of them. During this time, information about the calibration process can be found in the console:<br>

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

`mapping_launch.py` — ...<br>

`simulation_launch.py` — ...<br>

`surround_view_launch.py` — ...
