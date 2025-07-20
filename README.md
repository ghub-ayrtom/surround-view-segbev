# Table of Contents

1. [Stack](#stack)
2. [Quick Overview](#quick-overview)
3. [Architecture](#architecture)
4. [Getting Started](#getting-started)
5. [Documentation (RU)](#documentation-ru)
6. [Questions and Feedback](#questions-and-feedback)
7. [Copyright and License](#copyright-and-license)
8. [Disclaimer](#disclaimer)

# Stack

![](https://github.com/user-attachments/assets/420f7102-9055-41ad-8494-42dadf1868ef)

Left to right, top to bottom:
- [Python](https://www.python.org/) 3.10.12
- [OpenCV](https://opencv.org/) 4.10.0
- [YOLO11](https://docs.ultralytics.com/models/yolo11/) nano from Ultralytics
- [FastSeg](https://github.com/ekzhang/fastseg)
- [ROS 2 Humble](https://docs.ros.org/en/humble/index.html)
- [Webots](https://cyberbotics.com/) R2023b
- [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox/tree/humble)
- [Nav2](https://github.com/ros-navigation/navigation2/tree/humble) + [AMCL](https://docs.nav2.org/configuration/packages/configuring-amcl.html)
- [pointcloud_to_laserscan](https://github.com/ros-perception/pointcloud_to_laserscan/tree/humble)<br>

System:
- Linux [Ubuntu 22.04.5 LTS (Jammy Jellyfish)](https://releases.ubuntu.com/jammy/) 64-bit
- Memory – 16.0 GiB
- Processor – AMD® Ryzen 7 4800hs with radeon graphics × 16
- Graphics – RENOIR (renoir, LLVM 15.0.7, DRM 3.57, 6.8.0-60-generic). It's been tough, so a GPU is highly desirable

# Quick Overview

![](https://github.com/user-attachments/assets/b35496bd-0ca5-4ea2-aee5-91f22537bcdd)
<div align="center">Emulation of real camera models</div>
<br>

![](https://github.com/user-attachments/assets/94e3dc2e-396b-4951-a5d5-498f19a3383a)
<div align="center">Automated camera calibration</div>
<br>

![](https://github.com/user-attachments/assets/ea44403d-93c9-43e6-b699-2d507fbafeab)
<div align="center">2D surround view system</div>
<br>

![](https://github.com/user-attachments/assets/413a9e88-2576-49be-a8c1-ee756e1ccff7)
<div align="center">Test site for debugging and testing</div>
<br>

![](https://github.com/user-attachments/assets/7a7ca0b8-681b-42b8-8886-700a7665eda0)

https://github.com/user-attachments/assets/c632befd-78a3-4d4d-93bc-35605c59779f
<div align="center">Segmented local map</div>
<br>

![](https://github.com/user-attachments/assets/00f32cdd-1fa9-4004-aa5a-c418a42f127c)

![](https://github.com/user-attachments/assets/eec0f3b2-f8f3-4007-af41-748935fa9668)
<div align="center">Merged point cloud from two lidars</div>
<br>

https://github.com/user-attachments/assets/0b3ab847-4bed-4641-91da-c8fb87b6acd1
<div align="center">WASD ego vehicle control and global mapping</div>
<br>

https://github.com/user-attachments/assets/8ca46624-f7cf-4f30-ae96-13ebaeda4784

https://github.com/user-attachments/assets/b75f0e0a-2d0e-4c79-865d-928f9d3c2b97
<div align="center">Autonomous navigation with Ackermann kinematics support</div>

# Architecture

![](https://github.com/user-attachments/assets/e82693e9-c97c-4a0d-a422-0b1b3b69b45e)
<div align="center">General</div>
<br>

![](https://github.com/user-attachments/assets/9e3067f6-b51d-4834-8dcf-4c8be2029439)
<div align="center">«Calibration Tool» module</div>
<br>

![](https://github.com/user-attachments/assets/296e5eeb-34d2-4b98-80b7-27952326e724)
<div align="center">«Surround View» module</div>
<br>

![](https://github.com/user-attachments/assets/d50420aa-93f8-4c89-bf7e-92d674eb1d47)
<div align="center">«SegBEV» module</div>
<br>

![](https://github.com/user-attachments/assets/3472a106-da02-4986-945a-af793d6c3ef8)
<div align="center">«Navigation» module</div>

# Getting Started

...

# Documentation (RU)

1. A brief excerpt in the form of an ![article](docs/1.pdf)
2. Full text in the form of an ![explanatory note](docs/2.pdf)

Sorry, foreign friends, but I'll go crazy trying to translate all of this stuff, so you'll have to figure it out yourselves.

# Questions and Feedback

You are welcome to submit questions and bug reports as [GitHub Issues](https://github.com/ghub-ayrtom/surround-view-segbev/issues).<br>
Feedback is accepted at the following email address: molchanovlive@gmail.com.

# Copyright and License

...

# Disclaimer

...
