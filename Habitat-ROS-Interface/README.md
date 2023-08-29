# Habitat ROS Interface

The Interface connects the Habitat platform and the ROS environments to generalize the VSN policies to more realistic scenarios. The interface adopts the client-server construct, the client maintains a practical robot in the gazebo2 simulator and the server runs our HSPNav methods. We put the robot in the Autonomous Exploration Development Environment Developed by Carnegie Mellon University.

## Installation

1. Install Autonomous Exploration Development Environment 

   ```
   sudo apt update
   sudo apt install libusb-dev
   
   git clone https://github.com/HongbiaoZ/autonomous_exploration_development_environment.git
   
   cd autonomous_exploration_development_environment
   git checkout distribution-matterport && catkin_make
   ```

2.  Prepare Environment Model follow instructions [here](https://github.com/kangjiaxu639/HSPNav/blob/master/Habitat-ROS-Interface/docs/matterport3d_environment_setup_notes.pdf).

3. Install client

   ```
   # create ROS workspace
   mkdir -p ~/semantic_ws/src
   cd ~/semantic_ws/src/
   catkin_init_workspace
   
   # Move the repo into the workspace
   mv /HSPNav/Habitat-ROS-Interface /semantic_ws/src
   
   # Install dependencies
   rosdep install semantic_slam
   
   # make
   catkin_make
   ```

## Run the Interface with Habitat and ROS

1. Start the robot in ROS

   ```
   cd path/to/autonomous_exploration_development_environment
   source devel/setup.bash
   roslaunch vehicle_simulator system_matterport.launch
   ```

2. Start the Server(assuming with the method HSPNav)

   ```
   cd HSPNav/simulation
   python simulatior.py
   ```

3. Start the clinet

   ```
   cd path/to/semantic_ws
   source devel/setup.bash
   roslaunch semantic_slam semantic_mapping.launch
   ```

## Configuration

You can change parameters for launch. Parameters are in `./semantic_slam/params` folder.

***Note that you can set octomap/tree_type and semantic_cloud/point_type to 0 to generate a map with rgb color without doing semantic segmantation.***

### Parameters for octomap_generator node (octomap_generator.yaml)

namespace octomap

- pointcloud_topic
  - Topic of input point cloud topic
- tree_type
  - OcTree type. 0 for ColorOcTree, 1 for SemanticsOcTree using max fusion (keep the most confident), 2 for SemanticsOcTree using bayesian fusion (fuse top 3 most confident semantic colors). 
- world_frame_id
  -  Frame id of world frame.
- resolution
  - Resolution of octomap, in meters.
- max_range
  - Maximum distance of a point from camera to be inserted into octomap, in meters.
- raycast_range
  - Maximum distance of a point from camera be perform raycasting to clear free space, in meters.
- clamping_thres_min
  - Octomap parameter, minimum octree node occupancy during update.   
- clamping_thres_max
  -  Octomap parameter, maximum octree node occupancy during update.
- occupancy_thres
  - Octomap parameter, octree node occupancy to be considered as occupied
- prob_hit
  - Octomap parameter, hitting probability of the sensor model.
- prob_miss
  - Octomap parameter, missing probability of the sensor model.
- save_path
  - Octomap saving path. (not tested)

### Parameters for semantic_cloud node (semantic_cloud.yaml)

namespace camera

- fx, fy, cx, cy
  -  Camera intrinsic matrix parameters.
- width, height
  -  Image size.

namespace semantic_pcl

- color_image_topic
  - Topic for input color image.
- depth_image_topic
  - Topic for input depth image.
- point_type
  - Point cloud type, should be same as octomap/tree_type. 0 for color point cloud, 1 for semantic point cloud including top 3 most confident semanic colors and their confidences, 2 for semantic including most confident semantic color and its confident. 
- frame_id
  - Point cloud frame id.
- dataset
  - Dataset on which PSPNet is trained. "ade20k" or "sunrgbd".
- model_path
  - Path to pytorch trained model.