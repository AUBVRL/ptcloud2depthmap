# Depth Image Estimation From LiDAR Scans

This work was performed under the [AUB Vision & Robotics Lab](https://sites.aub.edu.lb/vrlab/).

# Installation

## Hardware Requirements

- A LiDAR (tested on velodyne VLP16, but works on other LiDARs)
- A computer

## Software Requirements and Procedure

1. Install OpenCv2 with python: `pip3 install opencv-python`
2. Install Open3d with python: `pip3 install open3d`
3. Install plyfile with python: `pip3 install plyfile`
4. Install numpy and matplotlib: `pip3 install numpy matplotlib` 

# Running

## Generate Single Depth Image 

Open `generate_depths_pc.py` and edit the paths as instructed in the code. The variables that need changing are: 
1. `Path_to_pointcloud` The path to the pointcloud saved on your computer
2. `file_type` The name of the file extension. The purpose of this is to be able to properly read the pointcloud if it was in different formats. Supported types so far are `'txt', 'bin', 'ply', 'pcd'`
3. `Depth_Folder_Path` The path to the folder you want to save the depth image in. 
4. `Depth_File_Name` The name of the file you want to save. Don't forget to include the extension `.tiff` in the naming of the file.

### Optional
Currently, the virtual camera that is being mimicked is the ZED2i camera. To change this, you need to change the camera paramaters: 
1. `camera_matrix` the camera intrinsic matrix. It should be a numpy array and in the form
$$
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1 \\
\end{bmatrix}
$$
2. `image_height` and `image_width` the resolution of the camera
3. `distCoeffs` the distortion coefficient matrix. This can be left as an array of zeros
4. If the camera is also real and you want to compare the estimated depth map to the real depth map, then you will need the transformation from the LiDAR to the camera. `translation_vector` is the translation component and `src_rot` is the rotational component. These parameters set our virtual camera anywhere in the pointcloud and estimates what the image would look like had the camera been physically there.

### Run
Open the terminal and go to the folder of the codes. then run `python3 generate_depths_pc`. A preview of the depth image will show in a popup window once the code finishes, the file will be saved after closing the popped window.

## Visualising Depth Images
open `Visualise_Depth_Images.py` and change the parameter in `DepthLIDAR` to the path to the depth map you want to visualise. Then run the code in terminal to view the depth map.

### Compare Virtual Depth Image to Real Depth Image
If you decided to compare the estimated depth image to the real depth image, then you should uncomment the second section of the code based on how you want the images displayed. Additionally, change the paths in `DepthZED` and `ImgZED` to the paths of the depth image and the rgb image respectively.

## Generate Multiple Depth Maps
This section is for generating multiple depth maps from multiple LiDAR scans. 
1. open `generate_depths_loop.py`, and repeat the same process as in the [single generation](#generate-single-depth-image), but instead of writing the path to the pointcloud, write the path to the folder containing all the pointclouds. 
2. The LiDAR scans are assumed to be an integer followed by the extension. If the name is an extension or if it is an index works fine.
3. The files are then saved as "000000.tiff" and increases iteratively

## Save Depth Images as Color Images
Open `Save_Depth_images.py` and change the following parameters:
1. `Folder_path` the path where the depth images are saved in, typically same path used in [generating depth maps](#generate-multiple-depth-maps)
2. `color_bar_dynamic` Set this to `True` if you want the color map to be dynamic and always be set as the minimum and maximum of each depth map. Set it to `False` if you want a set minimum and maximum for each depth image, and change `min_depth` and `max_depth` as suitable
3. `Save_Folder_Path` the path where the images will be saved as "000000.png" and increases iteratively

## Create A Video
Open `Save_Depth_images.py` and change the following parameters:
1. `Folder_path` the path where the images are saved in
2. `FPS` The video's frame per second

Currently, the video is saved in MP4 format. 