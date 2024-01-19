import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from bilinear_interpolation import bilinear_interpolation
import plyfile
from plyfile import PlyData, PlyElement
import open3d
import os



# write path to the pointcloud that you want to read
Path_to_pointcloud = '/home/omar/Desktop/research/pointclouds_from_ros/Test5'

# supported filetypes are 'txt' or 'text', 'bin' (binary), 'ply', and 'pcd'
file_type = 'pcd'

# path and name of the depth image to be saved
Depth_Folder_Path = "/home/omar/Desktop/research/Tests/Depth_maps/"
Depth_File_Name = "Test3_.tiff"

################### read points from text file
if file_type == 'text' or file_type == 'txt': 
    with open(Path_to_pointcloud,'r') as f:
            data = f.readlines()
            points =  [list(map(float, line.split())) for line in data]
            points = np.asarray(points, dtype=np.float32)

################### read bin files
if file_type == 'bin':
    bin_pcd = np.fromfile(Path_to_pointcloud, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    print(points.shape)
    print(bin_pcd[:10])
####################

#################### read ply data
if file_type == 'ply':
    plydata = PlyData.read(Path_to_pointcloud)
    vertex_data = plydata['vertex']
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    points = []
    for i in range(len(x)):
        points.append([x[i], y[i], z[i]])
    points = np.asarray(points, dtype=np.float32)
####################

#################### read pcd data
if file_type == 'pcd':
    point_cloud = open3d.io.read_point_cloud(Path_to_pointcloud) 
    points = np.asarray(point_cloud.points, dtype=np.float32)
####################



print(points.shape)

#sys.exit()
translation_vector = np.array([0., 0.0175, 0.13035])

theta = -0.5625 
# theta = -1+np.pi/2

alignment_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])

src_rot = np.array([[0., -1., 0.], 
                    [0., 0. , -1.], 
                    [1., 0. , 0.]])

src_rot = np.dot(src_rot, alignment_rotation)

rotation_vector = cv2.Rodrigues(src_rot)[0]

# Apply transformation to points
transformed_points = np.dot(points, src_rot.T) + translation_vector
print("transformed points: ", transformed_points)


# virtual/real camera intrinsics
## Zed 2i
camera_matrix = np.array([[523.2268676757812, 0, 643.6303100585938], 
                          [0, 523.2268676757812, 355.9408874511719], 
                          [0, 0, 1]])

# Set Image Resolution
image_height = 720
image_width = 1280

# distCoeffs=np.array([-1.49749994, 2.96620011, -1.63232005e-04, 2.24468997e-04, 6.42248988e-02, -1.4073798, 2.81257010, 2.37092003e-01])

distCoeffs=np.zeros((1,14))

cv_points = points.astype(np.float32)

# Project the 3D points onto the image plane
image_points, _ = cv2.projectPoints(cv_points, rotation_vector, translation_vector, camera_matrix, distCoeffs)

#sys.exit()

# Initialize the depth image with NaN or 0 values
depth_image = np.full((image_height, image_width), 0.) #np.nan
disparity_image = np.full((image_height, image_width), 0.) #np.nan

print(image_points.shape)
negatives = 0
positives = 0

focal_length = 523.2268676757812
baseline = 1.6 # for disparity calculation, this is the value for the KITTI dataset

# Populate the depth image with the depths of the projected points
for i in range(cv_points.shape[0]):
    u, v = image_points[i, 0]

    depth = transformed_points[i, 2]
    #print(u, v, depth)
    if 0 <= u < image_width and 0 <= v < image_height:
        if depth < 0:
            #print("Negative!", u, v, depth)
            negatives = negatives + 1
            continue
        #print("Positive!", u, v, depth)
        positives = positives + 1
        if depth_image[int(v), int(u)] == 0:
            depth_image[int(v), int(u)] = depth
            disparity_image[int(v), int(u)] = baseline * focal_length / depth
        else:
            depth_image[int(v), int(u)] = min(depth*1000, depth_image[int(v), int(u)])
            disparity_image[int(v), int(u)] = baseline * focal_length / depth_image[int(v), int(u)] # TODO: douaa check if this is zero

print(negatives, positives, negatives + positives)

dense_depth_image = bilinear_interpolation(depth_image)

# Display the depth image
custom_cmap = plt.cm.get_cmap('jet').copy()

# Set the range of depth values to display (you can adjust these values based on your data)
min_depth = np.min(depth_image)
max_depth = np.max(depth_image)

plt.figure()
# Display the depth image with the custom colormap and vmin/vmax settings
plt.subplot(2, 1, 1)
plt.imshow(depth_image, cmap=custom_cmap, vmin=min_depth, vmax=max_depth)

plt.colorbar()
plt.title('Estimated Depth Image From Lidar Scan')
################################################################################
plt.subplot(2, 1, 2)
min_depth = np.min(dense_depth_image)
max_depth = np.max(dense_depth_image)
plt.imshow(dense_depth_image, cmap=custom_cmap, vmin=min_depth, vmax=max_depth)
plt.colorbar()
plt.title("after interpolation")
plt.show()

# save image
cv2.imwrite(os.path.join(Depth_Folder_Path, Depth_File_Name), dense_depth_image)
