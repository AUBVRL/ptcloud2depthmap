import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from bilinear_interpolation import bilinear_interpolation
import plyfile
from plyfile import PlyData, PlyElement
import open3d
import os,glob
# from trilinear_interpolation import trilinear_interpolation

def extract_number(file_path):
    return int(''.join(filter(str.isdigit, os.path.basename(file_path).split()[-1])))

# Load binary point cloud
image_height = 720
image_width = 1280

# folder_path = '/home/omar/Desktop/next_ficosa/FICOSA/20230426_110200/pointclouds_rgb/'
# folder_path = '/home/omar/Desktop/FICOSA/20230426_110310/velodyne_points'
folder_path = '/home/omar/khara'
index = 0
for file in sorted(glob.glob(os.path.join(folder_path, '*')), key=extract_number):
    filename = '{:06d}'.format(index)
    print(filename)
    ################### read xyz points 
    with open(file,'r') as f:
            data = f.readlines()
            points =  [list(map(float, line.split())) for line in data]
            points = np.asarray(points, dtype=np.float32)

    ################### read bin files
    # bin_pcd = np.fromfile(path2, dtype=np.float32)
    # # print(bin_pcd.shape)
    # # Reshape and drop reflection values
    # points = bin_pcd.reshape((-1, 4))[:, 0:3]
    # print(points.shape)
    #print(bin_pcd[:10])
    ####################

    #################### read ply data
    # plydata = PlyData.read(file)
    # vertex_data = plydata['vertex']
    # x = vertex_data['x']
    # y = vertex_data['y']
    # z = vertex_data['z']
    # points = []
    # for i in range(len(x)):
    #     points.append([x[i], y[i], z[i]])
    # points = np.asarray(points, dtype=np.float32)
    ####################

    #################### read pcd data
    # point_cloud = open3d.io.read_point_cloud(file) 
    # points = np.asarray(point_cloud.points, dtype=np.float32)
    ####################



    print(points.shape)

    #sys.exit()
    # translation_vector = np.array([0., 0.0175, 0.13035])
    translation_vector = np.array([0., 0., 0.])
    # translation_vector = np.array([0.02, 0.0175, -0.094])
    #0.235
    #130.35


    theta = -0.5625

    alignment_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    src_rot = np.array([[0., -1., 0.], 
                        [0., 0. , -1.], 
                        [1., 0. , 0.]])

    # src_rot = np.dot(src_rot, alignment_rotation)

    # src_rot = np.array([[0., -1., 0.], 
    #                     [0., 0. , -1.], 
    #                     [1., 0. , 0.]])

    # src_transform = np.array([[2.347736981471e-04, -9.999441545438e-01, -1.056347781105e-02, -2.796816941295e-03], 
    #                           [1.044940741659e-02, 1.056535364138e-02 , -9.998895741176e-01, -7.510879138296e-02], 
    #                           [9.999453885620e-01, 1.243653783865e-04 , 1.045130299567e-02 , -2.721327964059e-01],
    #                           [0                 , 0                  , 0                  , 1                  ]])

    rotation_vector = cv2.Rodrigues(src_rot)[0]
    #print(rotation_vector)

    # Apply transformation to points
    transformed_points = np.dot(points, src_rot.T) + translation_vector
    print("transformed points: ", transformed_points)


    # virtual/real camera intrinsics
    # camera_matrix = np.array([[523.2268676757812, 0, 643.6303100585938], 
    #                           [0, 523.2268676757812, 355.9408874511719], 
    #                           [0, 0, 1]])

    camera_matrix = np.array([[533.91, 0, 640.285], 
                            [0, 534.11, 352.944], 
                            [0, 0, 1]])

    # camera_matrix = np.array([[261.3557434082031, 0, 337.3840637207031], 
    #                           [0, 261.3557434082031, 185.36297607421875], 
    #                           [0, 0, 1]])

    # distCoeffs=np.array([-0.0687264, 0.0541945, -0.000296994, 0.000296994, -0.0230035])
    distCoeffs=np.array([-1.49749994, 2.96620011, -1.63232005e-04, 2.24468997e-04, 6.42248988e-02, -1.4073798, 2.81257010, 2.37092003e-01])
    # 
    # distCoeffs=np.zeros((1,14))

    cv_points = points.astype(np.float32)
    #print(cv_points[:, :])
    # Project the 3D points onto the image plane
    image_points, _ = cv2.projectPoints(cv_points, rotation_vector, translation_vector, camera_matrix, distCoeffs)

    #sys.exit()

    # Initialize the depth image with NaN or 0 values
    depth_image = np.full((image_height, image_width), 0.) #np.nan
    disparity_image = np.full((image_height, image_width), 0.) #np.nan

    print(image_points.shape)
    negatives = 0
    positives = 0

    focal_length = 7.215377000000e+02
    # baseline = 1.6 # for disparity calculation, this is the value for the KITTI dataset

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
                # disparity_image[int(v), int(u)] = baseline * focal_length / depth
            else:
                depth_image[int(v), int(u)] = min(depth*1000, depth_image[int(v), int(u)])
                # disparity_image[int(v), int(u)] = baseline * focal_length / depth_image[int(v), int(u)] # TODO: douaa check if this is zero
    #print(depth_image[225, 825])
    print(negatives, positives, negatives + positives)
    dense_depth_image = bilinear_interpolation(depth_image)

    cv2.imwrite(f"/home/omar/Desktop/research/Data/Lab_Lidar/{filename}.tiff", dense_depth_image)
    index += 1
