import cv2
import matplotlib.pyplot as plt
import numpy as np

## Load depth images
# test = 5

## change paths of depthZED and DepthLidar to whatever paths you want 
DepthLIDAR = cv2.imread(f"/media/omar/7400-56BD/depth_images/000053.png", cv2.IMREAD_ANYDEPTH)


# DepthLIDAR = cv2.resize(DepthLIDAR, (int(640/2), int(360/2)))
# ImgZED = cv2.resize(ImgZED, (int(640/2), int(360/2)))

min_depth_lidar = np.min(DepthLIDAR)
max_depth_lidar = np.max(DepthLIDAR)



## Create a colormap similar to 'jet' colormap
cmap = plt.get_cmap('jet')

plt.figure()
plt.imshow(DepthLIDAR, cmap=cmap, vmin=min_depth_lidar, vmax=max_depth_lidar)
plt.colorbar()
plt.title('Lidar Depth Image Visualization')
###############################################################

# DepthZED = cv2.imread(f"/home/omar/Desktop/research/Tests/Test{test}/depth_images/000009.tiff", cv2.IMREAD_ANYDEPTH)
# ImgZED = cv2.imread(f"/home/omar/Desktop/research/Tests/Test{test}/images/000010.png")

# DepthZED = DepthZED/1000
# mask = DepthZED > max_depth_lidar
# DepthZED[mask] = 0


# min_depth = np.min(DepthZED)
# max_depth = np.max(DepthZED)

# Difference_depth = DepthLIDAR - DepthZED

################################################################ one combined map
# plt.imshow(ImgZED, alpha=0.5)
# # plt.imshow(DepthZED, cmap=cmap, vmin=min_depth, vmax=max_depth_lidar, alpha=0.5)
# plt.imshow(DepthLIDAR, cmap=cmap, vmin=min_depth, vmax=max_depth, alpha=0.5)
# plt.colorbar()
# plt.title('Lidar Image Visualization')
# plt.show()
################################################################

################################################################ 2 seperate depth maps
# plt.figure(1)
# # # plt.subplot(2,1,1)
# plt.imshow(DepthZED, cmap=cmap, vmin=min_depth_lidar, vmax=max_depth_lidar)
# plt.colorbar()
# plt.title('ZED Depth Image Visualization')
# plt.figure(2)
# # # plt.subplot(2 ,1,2)
# plt.imshow(DepthLIDAR, cmap=cmap, vmin=min_depth_lidar, vmax=max_depth_lidar)
# plt.colorbar()
# plt.title('Lidar Depth Image Visualization')

# plt.figure(3)
# plt.imshow(ImgZED)
# plt.title('RGB Image')

# plt.figure(4)
# plt.imshow(Difference_depth, cmap=cmap, vmin=min_depth_lidar, vmax=max_depth_lidar)
# plt.colorbar()
# plt.title('Depth Map Difference')

################################################################

plt.show()