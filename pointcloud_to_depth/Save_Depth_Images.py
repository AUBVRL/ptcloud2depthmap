import cv2
import matplotlib.pyplot as plt
import numpy as np
import os,glob

def extract_number(file_path):
    return int(os.path.basename(file_path).split('.')[0])


# files_img = sorted(glob.glob('/home/omar/Desktop/next_ficosa/FICOSA/20230426_110200/compressed_images/0/fisheye_better_nbrs/*.png'), key = extract_number)

# folder_path = "/home/omar/Desktop/research/Data/depth_images/"
folder_path = "/media/omar/7400-56BD/New folder/depth_images/"
index = 1
for i, file in enumerate(sorted(glob.glob(os.path.join(folder_path, '*.png')), key=extract_number)):
    # print(file)
    if i < -1:
        index += 1
        continue
    DepthLIDAR = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
    DepthLIDAR = DepthLIDAR/1000
    # DepthLIDAR = np.flipud(np.fliplr(DepthLIDAR))
    # img = cv2.imread(files_img[i])
    min_depth = 0
    # max_depth = np.max(DepthLIDAR)
    max_depth = 7
    id = '{:06d}'.format(index)
   
    cmap = plt.get_cmap('jet')
    plt.imshow(DepthLIDAR, cmap=cmap, vmin=min_depth, vmax=max_depth)
    plt.colorbar()

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # ax1.axis('off')  # Turn off axis

    # ax2.imshow(DepthLIDAR, cmap=cmap)
    # ax2.axis('off')
    # plt.tight_layout()


    plt.savefig(f'/home/omar/Desktop/research/Data/Lab_ZED2/{id}.png')
    plt.close()
    index += 1


    


