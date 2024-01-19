import numpy as np
import glob, os
import cv2
 
def extract_number(file_path):
    return int(os.path.basename(file_path).split('.')[0])

img_array = []
# '/home/omar/Desktop/research/Data/intel/*.png'
# '/home/omar/Desktop/next_ficosa/FICOSA/20230426_110200/compressed_images/0'   
for filename in sorted(glob.glob('/home/omar/Desktop/research/Data/Lab_ZED2/*.png'), key = extract_number):
    img = cv2.imread(filename)
    if img is None:
        print("Image is None, please check")
        print(filename)
        continue
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('/home/omar/Desktop/research/videos/upload/Zed_lab.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()