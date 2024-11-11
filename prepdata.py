import os
from PIL import Image
import cv2
import numpy as np

dir="./datasets/openpit/train/"
dir_A=os.path.join(dir,"A")
dir_B=os.path.join(dir,"B")
dir_gt=os.path.join(dir,"gt")

new_dir="./datasets/openpit_png/train/"
new_dir_A=os.path.join(new_dir,"A")
new_dir_B=os.path.join(new_dir,"B")
new_dir_gt=os.path.join(new_dir,"gt")

os.makedirs(new_dir_A,exist_ok=True)
os.makedirs(new_dir_B,exist_ok=True)
os.makedirs(new_dir_gt,exist_ok=True)

im_name_list=os.listdir(dir_A)
file_len=len(im_name_list)

for i in range(file_len):
    im_name=im_name_list[i]
    gt_name=im_name.replace("image","label")
    img_A=cv2.imread(os.path.join(dir_A,im_name))
    img_B=cv2.imread(os.path.join(dir_B,im_name))
    gt=cv2.imread(os.path.join(dir_gt,gt_name),cv2.IMREAD_UNCHANGED).astype(np.uint8)
    gt[gt==1]=255
    print(os.path.join(dir_gt,gt_name))
    print(gt)
    cv2.imwrite(os.path.join(new_dir_A,im_name.replace("tif","png")),img_A)
    cv2.imwrite(os.path.join(new_dir_B,im_name.replace("tif","png")),img_B)
    cv2.imwrite(os.path.join(new_dir_gt,gt_name.replace("tif","png").replace("label","image")),gt)
    

