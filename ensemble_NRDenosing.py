import os
import numpy as np
import glob
import cv2
import logging

# DAT+HATL+MambaIRv2


dir1 = r"NTIRE2025_Challenge/results/06_NAFNetLocal/test"
dir2 = r"NTIRE2025_Challenge/results/06_RCAN/test"

weights = [0.95, 0.05]


output_dir = "NTIRE2025_Challenge/results/06_NRDenosing_ensemble/test"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


print("weights: {}".format(weights))

img_path = glob.glob(dir1+"/*.png")


for p in img_path:
    name = os.path.basename(p)

    path1 = p
    path2 = os.path.join(dir2, name)

    img1 = cv2.imread(path1, cv2.IMREAD_COLOR).astype(np.float32)
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR).astype(np.float32)

    img_mean = img1*weights[0] + img2*weights[1]

    img_mean = img_mean.round().astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, name), img_mean)



def all(dir, ends):
    list = os.listdir(dir)
    for l in list:
        if l.endswith(ends):
            os.remove(os.path.join(dir, l))
    print("删除成功")

# all(output_dir, ".png")