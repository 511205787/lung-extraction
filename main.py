import pydicom
import numpy as np
import cv2
import os
import nibabel as nib
import extract_lung
from matplotlib import pyplot as plt

path = "E:/yjssjtu/image process/datasets/case_1"  # 待读取的文件夹
path_list = os.listdir(path)  # 获取指定路径的内容

path_list.sort(key=lambda x: int(x[-7:-4]))  # 对读取的路径进行排序

Level = -650  # Window Level 指定肺部CT的窗
Window = 1500  # Window Width
min_bound = Level - Window // 2
max_bound = Level + Window // 2
output = np.zeros((len(path_list), 512, 512))

z = 0
for filename in path_list:
    filepath = os.path.join(path, filename)
    ds = pydicom.dcmread(filepath)  # 读取dcm文件
    image = ds.pixel_array.astype(np.int16)
    intercept = ds.RescaleIntercept
    image += np.int16(intercept)
    image = (image - min_bound) / (max_bound - min_bound) * 255
    image[image > 255] = 255
    image[image < 0] = 0
    output[z, :, :] = image
    z = z + 1
    # -------------------------------------------------------------------
    #                               保存图像
    # -------------------------------------------------------------------
    # # res = np.array(image, dtype=np.uint8)
    # directory = r'G:\pythonProject\image'
    # os.chdir(directory)
    # filename = filename[:-4]
    # cv2.imwrite(f"{filename}.jpg", image)

# -------------------------------------------------------------------
#                         切割肺实质
# -------------------------------------------------------------------
extract_lung.raw2mask()

# -------------------------------------------------------------------
#                         腐蚀表皮
# -------------------------------------------------------------------
# extract_lung.erode_lung()

# directory = r'G:\test_outcome'  # 保存路径
# os.chdir(directory)
# -------------------------------------------------------------------
#                         将结果矩阵转化为nii格式
# -------------------------------------------------------------------
# new_image = nib.Nifti1Image(output, np.eye(4))
# nib.save(new_image, 'nifti.nii.gz')
