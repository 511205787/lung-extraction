import pydicom
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
import cv2


# Standardize the pixel values
def make_lungmask(img, display=False):
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([3, 3]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    # mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')

        plt.show()
    return mask


def erode_lung():
    path = "G:/pythonProject/outcome"  # 待读取的文件夹
    path_list = os.listdir(path)  # 获取指定路径的内容
    path_list.sort(key=lambda x: int(x[-7:-4]))  # 对读取的路径进行排序
    for filename in path_list:
        filepath = os.path.join(path, filename)
        image = cv2.imread(filepath)
        kernel = np.ones((6, 6), np.uint8)
        erosion = cv2.erode(image, kernel, iterations=1)
        directory = r'G:\pythonProject\erode_results'  # 保存路径
        os.chdir(directory)
        cv2.imwrite(f"{filename}", erosion)


def raw2mask():
    int_path = r"G:\pythonProject\image"
    i = 0
    for root, dirs, files in os.walk(int_path):
        for filename in files:  # 遍历所有文件
            i += 1
            print(filename)
            path = os.path.join(root, filename)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("32",img)
            mask = make_lungmask(gray, display=False)
            # Img = np.hstack((gray, mask * gray))     # 对比前后
            Img = mask * gray
            directory = r'G:\pythonProject\outcome'
            os.chdir(directory)
            cv2.imwrite(f"{filename}", Img)
