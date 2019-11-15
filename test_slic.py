# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import time
from sklearn.decomposition import NMF
import pywt
from sklearn.cluster import DBSCAN
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from pyhdf.SD import SD, SDC
from libtiff import TIFF
import gdal
from gdalconst import *
from sklearn.mixture import GaussianMixture


#%%
def writeImage(bands, path):
    if bands is None:
        return
    else:
        # 认为各波段大小相等，所以以第一波段信息作为保存
        if bands.ndim == 2:
            bands = bands[:, :, None]
        # 设置影像保存大小、波段数
        band1 = bands[:, :, 0]
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.shape[2]

        # 创建文件
        # 先创建驱动，再创建相应的栅格数据集
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            path, img_width, img_height, num_bands, gdal.GDT_Int16)
        if dataset is not None:
            for i in range(num_bands):
                dataset.GetRasterBand(i + 1).WriteArray(bands[:, :, i])
        print("save image success.")

# %%
imgFile = 'D:\Code\cloud\MOD09GA\MYD09GA.A2019229.h27v05.006.2019231025638.tiff'
# imgFile = 'D:\Code\cloud\MOD09GA\MOD09GA.A2019207.h28v05.006.2019209031253.tiff'
validFlag = np.ones((2400, 2400), np.bool)
data = np.zeros((2400, 2400, 7), np.int16)
ds = gdal.Open(imgFile)
for i in range(7):
    bi = ds.GetRasterBand(i+1).ReadAsArray()
    print(np.sum(bi < -100))
    data[:, :, i] = np.clip(bi, -100, 10000)
    if i == 3:
        data[:, :, i] = np.where(bi < -100, 10000, data[:, :, i])
        validFlag = np.where((bi >= -100) & (bi < 10000), True, False)
    elif i == 5:
        data[:, :, i] = np.where(bi < -100, 10, data[:, :, i])

#%%
image = img_as_float(data[:, :, :3])
image = np.clip(image, 0, 0.3) / 0.3
numSegments = 1000
# apply SLIC and extract (approximately) the supplied number of segments
segments = slic(image, n_segments = numSegments, compactness=5.0, sigma = 5)

img = mark_boundaries(image, segments)
# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(img)
plt.axis("off")

# show the plots
plt.show()
writeImage(np.int16(img * 1e4), './test/slic_1.tiff')

# %%
num = np.max(segments) + 1
meanData = np.zeros((num, 7))
for i in range(num):
    idx = np.where(segments == i)
    tmp = data[idx[0], idx[1]]
    meanData[i] = np.mean(tmp, 0)

clf = GaussianMixture(3)
clf.fit(meanData)
pred = clf.predict(meanData)

# pred = (meanData[:, 0] / meanData[:, 2] > 0.8) & (meanData[:, 0] > 500)

flag = np.zeros_like(segments)
for i in range(num):
    flag = np.where(segments == i, pred[i], flag)
plt.imshow(flag)
writeImage(flag, './test/slic_cloud_1.tiff')
# %%
img_GaussianBlur = cv2.GaussianBlur(img01, (7, 7), 0)

img_bilateralFilter=cv2.bilateralFilter(img01, 9, 75, 75)



# %%
