#%%
import numpy as np
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import os
import cv2
from libtiff import TIFF
from gdalconst import *
import pandas as pd
import gdal


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
#%%
path = 'D:/Data/landsat/tmp1-7.tif'
# qa = 'D:/Data/landsat/tmp_qa.tif'
# ds = gdal.Open(qa)
# QA = ds.GetRasterBand(1).ReadAsArray()
# w, h = QA.shape
landsat = np.zeros((7901, 7771, 7))
ds = gdal.Open(path)
for i in range(7):
    bi = ds.GetRasterBand(i+1).ReadAsArray()
    print(i, bi.shape)
    landsat[:, :, i] = bi

#%%
def fmask(bands):
    bs = bands * 1e-4
    r = bs[:, :, 3]
    nir = bs[:, :, 4]
    nir = np.where(nir < -100, 10000, nir)
    g = bs[:, :, 2]
    swir = bs[:, :, 5]
    ndvi = (nir - r) / (nir + r)
    ndsi = (g - swir) / (g + swir)
    cloud1 = (swir > 0.03) & (ndsi < 0.8) & (ndvi < 0.8)

    meanVis = np.mean(bs[:, :, 1:4], axis=-1)
    white = np.abs(np.sum(bs[:, :, 1:4] - meanVis[:, :, None])) / meanVis < 0.7
    hot = bs[:, :, 1] - 0.5*r - 0.08 > 0
    ratio = nir / swir > 0.75
    water = (ndvi<0.01) & (nir < 0.11) | (ndvi < 0.1) & (nir < 0.05)

    pcp = cloud1 & white & hot & ratio

    # clearWater = water & (band[:, :, 6] < 0.03)
    # T_water = 0.825 * np.sum(bt * clearWater) / np.sum(clearWater)
    # waterTP =  ()

    # clearLandFlag = (~pcp) & (~water)
    # clearLand = nir[clearLandFlag] 
    # nirSort = np.sort(clearLand)
    # lowLevel = nirSort[int(0.175 * len(nirSort)]
    # seedPoint = clearLandFlag < lowLevel

    # mask = np.zeros([h+2, w+2], np.uint8)
    # cv.floodFill(copyImage, mask, (0, 80), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    return pcp

#%%
cloud = fmask(landsat)
plt.imshow(cloud)
writeImage(cloud, './test/landsat.tiff')

# %%
cloud = bi & 32 == 32
plt.imshow(bi)

# %%
