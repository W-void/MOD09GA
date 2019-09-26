#%%
import gdal
from gdalconst import *
from pyhdf.SD import SD, SDC
import numpy as np
import os


#%%
def readHDF(img_path):
    ds = gdal.Open(img_path)
    sds = ds.GetSubDatasets()
    # print(len(sds))
    # for sd in sds:
        # print('Name:{0}\nDescription:{1}'.format(*sd))

    proj = ds.GetProjection() 
    geoTrans = ds.GetGeoTransform()

    bands = []
    validBands = [13, 14, 11, 12, 15, 16, 17]
    for v in validBands:
        band_i = gdal.Open(sds[v][0]).ReadAsArray()
        bands.append(band_i)
    return bands, proj, geoTrans


def readTIFF(img_path):
    ds = gdal.Open(img_path)
    # print(len(sds))
    # for sd in sds:
        # print('Name:{0}\nDescription:{1}'.format(*sd))

    bands = []
    bands_num = ds.RasterCount
    for i in range(bands_num):
        band_i = ds.GetRasterBand(i + 1)
        band_data = band_i.ReadAsArray(0, 0, band_i.XSize, band_i.YSize)
        bands.append(band_data)

    return bands


def writeImage(bands, path, geotrans=None, proj=None):
    # 认为各波段大小相等，所以以第一波段信息作为保存
    if bands.ndim == 2:
        bands = bands[:, :, None]
    # 设置影像保存大小、波段数
    band1 = bands[:, :, 0]
    img_width = band1.shape[1]
    img_height = band1.shape[0]
    num_bands = bands.shape[2]
    datatype = gdal.GDT_UInt16

    # 创建文件
    # 先创建驱动，再创建相应的栅格数据集
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
    if geotrans is not None:
        dataset.SetGeoTransform(geotrans)  # 写入仿射变换参数
    if proj is not None:
        dataset.SetProjection(proj)  # 写入投影
    for i in range(num_bands):
        dataset.GetRasterBand(i + 1).WriteArray(bands[:, :, i])
    print("save image success.")
