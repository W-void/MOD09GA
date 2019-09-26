#%%
import numpy as np 
import os
import gdal
from gdalconst import *


# #%%
# hdfPath = 'F:/MOD09GA/'
# tiffPath = 'D:/Data/MOD09GA/'

# hdfList = os.listdir(hdfPath)
# for hdf in hdfList[len(os.listdir(tiffPath)):]:
#     ds = gdal.Open(hdfPath + hdf)
#     sds = ds.GetSubDatasets()
#     # print(len(sds))
#     # for sd in sds:
#         # print('Name:{0}\nDescription:{1}'.format(*sd))

#     proj = ds.GetProjection() 
#     geoTrans = ds.GetGeoTransform()

#     bands = []
#     validBands = [13, 14, 11, 12, 15, 16, 17]
#     for v in validBands:
#         band_i = gdal.Open(sds[v][0]).ReadAsArray()
#         bands.append(band_i)

#     imgWidth, imgHeight = bands[0].shape
#     driver = gdal.GetDriverByName("GTiff")
#     fileName = tiffPath+hdf[:-3]+'tiff'
#     dataset = driver.Create(fileName, imgWidth, imgHeight, len(bands), gdal.GDT_UInt16)
#     dataset.SetGeoTransform(geoTrans)
#     dataset.SetProjection(proj)
#     for i in range(len(bands)):
#         dataset.GetRasterBand(i + 1).WriteArray(bands[i])


#%%
hdfPath = 'F:/MOD09GA/'
tiffPath = 'D:/Data/MOD09GA_ST/'

hdfList = os.listdir(hdfPath)
for hdf in hdfList[len(os.listdir(tiffPath)):]:
    ds = gdal.Open(hdfPath + hdf)
    sds = ds.GetSubDatasets()
    # print(len(sds))
    # for sd in sds:
        # print('Name:{0}\nDescription:{1}'.format(*sd))

    proj = ds.GetProjection() 
    geoTrans = ds.GetGeoTransform()

    bands = []
    validBands = [1, 5]
    for v in validBands:
        band_i = gdal.Open(sds[v][0]).ReadAsArray()
        bands.append(band_i)

    imgWidth, imgHeight = bands[0].shape
    driver = gdal.GetDriverByName("GTiff")
    fileName = tiffPath+hdf[:-3]+'tiff'
    dataset = driver.Create(fileName, imgWidth, imgHeight, len(bands), gdal.GDT_UInt16)
    dataset.SetGeoTransform(geoTrans)
    dataset.SetProjection(proj)
    for i in range(len(bands)):
        dataset.GetRasterBand(i + 1).WriteArray(bands[i])