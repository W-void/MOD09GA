#%%
import numpy as np 
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt 
import os
import cv2
from libtiff import TIFF
import gdal
from gdalconst import *
import pandas as pd
from sklearn.externals import joblib


#%%
hdfPath = 'F:/MOD09GA/'
tiffPath = './MOD09GA_ST/'

hdfList = os.listdir(hdfPath)
for hdf in hdfList[len(os.listdir(tiffPath)):]:
    ds = gdal.Open(hdfPath + hdf)
    sds = ds.GetSubDatasets()

    bands = []
    validBands = [1, 5, 17]  #[state_1Km, zenith, QC_500m]
    for v in validBands:
        band_i = gdal.Open(sds[v][0]).ReadAsArray()
        if v != 17:
            band_i = cv2.resize(band_i, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        bands.append(band_i)

    imgWidth, imgHeight = bands[0].shape
    driver = gdal.GetDriverByName("GTiff")
    fileName = tiffPath+hdf[:-3]+'tiff'
    dataset = driver.Create(fileName, imgWidth, imgHeight, len(bands), gdal.GDT_UInt16)
    for i in range(len(bands)):
        dataset.GetRasterBand(i + 1).WriteArray(bands[i])