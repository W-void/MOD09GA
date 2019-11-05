#%%
import numpy as np 
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt 
import os
import cv2
from libtiff import TIFF, TIFF3D
import gdal
from gdalconst import *
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import time
from sklearn.decomposition import NMF
import pywt 
%matplotlib qt5


#%%
avhPath = 'F:/avh_new/demo/'
hdfList = os.listdir(avhPath)
sd = SD(avhPath+hdfList[0])
datasets_dic = sd.datasets()
for idx, sds in enumerate(datasets_dic.keys()):
    print(idx, sds)

#%%
# 选出一个样本点查看拟合效果
# x, y = 1300, 5200
avhPath = 'F:/avh_new/demo/'
tiffList = os.listdir(avhPath)


flow = np.zeros((500, 500, 3, 364), dtype=np.float)
for i, tiff in enumerate(tiffList[:364]):
    print(i)
    ds = gdal.Open(avhPath+tiff)
    
    R = ds.GetRasterBand(1).ReadAsArray()
    NIR = ds.GetRasterBand(2).ReadAsArray()
    NDVI = (NIR-R)/(NIR+R)
    flow[:, :, :, i] = np.stack((R, NIR, NDVI), axis=-1)

# %%
def plot(d):
    x = np.where(d>=-100)
    y = d[x]
    plt.scatter(x[0], y, s=10, c='r', marker='o')

# %%
x, y = 200, 300
d = flow[x, y]
plot(d[0])
# %%
flow = np.load('./test.npy')

# %%
