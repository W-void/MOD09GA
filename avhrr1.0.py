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



#%%
class myTimer(object):
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *unused):
        self.end = time.time()
        self.secs = self.end - self.start
        print("elapsed time: %f s" %self.secs)


def detectCloud(R, NIR, T):
    flag = (T > 2900) & (NIR/R > 1.3) | (T/np.where(R<NIR, R, NIR) > 2)
    return flag


def detectCloudFromQA(QA):
    flag = (QA & 2 == 2)
    return ~flag


def writeImage(bands, path, geotrans=None, proj=None):
    img_width = bands.shape[1]
    img_height = bands.shape[0]

    # 设置保存影像的数据类型
    datatype = gdal.GDT_Int16

    # 创建文件
    # 先创建驱动，再创建相应的栅格数据集
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, img_width, img_height, 2, datatype)
    for i in range(2):
        dataset.GetRasterBand(i+1).WriteArray(bands[:, :, i])
    print("save image success.")


#%%
avhPath = 'F:/avh_new/1999/'
hdfList = os.listdir(avhPath)
sd = SD(avhPath+hdfList[0])
datasets_dic = sd.datasets()
for idx, sds in enumerate(datasets_dic.keys()):
    print(idx, sds)

# state = sd.select(9).get()
# cloudFlag = (state & 2) == 2
# tif = TIFF.open('F:/avh_new/output/flag.tiff', mode='w')
# tif.write_image(cloudFlag, compression=None)
# tif.close()

#%%
x, y = range(1100, 1600), range(4900, 5400)
yy, xx = np.meshgrid(y, x)
imgFlow = np.zeros((500, 500, 364), dtype=np.int16)
savePath = 'F:/avh_new/demo/'
for i, hdf in enumerate(hdfList[:364]):
    print(i)
    sd = SD(avhPath+hdf)
    R = sd.select(0).get()[xx, yy]
    # NIR = sd.select(1).get()[xx, yy]
    # writeImage(np.stack((R, NIR), -1), savePath+'orig_'+str(i)+'.tiff')
    # T = sd.select(4).get()[xx, yy]
    # landFlag = detectCloud(R, NIR, T)
    QA = sd.select(9).get()[xx, yy]
    landFlag = detectCloudFromQA(QA)
    imgFlow[:, :, i] = np.clip(np.where(landFlag, R, -200), a_min=-200, a_max=4000)

  
#%%
def myfunc(arr):
    # min{Ax-b}
    # x = (A.TA)^{-1}A.Tb
    idx = np.where(arr != -200)[0]
    b = arr[idx]
    if(len(b) < 3):
        return np.zeros((3))
    A = np.array([np.ones_like((idx)), np.sin(idx*2*np.pi/365), np.cos(idx*2*np.pi/365)]).T
    
    # 没有正则化的x
    # x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    
    # 加了 L2 正则化的x
    lambd = 1
    x = np.dot(np.linalg.inv(np.dot(A.T, A) + np.diag([0, lambd, lambd])), np.dot(A.T, b))
    # x = np.array([np.mean(arr), np.std(arr), 0])
    # lambd = 0.1
    # lr = 0.01
    # for i in range(100):
    #     delta = np.dot(A.T, np.dot(A, x)-b) + lambd * x
    #     if np.linalg.norm(delta, ord=1) < 1:
    #         break
    #     x -=  lr * delta
    return x


with myTimer():
    popt = np.apply_along_axis(myfunc, -1, imgFlow)

# km = KMeans(n_clusters=10, random_state = 8)
# y_pre = km.fit_predict(popt.reshape((-1, 3)))
# plt.imshow(y_pre.reshape((400, 400)))


#%%
# def func(x, a, b, c):
#     return a + b * np.sin(2*np.pi*x/365) + c * np.cos(2*np.pi*x/365)


# def curveFit(arr):
#     xdata = np.where(arr != -200)
#     ydata = arr[xdata]
#     if(len(ydata) < 3):
#         return np.zeros((3))
#     popt, pcov = curve_fit(func, xdata[0], ydata)
#     return popt


# with myTimer():
#     popt = np.apply_along_axis(curveFit, -1, imgFlow)

# km = KMeans(n_clusters=10, random_state = 8)
# y_pre = km.fit_predict(popt.reshape((-1, 3)))
# plt.imshow(y_pre.reshape((400, 400)))


#%%
# y = imgFlow[0, 0]
# para = popt[0, 0]
# x = np.arange(len(y))
# plt.scatter(x, y, s=5)
# plt.plot(x, np.dot(para, np.array([np.ones_like((idx)), np.sin(idx*2*np.pi/365), np.cos(idx*2*np.pi/365)])))

#%%
def writeImage(bands, path, geotrans=None, proj=None):
    img_width = bands.shape[1]
    img_height = bands.shape[0]

    # 设置保存影像的数据类型
    datatype = gdal.GDT_Int16

    # 创建文件
    # 先创建驱动，再创建相应的栅格数据集
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, img_width, img_height, 1, datatype)
    dataset.GetRasterBand(1).WriteArray(bands)
    print("save image success.")


savePath = 'F:/avh_new/R/'
for i in range(364):
    if i % 10 == 0:
        print(i)
    basis = np.array([1, np.sin(2*np.pi*i/365), np.cos(2*np.pi*i/365)])
    img = np.dot(popt, basis)
    # writeImage(img, savePath+str(i)+'.tiff')
    tif = TIFF.open(savePath+str(i)+'.tiff', mode = 'w')
    tif.write_image(img, compression=None)
    tif.close()

#%%
def readImage(img_path):
    dataset = gdal.Open(img_path, GA_ReadOnly)
    band = dataset.GetRasterBand(1).ReadAsArray()
    return band
        
RPath = 'F:/avh_new/R/'
demoPath = 'F:/avh_new/demo/'
outPath = 'F:/avh_new/cloud/'
demoList = os.listdir(demoPath)
RList = os.listdir(RPath)
for i in range(364):
    if i % 10 == 0:
        print(i)
    tif = TIFF.open(RPath+RList[i], mode='r')
    R = tif.read_image()
    tif.close()

    demoR = readImage(demoPath+demoList[i])
    cloudFlag = demoR - R > 1000

    tif = TIFF.open(outPath+'cloud_'+RList[i], mode = 'w')
    tif.write_image(cloudFlag, compression=None)
    tif.close()

#%%
