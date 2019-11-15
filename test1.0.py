# %%
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
from sklearn.mixture import GMM, GaussianMixture
from scipy.optimize import curve_fit


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
def detectCloud(b):
    ###
    cloud = (b[:, :, 3] > 2000) & (b[:, :, 0] > 1500)
    noCloud = (~cloud) & (((b[:, :, 0] < 400) & (
        b[:, :, 1] < 800)) | (b[:, :, 5] < 1000))
    masker = cloud * 1 + noCloud * 2
    img = np.clip(b[:, :, :3]*1e-4, a_min=0, a_max=1).astype(np.float32)
    mark4 = cv2.watershed(np.uint8(img*255), masker)    # 返回-1是边界，0是不确定，剩下的就是目标
    flag = mark4 ==1
    ###
    # flag = (b[:, :, 3] > 1200) & (b[:, :, 0] > 1400)
    # flag = (b & 3 == 1) | (b & 3 == 2)
    # flag = gmm(b)
    return cloud, flag


#%%
def mod_fmask(bands):
    bs = bands * 1e-4
    r = bs[:, :, 2]
    nir = bs[:, :, 3]
    nir = np.where(nir < -100, 10000, nir)
    g = bs[:, :, 1]
    swir = bs[:, :, 6]
    ndvi = (nir - r) / (nir + r)
    ndsi = (g - swir) / (g + swir)
    cloud1 = (swir > 0.03) & (ndsi < 0.8) & (ndvi < 0.8)

    meanVis = np.mean(bs[:, :, :3], axis=-1)
    white = np.abs(np.sum(bs[:, :, :3] - meanVis[:, :, None])) / meanVis < 0.7
    hot = bs[:, :, 0] - 0.5*r - 0.08 > 0
    ratio = nir / swir > 0.75
    water = (ndvi<0.01) & (nir < 0.11) | (ndvi < 0.1) & (nir < 0.05)

    pcp = cloud1 & white & hot & ratio
    return pcp


#%%
def detect_1(bands):
    b = bands[:, :, 0]
    r = bands[:, :, 2]
    ratio = np.clip(b/r, -1, 10)
    plt.hist(ratio.flatten(), 100)
    clf = GaussianMixture(3)
    x = ratio.reshape(-1, 1)
    idx = np.random.choice(len(x), size=10000, replace=False)
    clf.fit(x[idx])
    

#%%
def detect_2(bands):
    b = bands[:, :, 0]
    r = bands[:, :, 2]
    g = bands[:, :, 1]
    ratio = np.clip(b/g, -1, 10)
    # plt.hist(ratio.flatten(), 100, range=[-0.5, 3])
    x = ratio.reshape(-1, 1)
    x = x[(x<3)&(x>-0.5)]
    x = x[:,None]
    idx = np.random.choice(len(x), size=10000, replace=False)
    clf = GaussianMixture(3, max_iter=500)
    clf.fit(x[idx])

    gmm_x = np.linspace([-0.5], [3], 100, axis=0)
    gmm_y = np.exp(clf.score_samples(gmm_x))
    plt.figure()
    plt.hist(x, 100, normed=True, range=[-0.5, 3])
    plt.plot(gmm_x, gmm_y, color='r', lw=2)

    weights, means, covs = clf.weights_, clf.means_, clf.covariances_
    
    noCloud = (ratio < np.sort(means.flatten())[0] + 0.01) | (b < 600)
    cloud = ratio > np.sort(means.flatten() - 0.01)[1]
    # cloud = (b[:, :, 3] > 0.2) & (b[:, :, 0] > 0.15)
    # noCloud = (~cloud) & (((b[:, :, 0] < 0.04) & (
    #     b[:, :, 1] < 0.08)) | (b[:, :, 5] < 0.1))
    masker = cloud * 1 + noCloud * 2
    img = np.clip(ratio, 0, 2) / 2
    img = np.stack((img,img,img), -1)
    mark4 = cv2.watershed(np.uint8(img*255), masker)    # 返回-1是边界，0是不确定，剩下的就是目标
    flag = (mark4 == 1) | (mark4 == -1)

    return cloud, flag

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


# %%
cloud, flag = detectCloud(data)
plt.imshow(cloud)
plt.figure()
plt.imshow(flag)
writeImage(cloud, './test/mod_cloud_2.tiff')
writeImage(flag, './test/mod_test_2.tiff')
# %%
plt.imshow(cloud)

# %%
writeImage(flag, './test/mod_test.tiff')
                           

# %%
for i in range(7):
    plt.figure()
    plt.hist(np.log(101+data[:, :, i].flatten()), bins=100)


# %%
# 拟合直方图
def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1)+gauss(x, mu2, sigma2, A2)


hist, bin = np.histogram(data[:, :, 0], bins=100)
x = (bin[1:]+bin[:-1])/2
y = hist / 100000
expected = (6, 500, 25, 1, 6000, 125)
params, cov = curve_fit(bimodal, x, y, expected)
sigma = sqrt(diag(cov))
plot(x, bimodal(x, *params), color='red', lw=3, label='model')
legend()
print(params, '\n', sigma)


# %%
x = data[:, :, 0].reshape(-1, 1)
plt.hist(x, bins=100)
clf = GaussianMixture(2)
idx = np.random.choice(len(x), size=10000, replace=False)
clf.fit(x[idx])

print(clf.predict(1000), clf.predict(500), clf.predict(100))

# %%
clas = clf.predict(x).reshape(2400, 2400)
plt.imshow(clas)
writeImage(clas, './test/gmm_b.tiff')

# %%
gmm_x = np.linspace([0], [10000], 100, axis=0)
gmm_y = np.exp(clf.score_samples(gmm_x))
plt.hist(x, 100, normed=True)
plt.plot(gmm_x, gmm_y, color='r', lw=2)

# %%
prob = clf.predict_proba(gmm_x)
np.argmax(prob, axis=-1)
# %%
plt.hist()