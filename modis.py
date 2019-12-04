# %%
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
from glob import glob


# %%
# hdfPath = 'F:/mod09a1/h27v05_2009_2015/2009/'
# hdfPath = 'F:/mod09a1/h11v04/2008/'
hdfPath = 'F:/mod09a1/h27v06_2005_2010/2008/'
# hdfList = os.listdir(hdfPath)
# hdfList = glob(os.path.join(hdfPath, '**/*.hdf'), recursive=True)
hdfList = glob(os.path.join(hdfPath, '*.hdf'))
sd = SD(hdfList[0])
datasets_dic = sd.datasets()
for idx, sds in enumerate(datasets_dic.keys()):
    print(idx, sds)

l = len(hdfList)
w, h = sd.select(0).get().shape
print("len:{0}, w:{1}, h:{2}".format(l, w, h))

# %%
# x, y= 1700, 1500
# x, y = 1900, 1000
x, y = 700, 500
flow = np.zeros((l, 7 + 1), dtype=np.int16)
# savePath = 'F:/mod09a1/h11v04/2008/'
for i, hdf in enumerate(hdfList):
    print(i)
    sd = SD(hdf)
    for j in range(7):
        band = sd.select(j).get()[x, y]
        flow[i, j] = band
    state = sd.select(11).get()[x, y]
    flow[i, 7] = (state & 3 == 2) * 2 + (state & 3 == 1) * 1 # 1是纯云，2是混合

states = flow[:, 7]
flow = flow[:, :7].T * 1e-4

#%%
plt.plot(flow[0], marker='o')
a, b = flow[:, :-1], flow[:, 1:]
rho = np.sum(a * b, 0)/ np.sqrt(np.sum(a * a, 0) * np.sum(b * b, 0))

# %%
plt.figure()
for i, band in enumerate(flow):
    plt.plot(np.where(band > 0, band, 0), label='b'+str(i))
plt.plot(np.where(rho > 0, rho, 0), label='rho') 
plt.legend(loc=0)
plt.xlabel('date')
plt.ylabel('reflect')
my_y_ticks = np.arange(0, 1.1, 0.2)
my_x_ticks = np.arange(0, 50, 10)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show()

# %%
# states = np.zeros((l, 1), dtype=np.int16)
# for i, hdf in enumerate(hdfList):
#     print(i)
#     sd = SD(hdf)
#     state = sd.select(11).get()[x, y] # 11 波段是状态波段
#     states[i] = (state & 3 == 2) * 1 + (state & 3 == 1) * 2 # 1是纯云，2是混合

# %%

idx1 = np.where(states != 0)
idx2 = np.where((states == 0) & (flow[0] > 0.1))
idx3 = np.where(states == 0 & (flow[0] < 0.1))
# idx1 = np.where(flow[0] > 0.17)
# idx2 = np.where((flow[0] < 0.17) & (flow[0] > 0.08))
# idx3 = np.where(flow[0] < 0.08)

for i in range(1):
    plt.figure()
    plt.scatter(idx3[0]*8, flow[i, idx3[0]], label='clear')
    plt.scatter(idx1[0]*8, flow[i, idx1[0]], c='r', label='thick')
    plt.scatter(idx2[0]*8, flow[i, idx2[0]], marker='*', s=100, c='r', label='thin')
    plt.plot(np.arange(len(flow[i]))*8, flow[i])
    plt.legend(loc=0)
    plt.xlabel('date')
    plt.ylabel('reflect')
    plt.show()

# %%

idx1 = np.where(states != 0)
idx2 = np.where(states == 0)
# idx1 = np.where(flow[0] > 0.17)
# idx2 = np.where((flow[0] < 0.17) & (flow[0] > 0.03))
# idx3 = np.where(flow[0] < 0.03)

for i in range(7):
    plt.figure()
    plt.scatter(idx2[0]*8, flow[i, idx2[0]], label='clear')
    plt.scatter(idx1[0]*8, flow[i, idx1[0]], c='r', label='cloud')
    plt.plot(np.arange(len(flow[i]))*8, flow[i])
    plt.legend(loc=0)
    plt.xlabel('date')
    plt.ylabel('reflect')
    plt.show()
# %%
plt.plot(flow[6], marker='o')
plt.xlabel('date')
plt.ylabel('reflect')
# plt.scatter(len(range(states)), flow[6])

# %%
def myfunc(idx, arr, l):
    # min{Ax-b}
    # x = (A.TA)^{-1}A.Tb
    b = arr
    if(len(b) < 3):
        return np.zeros((3)), np.zeros((l))
    A = np.array([np.ones_like((idx)), np.sin(idx*2*np.pi/l), np.cos(idx*2*np.pi/l)]).T
    
    # 加了 L2 正则化的x
    lambd = 1
    x = np.dot(np.linalg.inv(np.dot(A.T, A) + np.diag([0, lambd, lambd])), np.dot(A.T, b))
    
    idx = np.arange(l)
    A2 = np.array([np.ones_like((idx)), np.sin(idx*2*np.pi/l), np.cos(idx*2*np.pi/l)]).T
    return x, A2 @ x

# %%
i = 1
idx1 = np.where(flow[i] > 0.4)
idx2 = np.where(flow[i] < 0.4)
plt.figure()
plt.scatter(idx1[0], flow[i, idx1[0]], c='r')
paras, values = myfunc(idx2[0], flow[i, idx2[0]], len(flow[i]))
plt.plot(values)
idx3 = np.where((flow[i] < 0.4) & (np.abs(values - flow[i]) > 0.07))
idx2 = np.where((flow[i] < 0.4) & (np.abs(values - flow[i]) < 0.07))
plt.scatter(idx3[0], flow[i, idx3[0]], c='r', marker='*', label='step3')
plt.scatter(idx2[0], flow[i, idx2[0]])
plt.plot(flow[i])
plt.legend()
plt.xlabel('date')
plt.ylabel('reflect')
plt.show()

# %%
def myfunc(idx, arr, l):
    # min{Ax-b}
    # x = (A.TA)^{-1}A.Tb
    b = arr
    if(len(b) < 3):
        return np.zeros((3))
    A = np.array([np.ones_like((idx)), np.sin(idx*2*np.pi/l), np.cos(idx*2*np.pi/l)]).T
    
    # 加了 L2 正则化的x
    lambd = 1
    x = np.dot(np.linalg.inv(np.dot(A.T, A) + np.diag([0, lambd, lambd])), np.dot(A.T, b))
    return x

# %%
