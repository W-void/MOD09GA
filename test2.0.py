#%%
import numpy as np 
import matplotlib.pyplot as plt 
import os
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import time
from sklearn.decomposition import NMF
import pywt 
from sklearn.cluster import DBSCAN
from scipy import interpolate 


#%%
imgFlow = np.load('./test/test_img.npy')
x, y = 150, 150
flow = imgFlow[x, y] * 1e-4
plt.plot(flow[0])
a, b = flow[:, :-1], flow[:, 1:]
rho = np.sum(a * b, 0)/ np.sqrt(np.sum(a * a, 0) * np.sum(b * b, 0))

#%%
plt.figure()
for i, band in enumerate(flow):
    plt.plot(np.where(band[:100] > 0, band[:100], 0), label='b'+str(i))
plt.plot(np.where(rho[:100] > 0, rho[:100], 0), label='rho') 
plt.legend(loc=1)
plt.xlabel('date')
plt.ylabel('reflect')
my_y_ticks = np.arange(0, 1.1, 0.2)
my_x_ticks = np.arange(0, 100, 10)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show()


# %%
plt.rcParams['axes.facecolor']='black'
plt.figure()
plt.plot(flow[0], c='green')
idx = np.where(rho > 0.992)
plt.plot(idx[0], flow[0][idx[0]], c='r')

# %%
db = DBSCAN(eps=0.1, min_samples=10).fit(flow[0].reshape(-1, 1))
plt.plot(db.labels_, c='red')  
plt.plot(flow[0], c='black')

# %%
clf = KMeans(n_clusters=3)
clf.fit(flow[0].reshape(-1, 1))
clf.labels_
plt.plot(clf.labels_, c='red')  
plt.plot(flow[0], c='green')

# %%
idx = np.where(clf.labels_== 0)[0]
mea = np.mean(flow[:, idx], 1)
# %%
rho = np.sum(mea[:, None] * flow, 0) / np.sqrt(np.sum(mea * mea) * np.sum(flow * flow, 0))

# %%
def myfunc(idx, arr):
    # min{Ax-b}
    # x = (A.TA)^{-1}A.Tb
    # idx = np.where(arr != -200)[0]
    b = arr[idx]
    if(len(b) < 3):
        return np.zeros((3))
    A = np.array([np.ones_like((idx)), np.sin(idx*2*np.pi/365), np.cos(idx*2*np.pi/365)]).T
    
    # 没有正则化的x
    # x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    
    # 加了 L2 正则化的x
    lambd = 0.1
    x = np.dot(np.linalg.inv(np.dot(A.T, A) + np.diag([0, lambd, lambd])), np.dot(A.T, b))

    return x

# idx = np.where((flow[0] < 0.3) & (flow[0] > 0))[0]
weights = myfunc(idx, flow[0])
print(weights)
plt.plot(flow[0])
x = np.arange(364)
plt.plot(x, weights[0] + weights[1]*np.sin(x*2*np.pi/365) + weights[2]*np.cos(x*2*np.pi/365))
my_y_ticks = np.arange(-1, 1.1, 0.2)
my_x_ticks = np.arange(0, 365, 60)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.ylim(0, 1)
plt.show()

# %%
def func(x, a, b, c):
    return a + b * np.sin(2*np.pi*x/365 + c) 


def curveFit(idx, arr):
    ydata = arr[idx]
    if(len(ydata) < 3):
        return np.zeros((3))
    popt, pcov = curve_fit(func, idx, ydata)
    return popt

weights = curveFit(idx, flow[0])
print(weights)
# plt.plot(flow[0])
x = np.arange(364)
plt.plot(x, weights[0] + weights[1] * np.sin(x*2*np.pi/365 + weights[2]))
my_y_ticks = np.arange(-1, 1.1, 0.2)
my_x_ticks = np.arange(0, 365, 60)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.ylim(0, 1)
plt.show()


# %%
pred_y = weights[0] + weights[1] * np.sin(x*2*np.pi/365 + weights[2])
y = np.where(clf.labels_== 0, flow[0], pred_y)

cA,cD = pywt.dwt(y, 'db2')
cD = np.zeros(len(cD))
new_data = pywt.idwt(cA, cD, 'db2')
plt.plot(flow[0])
plt.plot(new_data)
plt.ylim(0, 1)
# %%
