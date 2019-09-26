# %%
import os
from libtiff import TIFF
import numpy as np

# %%
path = 'D:/BaiduNetdiskDownload/'
path_x = 'LC08_L1TP_123033_20130731_20170503_01_T1_sr_band1-7.tif'
path_y = 'LC08_L1TP_123032_20130731_20170503_01_T1_pixel_qa.tif'
tif_x = TIFF.open(path + path_x)
tif_y = TIFF.open(path + path_y)

# %%
# for img in tif.iter_images():
#     print(type(img), img.shape)

# %%
bands = tif_x.read_image()
qa = tif_y.read_image()
print(bands.shape, qa.shape)

# %%
uni = np.unique(qa)

# %%
len(uni)

# %%
for i in uni:
    print(bin(i))

# %%
# 查看填充、云占的比例
print(np.sum(qa & 1) / qa.size)
print(np.sum((qa & 32) > 0) / qa.size)

# %%
# pass one -- get Potential Cloud Pixels (PCPs)
B, G, R, NIR, SWIR_1, BT, SWIR_2 = bands[0]
ndsi = (bands[1] - bands[4]) / (bands[1] + bands[4])
ndvi = (bansd[3] - bnads[2]) / (bands[3] + bands[2])

mask = bands[6]

mask = bands[6] > 0.03 
meanVis = np.mean(bands[0:3], axis=0)

#%%
from libtiff import TIFF
import numpy as np
import pandas as pd
import re
import os 

#%%
# read images
path = 'D:/Data/BC/LC80060102014147LGN00/'
dirlist = os.listdir(path)
valid_ext = ['.tif', '.TIF']
data = []
cols = []
for dir in dirlist:
    if os.path.splitext(dir)[-1]  in valid_ext:
        tif = TIFF.open(path + dir)
        data.append(tif.read_image())
        cols.append(re.split('[_.]', dir)[1])

print(cols)
#%%
# convert tif to array 
valid_band = [0, *range(3, 9), 10, 1, 2, -1]
num_of_bands = len(valid_band)
M, N = data[0].shape
data_np = np.zeros((num_of_bands, M, N))
for i, band in enumerate(valid_band):
    data_np[i] = data[band]

del data
#%%
# crop image through slice array
window_size = 512
x = np.random.randint(M*0.25, M*0.75)
y = np.random.randint(N*0.25, N*0.75)
print("x = {0}, y = {1}, M = {2}, N = {3}".format(x, y, M, N))
img = data_np[:, x:x+window_size, y:y+window_size]

#%%
# get reflectance image
sample_image = np.uint8(img[1:4].T / 100000 * 2 * 255)
# broadcast mask to 3-dim from 2-dim
mask = img[-1]
mask = mask[:, :, None] * np.ones([1, 1, 3])
# remove fill piexl
sample_image = np.where(mask==1, 0, sample_image)


#%%
from matplotlib import pyplot as plt

mean_img = np.mean(sample_image)
plt.hist(hisEqulColor(sample_image)[2].ravel(),256,[0,256])
plt.show()


#%%
import cv2

def  hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    print(len(channels))
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

winname = 'example'
cv2.namedWindow(winname, 0)
cv2.imshow(winname, sample_image)
cv2.waitKey(0)
cv2.destroyWindow(winname)

#%%
for i, item in enumerate(data):
    print(i, item.shape)

y = data[-1].flatten()
mask = np.where(y > 0)
total_pixels = len(mask[0])
print(total_pixels)

#%%
num_of_bands = len(data)
height, weight = data[0].shape
data_array = np.zeros((num_of_bands, total_pixels))
for i in range(num_of_bands):
    if i == 9:
        continue
    data_array[i] = data[i].flatten()[mask]

del(data)

#%%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


X, Y = data_array[:-1].T, data_array[-1].T
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.3)
idx = np.random.permutation(total_pixels)
test_size = 10000
idx_train = idx[:test_size]
X_train, Y_train = X[idx_train], Y[idx_train]

GBDT = GradientBoostingClassifier(n_estimators=10)
GBDT.fit(X_train, Y_train)

# del data_array, X, Y

#%%
OHE = OneHotEncoder()
OHE.fit(GBDT.apply(X_train).reshape((X_train.shape[0], -1)))
LR = LogisticRegression()
LR.fit(OHE.transform(GBDT.apply(X_train).reshape((X_train.shape[0], -1))), Y_train)

idx_test = idx[test_size:2*test_size]
X_test, Y_test = X[idx_test], Y[idx_test]
Y_pred = LR.predict_proba(OHE.transform(GBDT.apply(X_test).reshape((X_test.shape[0], -1))))

#%%
Y_pred_ = np.argmax(Y_pred, axis=1)
Y_test_ = np.argmax(OHE.fit_transform(Y_test[:, None]), axis=1)
acc = accuracy_score(Y_test_, Y_pred_)
print('GradientBoosting + LogisticRegression: ', acc)

#%%
from sklearn.metrics import confusion_matrix

C = confusion_matrix(Y_test, Y_pred)
print('confusion matrix:\n', C)

#%%
from sklearn.tree import export_graphviz
import graphviz


def plot_tree(clf):
    dot_data = export_graphviz(clf, out_file=None, node_ids=True,
                    filled=True, rounded=True, 
                    special_characters=True)
    graph = graphviz.Source(dot_data)  
    return graph


#%%
# now we can plot the first tree
plot_tree(GBDT.estimators_[0, 1])

#%%
Y_pred = GBDT.predict(X_test)
Y_pred_ = np.argmax(OHE.fit_transform(Y_pred[:, None]), axis=1)
accuracy_score(Y_test, Y_pred_)

#%%
