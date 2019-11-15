# %%
import numpy as np
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import os
import cv2
import imageio


# %%
avhPath = 'F:/avh_new/1999/'
hdfList = os.listdir(avhPath)
sd = SD(avhPath+hdfList[0])
datasets_dic = sd.datasets()
for idx, sds in enumerate(datasets_dic.keys()):
    print(idx, sds)


# %%
for i, hdf in enumerate(hdfList[:60]):
    print(i)
    sd = SD(avhPath+hdf)
    # temp = sd.select(4).get()
    # R = sd.select(0).get()
    # NIR = sd.select(1).get()
    # NDVI = (NIR - R) / (NIR + R)
    # cloudFlag = (NDVI < 0.8)& (temp < 2800)
    state = sd.select(9).get()
    cloudFlag = (state & 2) == 2
    cv2.imwrite('./test/avhrr/'+str('%03d'%i)+'.png', np.uint8(cloudFlag*200))

# %%
images = []
filenames = sorted((fn for fn in os.listdir('./test/avhrr/') if fn.endswith('.png')))
for filename in filenames[:60]:
    print(filename)
    images.append(imageio.imread('./test/avhrr/'+filename))
imageio.mimsave('./test/avhrr/cloud.gif', images, duration=1)


# %%
