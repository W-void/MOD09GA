#%%
import os
import cv2
import numpy as np
from libtiff import TIFF


#%%
def medianBlur(src):
    blur = cv2.medianBlur(src, 5)
    ratio = (src - blur) / blur
    flag = (ratio>1.2) | (ratio<0.8)
    src = np.where(flag, blur, src)


#%%
path = 'D:/Data/MOD09GQ/output/'
outPath = 'D:/Data/MOD09GQ/output1/'
for f in os.listdir(path):
    if f == 'h28v06_after_NIR.tiff':
        tiff = TIFF.open(path+f, mode='r')
        img = tiff.read_image() 
        tiff.close()
        
        medianBlur(img)
        tiff = TIFF.open(outPath+f, mode='w')
        tiff.write_image(img, compression=None)
        tiff.close()

#%%
