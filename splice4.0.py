'''
时间序列上的中值滤波 + 数据融合
'''

#%%
import numpy as np 
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt 
import os
import cv2
from libtiff import TIFF
import gdal
from gdalconst import *


#%%
def detectCloud(R, NIR):
    ratio = NIR / R
    flag = (R < -100) | (NIR < -100) | (R+NIR>4800) | (R+NIR>4000) & (ratio<2) | (R>2000) & (ratio<1.2) & (ratio>0.8)
    return flag


def readZenith(path):
    sd = SD(path, SDC.READ)
    zenith = sd.select(2).get()
    zenith = np.where(zenith > 0, zenith, zenith*-1)
    return cv2.resize(zenith, (4800, 4800))


def findBetter(GQ, GA):
    clearNIR = np.zeros((4800, 4800), np.int16)
    clearR = np.zeros((4800, 4800), np.int16)

    zenith = []
    flag = []
    NIRList = []
    RList = []
    for i in range(2):
        zenith.append(readZenith(GA[i]))

        sdFile = SD(GQ[i], SDC.READ)
        RList.append(sdFile.select(1).get())
        NIRList.append(sdFile.select(2).get())
        flag.append(detectCloud(RList[i], NIRList[i]))

    # flag : 0-not cloud, 1-cloud
    switchFlag = (flag[0] == 0) & ((flag[1] == 1) | (zenith[0] < zenith[1]))
    flag = flag[0] & flag[1]
    clearR = np.where(switchFlag, RList[0], RList[1])
    clearR = np.where(flag, -200, clearR)
    clearNIR = np.where(switchFlag, NIRList[0], NIRList[1])
    clearNIR = np.where(flag, -200, clearNIR)

    return clearNIR, clearR, ~flag * (~switchFlag + 1)


def medianBlur(src):
    blur = cv2.medianBlur(src, 3)
    flag = (src - blur) / blur > 1.5
    src = np.where(flag, blur, src)


def readImage(img_path):
    data = []
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    if dataset is None:
        print("Unable to open image file.")
        return data
    else:
        print("Open image file success.")
        geoTransform = dataset.GetGeoTransform()
        print('geoTransform:\n', geoTransform)
        im_proj = dataset.GetProjection()  # 获取投影信息
        print('im_proj:\n', im_proj)
        bands_num = dataset.RasterCount
        print("Image height:" + dataset.RasterYSize.__str__() + " Image width:" + dataset.RasterXSize.__str__())
        print(bands_num.__str__() + " bands in total.")
        # for i in range(bands_num):
        #     # 获取影像的第i+1个波段
        #     band_i = dataset.GetRasterBand(i + 1)
        #     # 读取第i+1个波段数据
        #     band_data = band_i.ReadAsArray(0, 0, band_i.XSize, band_i.YSize)
        #     data.append(band_data)
        #     print("band " + (i + 1).__str__() + " read success.")
        
        return geoTransform, im_proj


def writeImage(bands, path, geotrans=None, proj=None):
    projection = [
        # WGS84坐标系(EPSG:4326)
        """GEOGCS["WGS 84", DATUM["WGS_1984", SPHEROID["WGS 84", 6378137, 298.257223563, AUTHORITY["EPSG", "7030"]], AUTHORITY["EPSG", "6326"]], PRIMEM["Greenwich", 0, AUTHORITY["EPSG", "8901"]], UNIT["degree", 0.01745329251994328, AUTHORITY["EPSG", "9122"]], AUTHORITY["EPSG", "4326"]]""",
        # Pseudo-Mercator、球形墨卡托或Web墨卡托(EPSG:3857)
        """PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs"],AUTHORITY["EPSG","3857"]]"""
    ]

    if bands is None:
        return
    else:
        # 认为各波段大小相等，所以以第一波段信息作为保存
        band1 = bands[:, :, 0]
        # 设置影像保存大小、波段数
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.shape[2]

        # 设置保存影像的数据类型
        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 创建文件
        # 先创建驱动，再创建相应的栅格数据集
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
        if dataset is not None:
            if geotrans is not None:
                dataset.SetGeoTransform(geotrans)  # 写入仿射变换参数
            if proj is not None:
                if proj is 'WGS84' or proj is 'wgs84' or proj is 'EPSG:4326' or proj is 'EPSG-4326' or proj is '4326':
                    dataset.SetProjection(projection[0])  # 写入投影
                elif proj is 'EPSG:3857' or proj is 'EPSG-3857' or proj is '3857':
                    dataset.SetProjection(projection[1])  # 写入投影
                else:
                    dataset.SetProjection(proj)  # 写入投影
            for i in range(num_bands):
                dataset.GetRasterBand(i + 1).WriteArray(bands[:, :, i])
        print("save image success.")


#%%
filePath = ['D:/Data/MOD09GQ/MOD/', 'D:/Data/MOD09GQ/MYD/']
savePath = 'D:/Data/MOD09GQ/output/'
GAPath = 'D:/Data/MOD09GA/'

# 找到路径下的目标文件
# hdfFileAfter = ['A2019223.h28v06', 'A2019224.h28v06', 'A2019225.h28v06']
# hdfFileBefore = ['221.h28v06', 'A2019220.h28v06', 'A2019219.h28v06', 'A2019218.h28v06', 'A2019217.h28v06']
hdfFileAfter = ['224.h27v05', '225.h27v05', '226.h27v05', '227.h27v05', '228.h27v05', '229.h27v05']
hdfFileBefore = ['222.h27v05', '221.h27v05', '220.h27v05', '217.h27v05', '216.h27v05', '215.h27v05', '214.h27v05', '213.h27v05', '212.h27v05', '207.h27v05']

H = W = 4800

for i in range(0, 2):
    if i == 0:
        date = hdfFileBefore
        name = 'before'
    else:
        date = hdfFileAfter
        name = 'after'
    
    clearImageR = np.ones((H, W, len(date)), np.int16) * -200
    clearImageNIR = np.ones_like(clearImageR) * -200
    landMask = np.zeros_like(clearImageR, np.int16)
    x = y = range(4800)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx, np.int8)
    flag = np.ones((H, W), bool)

    for i, word in enumerate(date):
        hdfList = []
        zenithList = []

        for p in filePath:
            for fileName in os.listdir(p):
                if word in fileName:
                    hdfList.append(p+fileName)

    
        for fileName in os.listdir(GAPath):
            if word in fileName:
                print(fileName)
                zenithList.append(GAPath+fileName)
     
        # find better from MOD and MYD
        NIR, R, tmpLandMask = findBetter(hdfList, zenithList)
        clearImageNIR[:, :, i] = NIR 
        clearImageR[:, :, i] = R 
        zz += (tmpLandMask > 0)
        landMask[:, :, i] = tmpLandMask
        flag = flag & (tmpLandMask == 0)
    print(np.sum(flag))


    medianR = np.zeros_like(clearImageR)
    clearImageR_sort = np.sort(clearImageR, axis=-1) # 从小到大排
    clearImageR_sort = clearImageR_sort[:, :, ::-1] # 从大到小排
    medianR = clearImageR_sort[yy, xx, np.where(zz, zz-1, 0)]

    medianNIR = np.zeros_like(clearImageR)
    clearImageNIR_sort = np.sort(clearImageNIR, axis=-1)
    clearImageNIR_sort = clearImageNIR_sort[:, :, ::-1]
    medianNIR = clearImageNIR_sort[yy, xx, np.where(zz, zz-1, 0)]

    NO = np.zeros((H, W), np.int8)
    flag1 = np.ones((H, W), bool)
    for i in range(len(date)):
        tmpFlag = (clearImageR[:, :, i] == medianR) & (landMask[:, :, i] > 0)
        NO += np.where(tmpFlag & flag1, 2*i+landMask[:, :, i]-1, 0)
        flag1 = flag1 & ~tmpFlag
    print(np.sum(flag1))

    # 利用标号数组NO，把GQ和GA的数据融合
    bandOrder = [13, 14, 11, 12, 15, 16, 17, 1]
    bandsGA = np.zeros((H, W, len(bandOrder)), np.int16)
    bandsTmp = np.zeros_like(bandsGA)
    for i, word in enumerate(date):
        for fileName in os.listdir(GAPath):
            if word in fileName:
                sd = SD(GAPath+fileName)
                for j, idx in enumerate(bandOrder):
                    band = sd.select(idx).get()
                    if idx == 1:
                        band = cv2.resize(band, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                    else:
                        band = cv2.resize(band, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                    band = cv2.medianBlur(band, 5)
                    bandsTmp[:, :, j] = band
                sameDateFlag = (NO == 2*i+ (0 if 'MOD'in fileName else 1))
                # scaling = medianR / bandsTmp[:, :, 2] * sameDateFlag
                bandsGA += np.int16(bandsTmp * sameDateFlag[:, :, None])
    scaling = medianR / bandsGA[:, :, 2]
    bandsGA = np.where(flag[:, :, None], -200, bandsGA)

    geotrans, proj = readImage('D:\\Data\\MOD09GQ\\'+date[0].split('.')[1]+'_AllWDays_percent.tiff')
    geotrans /= np.array([1, 2, 1, 1, 1, 2])
    writeImage(bandsGA, 'D:\\Data\\MOD09GQ\\output1\\'+date[0].split('.')[1]+'_'+name+'.tiff', \
        geotrans=geotrans, proj=proj)

    writeImage(np.stack((medianR, medianNIR), -1), 'D:\\Data\\MOD09GQ\\output1\\'+date[0].split('.')[1]+'_'+name+'_GQ.tiff', \
        geotrans=geotrans, proj=proj)




   
    # tif = TIFF.open('D:/Data/MOD09GQ/output1/'+date[0].split('.')[1]+'_'+name+'_NIR.tiff', mode = 'w')
    # tif.write_image(clearNIR, compression=None)
    # tif.close()
    # tif = TIFF.open('D:/Data/MOD09GQ/output1/h27v05_'+name+'_R.tiff', mode = 'w')
    # tif.write_image(clearR, compression=None)
    # tif.close()

    # tif = TIFF.open('D:/Data/MOD09GQ/output1/h28v06_'+name+'_NIR.tiff', mode = 'w')
    # tif.write_image(clearNIR, compression=None)
    # tif.close()
    # tif = TIFF.open('D:/Data/MOD09GQ/output1/h28v06_'+name+'_R.tiff', mode = 'w')
    # tif.write_image(clearR, compression=None)
    # tif.close()
    

#%%
