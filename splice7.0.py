'''
GA数据的云检测
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
import pandas as pd
from sklearn.externals import joblib


#%%
def detectCloud(b):
    cloud = (b[:, :, 3] > 1600) & (b[:, :, 0] > 2000)
    noCloud = (~cloud) & ((b[:, :, 0] < 700) & (b[:, :, 1] < 1000) | (b[:, :, 5] < 600) & (b[:, :, 6] < 600))
    masker = cloud * 1 + noCloud * 2
    img = np.clip(b[:, :, :3]*1e-4, a_min=0, a_max=1).astype(np.float32)
    mark4 = cv2.watershed(np.uint8(img*255) ,masker)    # 返回-1是边界，0是不确定，剩下的就是目标
    flag = mark4 != 2
    # flag = (b[:, :, 3] > 1200) & (b[:, :, 0] > 1400)
    return flag


def findBetter(GA, tifList):
    blue = np.zeros((2400, 2400), np.float32)
    blueList = []
    zenithList = []
    flag = []
    # dataset = gdal.Open(allW, GA_ReadOnly)
    # band_i = dataset.GetRasterBand(1)
    # water = band_i.ReadAsArray(0, 0, band_i.XSize, band_i.YSize)

    pkl = joblib.load('./pkl/train_model.pkl')
    clf = pkl['DT']
    for i in range(2):
        ds = gdal.Open(GA[i])
        zenith = ds.GetRasterBand(2).ReadAsArray()
        # zenith = np.where(zenith > 0, zenith, zenith*-1)
        # zenith = cv2.resize(zenith.astype(np.int8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST).astype(bool)
        zenithList.append(zenith)
        
        validFlag = np.ones((2400, 2400), np.bool)
        # data = pd.DataFrame()
        # bandOrder = [13, 14, 11, 12, 15, 16, 17]
        data = np.zeros((2400, 2400, 7), np.int16)
        ds = gdal.Open(tifList[i])
        for i in range(7):
            bi = ds.GetRasterBand(i+1).ReadAsArray()
            data[:, :, i] = bi
            # if i >=5:
            #     continue
            # validFlag = validFlag & np.where((bi>=-100)&(bi<10000), True, False)
        qc = ds.GetRasterBand(3).ReadAsArray()
        validFlag = (qc & 3) == 2
        
        NDVI = (data[:, :, 3] - data[:, :, 2]) / (data[:, :, 3] + data[:, :, 2])
        maxVis = np.max(data[:, :, :3], axis=-1)
        SWIR = np.where(data[:, :, 6] > -200, data[:, :, 6], 1]
        NDWI = (maxVis - SWIR) / (maxVis + SWIR)
        blueList.append(np.where(NDVI > NDWI, NDVI, NDWI*10))
        # flag.append(clf.predict(data).reshape((2400, 2400) | ~validFlag))
        flag.append(detectCloud(data) | ~validFlag) 

    # flag : 0-not cloud, 1-cloud
    switchFlag = (flag[0] == 0) & ((flag[1] == 1) | (zenith[0] < zenith[1]))
    flag = flag[0] & flag[1]
    # flag = cv2.resize(flag.astype(np.int8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST).astype(bool)
    # switchFlag = cv2.resize(switchFlag.astype(np.int8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST).astype(bool)
    blue = np.where(switchFlag, blueList[0], blueList[1])
    blue = np.where(flag, -200, blue)

    return blue, ~flag * (~switchFlag + 1)

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
        im_proj = dataset.GetProjection()  # 获取投影信息
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
        if bands.ndim == 2:
            bands = bands[:, :, None]
        # 设置影像保存大小、波段数
        band1 = bands[:, :, 0]
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.shape[2]

        # 设置保存影像的数据类型
        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_Int16
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
GAPath = './MOD09GA_ST/'
regionList = ['.h27v05', '.h28v06', '.h28v05', '.h27v04']
tifFileAfterList = [[224, 229], [223, 225], [224, 226], [229, 233]]
tifFileBeforeList = [[212, 207], [219, 207], [217, 213], [218, 207]]
H = W = 2400
tifPath = './MOD09GA/'
GQ = 'D:/Data/MOD09GQ/'
qc = './QC_500m/'
allWList = os.listdir(GQ)
allWList = [i for i in allWList if i[:2] == 'h2']
for i in range(4):
    tifFileBefore = tifFileBeforeList[i]
    tifFileAfter = tifFileAfterList[i]
    region = regionList[i]
    allW = GQ + [x for x in allWList if x[:6] == region[1:]][0]

    for i in range(2):
        if i == 1:
            date = tifFileBefore
            s, e = date[0], date[1] 
            date = [str(i)+region for i in range(s, e-1, -1)]
            name = 'before'
        else:
            date = tifFileAfter
            s, e = date[0], date[1] 
            date = [str(i)+region for i in range(s, e+1)]
            name = 'after'

        # 读取所有数据，得到去云之后的图像
        landMask = np.zeros((H, W, len(date)), np.int8)
        x = y = range(H)
        xx, yy = np.meshgrid(x, y) 
        zz = np.zeros_like(xx, np.int8)
        flag = np.ones((H, W), bool)
        blue = np.ones_like(landMask, np.float32) * -200

        for i, word in enumerate(date):
            zenithList = []
            tifList = []

            for fileName in os.listdir(GAPath):
                if word in fileName:
                    print(fileName)
                    zenithList.append(GAPath+fileName)
                    tifList.append(tifPath+fileName)

            # find better from MOD and MYD
            b, tmpLandMask = findBetter(zenithList, tifList)
            blue[:, :, i] = b
            zz += (tmpLandMask > 0)
            landMask[:, :, i] = tmpLandMask
            flag = flag & (tmpLandMask == 0)
        print(np.sum(flag))
        # print(np.isinf(blue).any())

        # 根据蓝波段选择最优值
        blue_sort = np.sort(blue, axis=-1) # 从小到大排
        blue_sort = blue_sort[:, :, ::-1] # 从大到小排
        # medianB = blue_sort[yy, xx, np.where(zz, zz-1, 0)] # 取最小值
        # medianB = blue_sort[yy, xx, zz//2] # 取中值中的小值
        medianB = blue_sort[yy, xx, 0]

        # 得到最优值所在的时间
        NO = np.zeros((H, W), np.int8)
        flag1 = np.ones((H, W), bool)
        for i in range(len(date)):
            tmpFlag = (blue[:, :, i] == medianB) & (landMask[:, :, i] > 0)
            NO += np.where(tmpFlag & flag1, 2*i+landMask[:, :, i]-1, 0)
            flag1 = flag1 & (tmpFlag == 0)
        print(np.sum(flag1))

        # 根据最优时间合成图像
        # bandOrder = [13, 14, 11, 12, 15, 16, 17]
        bandsGA = np.zeros((H, W, 7), np.int16)
        bandsTmp = np.zeros_like(bandsGA)
        for i, word in enumerate(date):
            for fileName in os.listdir(tifPath):
                if word in fileName:
                    ds = gdal.Open(tifPath+fileName)
                    for j in range(7):
                        band = ds.GetRasterBand(j+1).ReadAsArray()
                        # if idx == 1:
                        #     band = cv2.resize(band, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                        bandsTmp[:, :, j] = band
                    sameDateFlag = (NO == 2*i+ (0 if 'MOD'in fileName else 1))
                    bandsGA += np.int16(bandsTmp * sameDateFlag[:, :, None])
        bandsGA = np.where(flag[:, :, None], -200, bandsGA)

        maxVis = np.max(bandsGA[:, :, :3], axis=-1)
        maxSWIR = np.max(bandsGA[:, :, 5:], axis=-1)
        WI = maxVis > maxSWIR
        # kernel = np.ones((3, 3), np.float32)
        # WI = cv2.filter2D(WI.astype(np.uint8), -1, kernel)
        # WI = np.where(WI > 3, 200, 0)

        # 保存图像，并写入地理信息
        geotrans, proj = readImage('D:\\Data\\MOD09GQ\\'+date[0].split('.')[1]+'_AllWDays_percent.tiff')
        # geotrans /= np.array([1, 2, 1, 1, 1, 2])
        writeImage(bandsGA, './output/'+date[0].split('.')[1]+'_'+name+'.tiff', \
            geotrans=geotrans, proj=proj)
        writeImage(WI, './output/'+date[0].split('.')[1]+'_'+name+'_WI.tiff', \
            geotrans=geotrans, proj=proj)

