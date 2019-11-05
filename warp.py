#%%
from osgeo import gdal
import shapefile

#%%
root_ds = gdal.Open('F:\MOD09GA\MOD09GA.A2019207.h28v05.006.2019209031253.hdf')
# 返回结果是一个list，list中的每个元素是一个tuple，每个tuple中包含了对数据集的路径，元数据等的描述信息
# tuple中的第一个元素描述的是数据子集的全路径
ds_list = root_ds.GetSubDatasets()

# 取出第1个数据子集（MODIS反射率产品的第一个波段）进行转换
# 第一个参数是输出数据，第二个参数是输入数据，后面可以跟多个可选项
gdal.Warp('./reprojection.tif', ds_list[11][0], dstSRS='EPSG:32649')

# 关闭数据集
root_ds = None

#%%
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

#输入为待操作shp的路径，和一组待转换的投影坐标
def prj2geo(path,x,y):
    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # 为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCODING", "")
    # 注册所有的驱动
    ogr.RegisterAll()
    # 数据格式的驱动
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(path)
    layer0 = ds_simple.GetLayerByIndex(0)

    #或取到shp的投影坐标系信息
    prosrs = layer0.GetSpatialRef()
    geosrs = osr.SpatialReference()
    #设置输出坐标系为WGS84
    geosrs.SetWellKnownGeogCS("WGS84")

    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    #输出为转换好的经纬度
    return coords[:2]


#%%
shpPath = 'D:\Data\MOD09GQ\china-board\China_Province_Proj.shp'
sf = shapefile.Reader(tpath)

#%%
from geopandas import *

shpPath = 'D:\Data\MOD09GQ\china-board\China_Province_Proj.shp'
shp_df = GeoDataFrame.from_file(shpPath)
shp_df.head()
shp_df.plot()
shp_df.crs
# {'datum': 'WGS84',
#  'lat_0': 30,
#  'lat_1': 55,
#  'lat_2': 15,
#  'lon_0': 110,
#  'no_defs': True,
#  'proj': 'aea',
#  'units': 'm',
#  'x_0': 0,
#  'y_0': 0}
shp_df.to_crs(from_epsg(4326))

#%%
import rasterio as rio
from rasterio.warp import (reproject,RESAMPLING, transform_bounds,calculate_default_transform as calcdt)

tifPath = 'D:\data\MOD09GQ\h28v05_AllWDays_percent.tiff'
# tifPath = './output2/h28v05_after.tiff'
src = rio.open(tifPath)

affine, width, height = calcdt(src.crs, shp_df.crs, src.width, src.height, *src.bounds)
kwargs = src.meta.copy()
kwargs.update({
    'crs': shp_df.crs,
    'transform': affine,
    'affine': affine,
    'width': width,
    'height': height,
    'geotransform':(0,1,0,0,0,-1) ,
    'driver': 'GTiff'
})

newtiffname = 'D:\Data\MOD09GQ/test.tiff'
dst = rio.open(newtiffname, 'w', **kwargs)

for i in range(1, src.count + 1):
    reproject(
        source = rio.band(src, i), 
        destination = rio.band(dst, i), 
        src_transform = src.affine,
        src_crs = src.crs,
        dst_transform = affine,
        dst_crs = shp_df.crs,
        dst_nodata = src.nodata,
        resampling = RESAMPLING.bilinear)
dst.close()


#%%
from geopandas import GeoSeries
features = [shp_df.geometry.__geo_interface__]

import rasterio.mask
src = rio.open('D:/Data/MOD09GQ/test.tiff')
out_image, out_transform = rio.mask.mask(src, features, crop=True)
out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})
band_mask = rasterio.open(newtiffname, "w", **out_meta)
band_mask.write(out_image)
band_mask.close()

#%%
