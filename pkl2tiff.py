#%%
from sklearn.externals import joblib
from utilze import *
import pandas as pd


print(os.getcwd())
#%%
pkl = joblib.load('./pkl/train_model.pkl')
clf = pkl['DT']
p = '../../Data/MOD09GA/MYD09GA.A2019229.h28v06.006.2019231025426.tiff'
ds = gdal.Open(p)
bands_num = ds.RasterCount
data = pd.DataFrame()
for i in range(bands_num):
    bi = ds.GetRasterBand(i+1).ReadAsArray()
    name = 'B' + str(i+1)
    data[name] = bi.flatten()

# vis = np.int16(data[['B1', 'B2', 'B3']])
# data['vis'] = vis.sum(axis=1)
# data['white'] = vis.var(axis=1)
y = clf.predict(data).reshape((2400, 2400))
writeImage(y, './log/test.tiff')
#%%
