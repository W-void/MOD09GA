#%%
from sklearn.externals import joblib
from utilze import *
import pandas as pd


print(os.getcwd())
#%%
p = '.\MOD09GA\MYD09GA.A2019229.h27v05.006.2019231025638.tiff'
ds = gdal.Open(p)
bands_num = ds.RasterCount
data = pd.DataFrame()
for i in range(bands_num):
    if i == 5:
        continue
    bi = ds.GetRasterBand(i+1).ReadAsArray()
    name = 'B' + str(i+1)
    data[name] = bi.flatten()

# vis = np.int16(data[['B1', 'B2', 'B3']])
# data['vis'] = vis.sum(axis=1)
# data['white'] = vis.var(axis=1)
pkl = joblib.load('./pkl/train_model.pkl')
clf = pkl['LDA']
y = clf.predict(data).reshape((2400, 2400))
writeImage(1-y, './log/229_LDA.tiff')
# for name, clf in pkl.items():
#     print(name)
#     y = clf.predict(data).reshape((2400, 2400))
#     writeImage(1-y, './log/229_'+name+ '.tiff')
#%%
