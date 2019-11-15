#%%
import pandas as pd 
import numpy as np


#%%
txt = pd.read_csv('./csv/1105.txt', sep='\n', header=None)
txt.head()
numOfROI = int(txt.iloc[1][0].split()[-1])
tar = []
tarNum = []
for i in range(numOfROI):
    # tar.append(txt.iloc[4 + i * 4][0].split()[-1])
    tarNum.append(int(txt.iloc[4 + i * 4 + 2][0].split()[-1]))

idx = 4 * (numOfROI + 1)
data = txt.iloc[idx:, 0].str.split(expand=True)
columns = txt.iloc[idx-1, 0].split()[1:]
data.columns = columns
# data.rename(columns=dict(zip(data.columns, columns)), inplace=True)
data.drop(['ID', 'X', 'Y'], axis=1, inplace=True)
data.reset_index(drop=True,inplace=True)
# vis = np.int16(data[['B1', 'B2', 'B3']])
# data['vis'] = vis.sum(axis=1)
# data['white'] = vis.var(axis=1)

y = np.zeros(sum(tarNum), np.int8)
y[tarNum[0]:] = 1
data['class'] = y

data.to_csv('./csv/1105.csv', index=False)


#%%
