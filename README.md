# MOD09GA

browser.py      --利用爬虫，批量下载数据
splice.py       --对时序遥感数据MOD09GA去云并拼接出完整的图
getState.py      --从hdf文件中提取标志位

hdf2tiff.py     --从hdf中提取波段并写为tiff，以提取ROI
txt2csv.py      --将ROI生成的txt文件转换成模型训练可用的csv
detectCloud2.0.py   --利用sklearn库训练模型
pkl2tiff.py     --用训练好的模型分类整幅图

CloudDetect.py  --利用网上的数据集训练决策树，并画图

avhrr1.0.py     --处理avhrr数据
