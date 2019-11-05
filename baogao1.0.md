# 台风利奇马对湖泊江流的影响

## 1. 利奇马介绍

台风“利奇马”是2019年对中国影响最大的台风。
8月10日1时，“利奇马”在浙江省温岭市城南镇沿海登陆，当日22时许，“利奇马”由浙江移入江苏境内。
8月11日12时许，“利奇马“从江苏省连云港市附近出海，移入黄海海面，并继续向偏北方向移动，向山东半岛南部沿海靠近。当日20时50分许，台风“利奇马”的中心在山东省青岛市黄岛区沿海再次登陆
8月12日2时许，联合台风警报中心认定其已转化为副热带风暴。当日5时许，“利奇马”穿过山东半岛移入莱州湾海面，并开始回旋打转，强度继续减弱。
8月13日14时，中央气象台对其停止编号。

![avatar](./picture/2.png)

## 2. 评估方法

对时间序列图像进行去云操作，并分别合成台风过境前后两幅图像，通过对比两幅图像的水指数，对受灾情况进行评估。

## 3. 数据来源

研究采用MOD09GA数据，该数据每天上午下午两景，上午下午各有一景；空间分辨率为500m；光谱上有7个波段，可见光3个波段，近红外2个波段，短波红外2个波段。

## 3. 数据处理

### 3.1 数据来源

研究采用MOD09GA数据，该数据每天两景，上午下午各有一景；空间分辨率为500m；光谱上有7个波段，可见光3个波段，近红外2个波段，短波红外2个波段。

### 3.2 数据下载

利用爬虫，在MODIS官网上下载了09GA从7月下旬至8月中旬的数据。

### 3.3 处理流程

MOD09GA数据，覆盖中国的有h27v04,h27v05,h28v05,h28v06，4景数据，每一景的台风过境时间都不一样，需要选择不同的时间。对于每天上午下午两景数据，除了选择无云的像素，还选择了天顶角小的像素。去云算法采用了分水岭算法，分水岭算法不仅阈值灵活，而且考虑了图像的空间信息，能较好地识别单幅图像的云。同一个像素在时间上可能有多个值没有受到云污染，对于有多个备选值的像素，如何选择备选值也是一个值得研究的点。暂时选择NDVI或NDWI大的时间点，合成最终的的图像。MOD09GA数据大气校正做的不是很好，对于低反射率的水，总是会出现负值，这增加了数据处理的难度。

## 4. 结果分析

### 4.1 江苏骆马湖及其下游水稻带，台风过境前后假彩色图及水指数图

（左边两幅为台风过境前，右边两幅为台风过境后）
<div align="center">
    <img src="./picture/5.png"  width="40%"><img src="./picture/6.png"  width="40%">
</div>
<div align="center">
    <img src="./picture/3.png"  width="40%"><img src="./picture/4.png"  width="40%">
</div>

<!-- <img src="./picture/5.png" width="30%" height="40%"><img src="./picture/6.png" width="30%" height="40%"> -->

<!-- <img src="./picture/3.png" width="30%" height="40%"><img src="./picture/4.png" width="30%" height="40%"> -->
<!-- ![avatar](./picture/5.png)![avatar](./picture/6.png)
![avatar](./picture/3.png)![avatar](./picture/4.png) -->

### 4.2 黄河下游，台风过境前后假彩色图及水指数图

（左边两幅为台风过境前，右边两幅为台风过境后）

<div align="center">
    <img src="./picture/7.png"  width="40%"><img src="./picture/8.png"  width="40%">
</div>
<div align="center">
    <img src="./picture/9.png"  width="40%"><img src="./picture/10.png"  width="40%">
</div>
<!-- <img src="./picture/7.png" width="30%" height="40%"><img src="./picture/8.png" width="30%" height="40%">

<!-- <img src="./picture/9.png" width="30%" height="40%"><img src="./picture/10.png" width="30%" height="40%"> --> -->
<!-- ![avatar](./picture/7.png)![avatar](./picture/8.png)
![avatar](./picture/9.png)![avatar](./picture/10.png) -->

### 4.3 长江下游，常州苏州杭州，太湖，滆湖，长荡湖，南湖，台风过境前后假彩色图及水指数图

（左边两幅为台风过境前，右边两幅为台风过境后）

<div align="center">
    <img src="./picture/11.png"  width="20%"><img src="./picture/12.png"  width="20%">
</div>
<div align="center">
    <img src="./picture/13.png"  width="20%"><img src="./picture/14.png"  width="20%">
</div>
<!-- <img src="./picture/11.png" width="30%" height="40%"><img src="./picture/12.png" width="30%" height="40%">

<!-- <img src="./picture/13.png" width="30%" height="40%"><img src="./picture/14.png" width="30%" height="40%"> --> -->
<!-- ![avatar](./picture/11.png)![avatar](./picture/12.png)
![avatar](./picture/13.png)![avatar](./picture/14.png) -->

### 4.4 浙江省台州温洲一带，台风过境前后假彩色图及水指数图

（左边两幅为台风过境前，右边两幅为台风过境后）

<div align="center">
    <img src="./picture/15.png"  width="20%"><img src="./picture/16.png"  width="20%">
</div>
<div align="center">
    <img src="./picture/17.png"  width="20%"><img src="./picture/18.png"  width="20%">
</div>

<!-- <img src="./picture/15.png" width="30%" height="40%"><img src="./picture/16.png" width="30%" height="40%">

<img src="./picture/17.png" width="30%" height="40%"><img src="./picture/18.png" width="30%" height="40%"> -->
<!-- ![avatar](./picture/15.png)![avatar](./picture/16.png)
![avatar](./picture/17.png)![avatar](./picture/18.png) -->
