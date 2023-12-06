# Hydrology

## Setup
Please do the following unless you know what you are doing:
```bash
git checkout -b <your_name>
```

Run the following command to install all the required packages:
```bash
conda create -n hydrology python=3.8.8
conda activate hydrology
pip install -r requirements.txt
unzip data_demo.zip
```

Download all the data from the following link and put them in the `data` folder:

[温度数据：temp_data.pkl](https://cloud.tsinghua.edu.cn/f/6b3764f9659a4aa4a542/?dl=1)

[降水数据：pcp_dat.pkl](https://cloud.tsinghua.edu.cn/f/197f0e300ee04b5da012/)

[温度-降水-经纬度集成数据：gpt_data.pkl](https://cloud.tsinghua.edu.cn/f/3ce61db42f3f459cb3c3/?dl=1) (If nothing unexpected happens, you should use this data only for now)

After finishing the above, your folder structure should look like this:
```
.
├── CMADS
│   ├── CMADSV1.1_station
│   │   ├── CMADS-area.dbf
│   │   ├── CMADS-area.prj
│   │   ├── CMADS-area.sbn
│   │   ├── CMADS-area.sbx
│   │   ├── CMADS-area.shp
│   │   ├── CMADS-area.shp.xml
│   │   ├── CMADS-area.shx
│   │   ├── CMADS1.1.dbf
│   │   ├── CMADS1.1.prj
│   │   ├── CMADS1.1.sbn
│   │   ├── CMADS1.1.sbx
│   │   ├── CMADS1.1.shp
│   │   ├── CMADS1.1.shx
│   │   └── CMADSV1.1.mxd
│   ├── CMADS_V1.1_Station_Information.xlsx
│   ├── CMADS_V1.1_User_Guide(Chinese)_update-2018.pdf
│   ├── For-swat-2012
│   │   ├── Fork
│   │   └── Station
│   ├── reference
│   │   ├── Hydrological Modeling in the Manas River Basin Using Soil and Water Assessment Tool Driven by CMADS.pdf
│   │   ├── Investigating spatiotemporal changes of the land.pdf
│   │   ├── The China Meteorological Assimilation Driving Datasets for the SWAT Model (CMADS) Application in China A Case Study in Heihe River Basin .pdf
│   │   └── water-09-00765.pdf
│   └── 中国轮廓
│       └── 国界
├── README.md
├── data
│   ├── pcp_data.pkl
│   └── temp_data.pkl
├── data_demo.zip
├── data_preprocessing.py
├── dataclean.ipynb
├── geology.ipynb
├── primary.ipynb
└── 开题.pdf
```


### Quick Onboard
preprocess.py: preprocess the data, generate pkl files
utils.py: some useful functions (TODO: hotmap API)
main,py: generate the results


## 方法：raw

#### 时间地点

选取时间：2008-2016

选取地点：73-135°E，18-54°N，共35712个测站

#### 初步计划

取消空间聚类、降维，



提取该区域内降水量特征（年均值，年际方差 etc.)、温度、地形特征，后续可增加


##### 整体，不分区

获取整个区域内的情况(已完成)



##### 以降水量为特征的方法

1. 按提取的年降水量特征分区（0-200，200-400，400-800， 800-1600）
2. 以0.5℃为一档，每档取90，95，99 percentile以上，删去300条数据以下的区间
3. 对于每个分区，分析极端降水与温度的关系，绘图


##### 以地理位置为因子的方法

1. 以 4 * 4 的地理位置对测站进行直接聚类，此时数据代表1°*1°的区域

2. $$9 \times 365 \times 16 = 52560$$条日频的降水和温度

3. 以0.5℃为一档，每档取90，95，99 percentile以上，删去300条数据以下的区间

4. 对于每个点，线性回归；绘制地图




##### 重复可获得以海拔、年均温、等其他特征为因子的极端降水与温度关系



#### 对温度-降水回归因子$$\beta_1$$影响因素定量分析：

1. 对于每个测站，在指定温度范围内，每隔0.5℃找到top5% percent；样本数量少，则采用随机数决定是否选取
2. 线性回归，获得每个测站的$$\beta_1$$
3. 以$$\beta_1$$作为y，以之前提取的特征（降水量特征（年均值，年际方差 etc.)、温度、地形特征）为X，做整体预测



