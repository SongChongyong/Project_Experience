# coding:utf-8 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, LSTM
from keras.models import Model
import time
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================== (1)数据读取及处理 ==============================
# 用pandas读取csv文件为DataFrame
data = pd.read_csv('./data_stocks.csv')
# describe()函数查看特征的数值分布
data.describe()

# 
data.info()
'''
输出结果为：
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 41266 entries, 0 to 41265             # 数据共41266行
Columns: 502 entries, DATE to NYSE.ZTS            # 数据共502列，502列分别为：DATE(时间)，SP500(大盘指数)，其他(有500列，分别为:NASDAQ.AAL, NASDAQ.AAPL,...)
dtypes: float64(501), int64(1)
memory usage: 158.0 MB
'''

# head()默认显示前5行的数据
data.head()
'''
DATE    SP500    NASDAQ.AAL    NASDAQ.AAPL    NASDAQ.ADBE    NASDAQ.ADI    NASDAQ.ADP    NASDAQ.ADSK ...
1491226200    2363.6101    42.33    143.68    129.63    82.04    102.23    85.22 ...
1491226260    2364.1001    42.36    143.7    130.32    82.08    102.14    85.65 ...
1491226320    2362.6799    42.31    143.6901    130.225    82.03    102.2125    85.51 ...
1491226380    2364.3101    42.37    143.64    130.0729    82    102.14    85.4872 ...

'''

# 查看时间跨度
print(time.strftime('%Y-%m-%d', time.localtime(data['DATE'].max())),     #获取最大时间
      time.strftime('%Y-%m-%d', time.localtime(data['DATE'].min())))     #获取最小时间
# 2017-09-01 2017-04-03

# 绘制大盘趋势折线图
plt.plot(data['SP500'])         # 把SP500列的数据取出来，绘制折线图，见'./pictures_lstm_stocks/Figure_1_大盘趋势折线图.png'
# plt.show()


# ==== 去掉DATE一列，训练集测试集分割
data.drop('DATE', axis=1, inplace=True)                 # 去掉DATE列
data_train = data.iloc[:int(data.shape[0] * 0.8), :]    # data.shape(0)表示取data的行数
                                                        # ':int(data.shape[0] * 0.8)'表示从开始到0.8倍data行数的行(即前80%的行)，':'表示取所有列
                                                        # 将[前80%行,所有列]构成data_train
data_test = data.iloc[int(data.shape[0] * 0.8):, :]     # 将[后20%行,所有列]构成data_test
print(data_train.shape, data_test.shape)                # 打印data_train和data_test的结构：(33012, 501) (8254, 501)
                                                        # data_train为33012行，501列的数组；data_test为8254行，501列的数组

'''
print(data_train)
SP500  NASDAQ.AAL  NASDAQ.AAPL  NASDAQ.ADBE  NASDAQ.ADI  ...
0      2363.6101     42.3300     143.6800     129.6300     82.0400   ...
1      2364.1001     42.3600     143.7000     130.3200     82.0800   ...
...  
33010  2474.8601     50.5200     157.8701     146.6000     78.8800   ...
33011  2474.6201     50.5200     157.8000     146.5300     78.8600   ...
'''
# ==== 数据归一化
# 用fit()函数把data.train和data_test的数据映射到(-1,1)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
'''
print(data_train)  
[[-0.5515494  -0.78013925 -0.6376737  ..., -0.89728097 -0.24863686 -0.81802426]
 ..., 
 [ 0.8926691   0.42030048  0.81574884 ...,  0.98716012 -0.70338059 0.60138648]]
 
'''
# 最终的data_train为33012行, 501列，且元素都在(-1,1)的列表；
# data_test为8254行, 501列，且元素都在(-1,1)的列表；


# ============================== (3)用keras实现异步预测==============================
'''
异步预测:用历史若干个时刻的大盘指数，预测当前时刻的大盘指数
例如，使用前五个大盘指数，预测当前的大盘指数，每组输入包括5个step，
每个step对应⼀一个历史时刻的大盘指数，输出⼀一维，即[None=数据个数, 5, 1] => [None数据个数, 1]
使用Keras实现异步预测，主要⽤用到循环神经网络即RNN（Recurrent Neural Network）中的LSTM（Long Short Term Memory）
'''
output_dim = 1
batch_size = 256
epochs = 10
seq_len = 5
hidden_size = 128  # 这里就一层神经网络

# 以X_train=0-4行数据，Y_train=第5行数据；
#   X_train=1-5行数据，Y_train=第6行数据；
#   ...这样循环遍历data_train (对data_test同理)
X_train = np.array([data_train[i : i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])[:, :, np.newaxis]
y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0]- seq_len)])
X_test = np.array([data_test[i : i + seq_len, 0] for i in range(data_test.shape[0]- seq_len)])[:, :, np.newaxis]
y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# (33007, 5, 1) (33007,) (8249, 5, 1) (8249,)

X = Input(shape=[X_train.shape[1], X_train.shape[2],])
h = LSTM(hidden_size, activation='relu')(X)
Y = Dense(output_dim, activation='sigmoid')(h)

model = Model(X, Y)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
y_pred = model.predict(X_test)
print('MSE Train:', model.evaluate(X_train, y_train, batch_size=batch_size))
print('MSE Test:', model.evaluate(X_test, y_test, batch_size=batch_size))
plt.plot(y_test, label='test')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()

'''
Epoch 1/10

  256/33007 [..............................] - ETA: 1:50 - loss: 1.3914
 1280/33007 [>.............................] - ETA: 22s - loss: 1.1924 
 ...
32256/33007 [============================>.] - ETA: 0s - loss: 0.3344
33007/33007 [==============================] - 3s 89us/step - loss: 0.3279
Epoch 2/10

  256/33007 [..............................] - ETA: 2s - loss: 0.4575
 1280/33007 [>.............................] - ETA: 1s - loss: 0.3608
 ...
32512/33007 [============================>.] - ETA: 0s - loss: 0.1082
33007/33007 [==============================] - 2s 62us/step - loss: 0.1066

Epoch 3/10
...
Epoch 10/10

  256/33007 [..............................] - ETA: 1s - loss: 0.4574
 1280/33007 [>.............................] - ETA: 2s - loss: 0.3601
 2304/33007 [=>............................] - ETA: 1s - loss: 0.3657
 3072/33007 [=>............................] - ETA: 1s - loss: 0.4202
 4096/33007 [==>...........................] - ETA: 1s - loss: 0.4933
 5120/33007 [===>..........................] - ETA: 1s - loss: 0.5110
 6144/33007 [====>.........................] - ETA: 1s - loss: 0.4695
 7168/33007 [=====>........................] - ETA: 1s - loss: 0.4088
 7936/33007 [======>.......................] - ETA: 1s - loss: 0.3747
 8960/33007 [=======>......................] - ETA: 1s - loss: 0.3389
 9984/33007 [========>.....................] - ETA: 1s - loss: 0.3061
11008/33007 [=========>....................] - ETA: 1s - loss: 0.2802
12032/33007 [=========>....................] - ETA: 1s - loss: 0.2579
12800/33007 [==========>...................] - ETA: 1s - loss: 0.2562
13824/33007 [===========>..................] - ETA: 1s - loss: 0.2423
14848/33007 [============>.................] - ETA: 1s - loss: 0.2259
15872/33007 [=============>................] - ETA: 1s - loss: 0.2114
16640/33007 [==============>...............] - ETA: 1s - loss: 0.2016
17664/33007 [===============>..............] - ETA: 0s - loss: 0.1900
18432/33007 [===============>..............] - ETA: 0s - loss: 0.1821
19456/33007 [================>.............] - ETA: 0s - loss: 0.1725
20480/33007 [=================>............] - ETA: 0s - loss: 0.1639
21504/33007 [==================>...........] - ETA: 0s - loss: 0.1561
22528/33007 [===================>..........] - ETA: 0s - loss: 0.1490
23552/33007 [====================>.........] - ETA: 0s - loss: 0.1425
24576/33007 [=====================>........] - ETA: 0s - loss: 0.1366
25600/33007 [======================>.......] - ETA: 0s - loss: 0.1311
26624/33007 [=======================>......] - ETA: 0s - loss: 0.1261
27648/33007 [========================>.....] - ETA: 0s - loss: 0.1214
28672/33007 [=========================>....] - ETA: 0s - loss: 0.1171
29696/33007 [=========================>....] - ETA: 0s - loss: 0.1131
30720/33007 [==========================>...] - ETA: 0s - loss: 0.1094
31744/33007 [===========================>..] - ETA: 0s - loss: 0.1059
32768/33007 [============================>.] - ETA: 0s - loss: 0.1027
33007/33007 [==============================] - 2s 62us/step - loss: 0.1019

  256/33007 [..............................] - ETA: 12s
 4352/33007 [==>...........................] - ETA: 1s 
 7936/33007 [======>.......................] - ETA: 0s
11520/33007 [=========>....................] - ETA: 0s
15872/33007 [=============>................] - ETA: 0s
18944/33007 [================>.............] - ETA: 0s
22272/33007 [===================>..........] - ETA: 0s
25600/33007 [======================>.......] - ETA: 0s
28928/33007 [=========================>....] - ETA: 0s
33007/33007 [==============================] - 1s 18us/step
MSE Train: 0.10340568840537946

 256/8249 [..............................] - ETA: 0s
3840/8249 [============>.................] - ETA: 0s
7424/8249 [=========================>....] - ETA: 0s
8249/8249 [==============================] - 0s 16us/step
MSE Test: 0.0040789693311443454
'''

