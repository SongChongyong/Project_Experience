# coding:utf-8 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================== (1)数据读取 ==============================
# 用pandas读取csv文件为DataFrame
data = pd.read_csv('./data_stocks.csv')
# describe()函数查看特征的数值分布
# data.describe()

# 打印数据的基本信息
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
plt.xlabel("Time")
plt.ylabel("SP500")
plt.title("SP500 time series")
# plt.show()

# ============================== (2)数据处理 ==============================
# 用pandas的各种函数处理数据：
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


# ============================== (3)用Tensorflow实现异步预测==============================
'''
异步预测:用历史若干个时刻的大盘指数，预测当前时刻的大盘指数
例如，使用前五个大盘指数，预测当前的大盘指数，每组输入包括5个step，
每个step对应⼀一个历史时刻的大盘指数，输出⼀一维，即[None=数据个数, 5, 1] => [None数据个数, 1]
使用Tensorflow实现异步预测，
'''




# model = Model(X, Y)
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
# y_pred = model.predict(X_test)
# print('MSE Train:', model.evaluate(X_train, y_train, batch_size=batch_size))
# print('MSE Test:', model.evaluate(X_test, y_test, batch_size=batch_size))
# plt.plot(y_test, label='test')
# plt.plot(y_pred, label='pred')
# plt.legend()
# plt.show()



# input_size = 
output_size = 1
batch_size = 256
epochs = 10
seq_len = 5
state_size = 128

# 以X_train=0-4行数据，Y_train=第5行数据；
#   X_train=1-5行数据，Y_train=第6行数据；
#   ...这样循环遍历data_train (对data_test同理)
X_train = np.array([data_train[i : i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])[:, :, np.newaxis]
y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0]- seq_len)])
X_test = np.array([data_test[i : i + seq_len, 0] for i in range(data_test.shape[0]- seq_len)])[:, :, np.newaxis]
y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# (33007, 5, 1) (33007,) (8249, 5, 1) (8249,)




# Keras：
# X = Input(shape=[X_train.shape[1], X_train.shape[2],])
# h = LSTM(hidden_size, activation='relu')(X)                                  
# Y = Dense(output_dim, activation='sigmoid')(h)



# X = tf.placeholder(shape=[None], dtype=tf.float32)
# h = tf.placeholder(shape=[None],dtype=tf.float32)
# Y = tf.placeholder(shape=[None], dtype=tf.float32)

state_size_1 = 1024
state_size_2 = 512
state_size_3 = 256
state_size_4 = 128




                     

lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=1024)
X_1 = tf.placeholder(shape=[None], dtype=tf.float32)
h_0 = lstm_cell_1.zero_state(256, dtype=tf.float32)
output_1, h_1 = lstm_cell_1.__call__(X_1, h_0)                # 这里的output_1还不是输出
V_1 = tf.get_variable('V_1',[state_size_1,output_size],initializer=tf.contrib.layers.xavier_initializer(seed=1)) 
Y_1 = tf.nn.relu(tf.matmul(output_1, V_1))

Y = tf.placeholder(shape=[None], dtype=tf.float32)


cost = tf.reduce_mean(tf.squared_difference(Y_1, Y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for e in range(epochs):
#不需要打乱数据         shuffle_indices = np.random.permutation(np.arange(y_train.shape[0]))   #打乱数据
        X_train = [X_train[1],X_train[2]]
        y_train = [y_train[1],y_train[2]]
        
        for i in range(y_train.shape[0] // batch_size):
            start = i * batch_size
            batch_x = X_train[start : start + batch_size]
            batch_y = y_train[start : start + batch_size]
            sess.run(optimizer, feed_dict={X_1: batch_x, Y: batch_y})
            
            if i % 500 == 0:
                print('MSE Train:', sess.run(cost, feed_dict={X: X_train, Y: y_train}))
                print('MSE Test:', sess.run(cost, feed_dict={X: X_test, Y: y_test}))
                print('MSE Test:', sess.run(cost, feed_dict={X: X_test, Y: y_test}))
                
                y_pred = sess.run(out, feed_dict={X: X_test})
                y_pred = np.squeeze(y_pred)
                plt.plot(y_test, label='test')
                plt.plot(y_pred, label='pred')
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                plt.legend()
                plt.show()


# 保存计算图
# 使用tf.get_default_graph()保存默认的TensorBoard,事件文件保存在'./Lstm_stocks_Board'下
# 而后可以在命令行输入: tensorboard --logdir=./Lstm_stocks_Board启动tensorboard,
# 然后在浏览器中查看张量的计算图(见Lstm_stocks_Board.png)
writer = tf.summary.FileWriter(logdir='./Lstm_stocks_Board',graph=tf.get_default_graph())
writer.flush()