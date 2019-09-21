import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf

random.seed(666)
np.random.seed(666)

PATH = "..//train_set/"
# 选取最大文件数
SELECT_FILES = 100
# 最大文件数
MAX_FILES = 4000
# 方差值小于15的特征会被剔除
LITTLE_VARIANCE = 15
# 学习率
LEARN_RATE = 0.01

# 加载csv数据
def load_csv_file(path):
    return remove_head(np.loadtxt(path, delimiter=",", dtype=str))


# 移除表头
def remove_head(data):
    return data[1:, :]


# 随机读取200个文件到内存中
def read_resoures():
    files = os.listdir(PATH)
    random_index = np.random.randint(0, MAX_FILES, SELECT_FILES)
    files = np.array(files)
    return files[random_index]


# 把获取的数据读取到data中[200,2291,18]
def load_memory(file_names):
    datas = []
    for index in file_names:
        data = load_csv_file(PATH + index)
        datas.append(data)

    result = []
    for i1 in datas:
        for i2 in i1:
            for i3 in i2:
                result.append(i3)

    result = np.array(result)
    result = result.reshape(-1, 18)

    return result


# 随机
data = load_memory(read_resoures())
np.random.shuffle(data)

# print(len(data))

label_data = data[:, -1]
futures_data = data[:, 0:-1]
# print(label_data.shape) # 标签形状 (2861,)
# print(futures_data.shape) # 数据形状(2861,17)

futures_data = futures_data.astype(np.float64)
# print(futures_data.shape)  # (2861, 17)


# hv 是栅格与发射机的距离d以及栅格与信号线的相对高度
# 计算hv
# 3 h_b 发射机相对地面的高度h_b
# 6 sit_md 下倾角
# 5 sit_ed 垂直电下倾角
# 1 position_x 发射机所在x坐标
# 2 position_y 发射机所在y坐标
# 12 x 目标所在x
# 13 y 目标所在y
def culter_hv(h_b, sit_md, sit_ed, position_x, position_y, x, y):
    result_x = np.power(x - position_x, 2)
    result_y = np.power(y - position_y, 2)
    d = np.sqrt(result_x + result_y)

    tan = np.tan(np.radians(sit_md + sit_ed))
    h_v = h_b - tan * d
    return h_v,d


# 计算hvs 返回的是(2861,)
def culter_hvs(futures_data):
    return culter_hv(futures_data[:, 3], futures_data[:, 6], futures_data[:, 5], futures_data[:, 1], futures_data[:, 2],
                     futures_data[:, 12], futures_data[:, 13])


hv,d = culter_hvs(futures_data)  # (2861,)

futures_data = np.delete(futures_data, [ 5, 6, 1, 2, 12, 13], axis=1)  # (2861, 10)

# 融入特征hv
futures_data = np.c_[futures_data, hv]  # (2861, 11)
futures_data = np.c_[futures_data , d]


# 方差
def variance(futures_data):
    mean_futures = np.mean(futures_data, axis=0)
    a = futures_data - mean_futures
    b = np.power(a, 2)
    c = np.sum(b, axis=0)
    d = c / len(futures_data)
    return d


# 获取小于阈值的index
def getIndex(vari):
    size = vari.size
    indexs = []
    for index in range(size):
        if (vari[index] < LITTLE_VARIANCE):
            indexs.append(index)
    return indexs


vari = variance(futures_data) # (12,)
delete_index = getIndex(vari)

futures_data = np.delete(futures_data, delete_index, axis=1) # (*,10)
# print(futures_data.shape)

# print(futures_data.shape[0])

rows = 100000






# 建立模型
# f = tf.placeholder(dtype=tf.float32 ,shape=(100000,1))
# h_b = tf.placeholder(dtype=tf.float32 , shape=(100000,1))
alpha = tf.Variable(initial_value=1,dtype=tf.float32)
# h_ue = tf.placeholder(dtype=tf.float32,shape=(100000,1))
# d = tf.placeholder(dtype=tf.float32,shape=(100000,1))
RSRP = tf.placeholder(dtype=tf.float32,shape=(100000,1))
p_t = tf.placeholder(dtype=tf.float32,shape=(100000,1))

X = tf.placeholder(dtype=tf.float32 , shape=(rows , 10))
W = tf.Variable(initial_value=tf.random.normal(shape=(10,1)) ,dtype=tf.float32)


PL =  tf.matmul(X,W) + alpha
# PL = tf.nn.relu(PL)
RSRP_PRE = p_t - PL
#RSRP_PRE = tf.nn.relu(RSRP_PRE)
loss = tf.sqrt(tf.reduce_mean(tf.square(RSRP_PRE - RSRP)))
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=RSRP , logits=RSRP_PRE)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
train_op = optimizer.minimize(loss)

init_op = tf.group(tf.global_variables_initializer())

# 测试集的op
test_op = RSRP_PRE


x = futures_data[0:100000,3].reshape(-1,1)
# print(x.shape)

label_data = label_data.reshape(-1,1)
# print(label_data)

zz = data[0:rows, 8].reshape(100000, 1).astype(np.float64)


with tf.Session() as sess:
    sess.run(init_op)

    for step in range(10000):
        # _ , loss_cur = sess.run([train_op , loss],feed_dict={
        #     f:futures_data[100*step:100*step+100,3].reshape(100,1),
        #     h_b:futures_data[100*step:100*step+100,1].reshape(100,1),
        #     h_ue:futures_data[100*step:100*step+100,-2].reshape(100,1),
        #     d:futures_data[100*step:100*step+100,-1].reshape(100,1),
        #     RSRP:label_data[100*step:100*step+100,0].reshape(100,1),
        #     p_t:data[100*step:100*step+100,8].reshape(100,1).astype(np.float64)
        # })
        # _ , loss_cur = sess.run([train_op , loss],feed_dict={
        #     f:futures_data[0:100000,3].reshape(100000,1),
        #     h_b:futures_data[0:100000,1].reshape(100000,1),
        #     h_ue:futures_data[0:100000,-2].reshape(100000,1),
        #     d:futures_data[0:100000,-1].reshape(100000,1),
        #     RSRP:label_data[0:100000,0].reshape(100000,1),
        #     p_t:data[0:100000,8].reshape(100000,1).astype(np.float64)
        # # })

        _, loss_cur=sess.run([train_op, loss], feed_dict={
            X:futures_data[0:rows,:],
            RSRP: label_data[0:rows, 0].reshape(100000, 1),
            p_t: data[0:rows, 8].reshape(100000, 1).astype(np.float64)
        })

        if step % 100 == 0:
            print(loss_cur)
    print(label_data[rows:rows+100000,:])
    pre = sess.run([test_op], feed_dict={
        X: futures_data[rows:rows+100000, :],
        p_t: data[rows:rows+100000, 8].reshape(100000, 1).astype(np.float64)
    })

    print(pre)