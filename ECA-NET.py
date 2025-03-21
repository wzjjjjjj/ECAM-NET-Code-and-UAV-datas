#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import scipy.io.wavfile as wav
from python_speech_features import mfcc, delta
import os
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from spafe.features.gfcc import gfcc
import math


# 添加批处理功能的函数

def next_batch(num, data1, data2, labels):
    '''
    返回从两个数据集中随机选择的`num`个样本和标签。
    `data1` 和 `data2` 应该有相同的样本数，且对应的样本相关联。
    '''
    idx = np.arange(0, len(data1))  # 假设 data1 和 data2 长度相同
    np.random.shuffle(idx)
    idx = idx[:num]
    data1_shuffle = [data1[i] for i in idx]
    data2_shuffle = [data2[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data1_shuffle), np.asarray(data2_shuffle), np.asarray(labels_shuffle)


path_film = os.path.abspath('.')
path = path_film + "/UAV/ddd/"  # 训练数据所在的文件夹路径
test_path = path_film + "/UAV/test_data/"
val_path=path_film + "/UAV/val_data/"
label_binarizer = ""


def def_one_hot(x):
    if label_binarizer == "":
        binarizer = sklearn.preprocessing.LabelBinarizer()
    else:
        binarizer = label_binarizer
    y = binarizer.fit_transform(x)
    return y


def read_wav_path(path):
    map_path = []
    labels = []
    for x in os.listdir(path):
        [map_path, labels] = circle(path, x, map_path, labels)
    return map_path, labels


def circle(path, x, map_path, labels):
    if os.path.isfile(str(path) + str(x)):
        map_path.append(str(path) + str(x))
    else:
        for y in os.listdir(str(path) + str(x) + "/"):
            labels.append(x)
            circle(str(path) + str(x) + "/", y, map_path, labels)  # 实现对每个子文件夹下的所有wav文件的路径和类别标签的获取。这里递归调用circle函数，注意理解
    return map_path, labels


def def_wav_read_mfcc(file_name):
    fs, audio = wav.read(file_name)
    mfcc_features = mfcc(audio, samplerate=fs)
    gfcc_features = gfcc(audio, fs=fs, num_ceps=13)
    min_length = min(mfcc_features.shape[0], gfcc_features.shape[0])
    norm_mfcc_features = mfcc_features[:min_length]
    norm_gfcc_features = gfcc_features[:min_length]
    return norm_mfcc_features, fs


def def_wav_read_gfcc(file_name, fs):
    _, audio = wav.read(file_name)
    gfcc_features = gfcc(audio, fs=fs, num_ceps=13)
    return gfcc_features


def find_matrix_max_shape(audio):
    h, l = 0, 0
    for a in audio:
        a, b = np.array(a).shape
        if a > h:
            h = a
        if b > l:
            l = b
    return h, l


def matrix_make_up(audio):
    h, l = find_matrix_max_shape(audio)
    new_audio = []
    for aa in audio:
        a, b = np.array(aa).shape
        zeros_matrix = np.zeros([h, l], np.int8)
        for i in range(a):
            for j in range(b):
                zeros_matrix[i, j] = zeros_matrix[i, j] + aa[i, j]
        new_audio.append(zeros_matrix)
    return new_audio, h, l


def read_wav_matrix(path):
    map_path, labels = read_wav_path(path)
    audio = []
    for idx, folder in enumerate(map_path):
        processed_audio_delta, fs = def_wav_read_mfcc(folder)
        audio.append(processed_audio_delta)
    x_data, h, l = matrix_make_up(audio)
    x_data = np.array(x_data)
    x_label = np.array(def_one_hot(labels))
    return x_data, x_label, h, l


def read_wav_matrix2(path):
    map_path, labels = read_wav_path(path)
    audio = []
    for idx, folder in enumerate(map_path):
        processed_audio_delta, fs = def_wav_read_mfcc(folder)
        processed_audio_delta2 = def_wav_read_gfcc(folder, fs)
        audio.append(processed_audio_delta2)
    x_data2, h2, l2 = matrix_make_up(audio)
    x_data2 = np.array(x_data2)
    x_label2 = np.array(def_one_hot(labels))
    return x_data2, x_label2, h2, l2


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def MSC_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5,
                     filters_pool_proj):
    conv_1x1 = tf.layers.conv2d(x, filters_1x1, (1, 1), padding='same', activation=tf.nn.relu)

    conv_3x3_reduce = tf.layers.conv2d(x, filters_3x3_reduce, (1, 1), padding='same', activation=tf.nn.relu)
    conv_3x3 = tf.layers.conv2d(conv_3x3_reduce, filters_3x3, (3, 3), padding='same', activation=tf.nn.relu)

    conv_5x5_reduce = tf.layers.conv2d(x, filters_5x5_reduce, (1, 1), padding='same', activation=tf.nn.relu)
    conv_5x5 = tf.layers.conv2d(conv_5x5_reduce, filters_5x5, (5, 5), padding='same', activation=tf.nn.relu)

    pool_proj = tf.layers.max_pooling2d(x, (3, 3), strides=(1, 1), padding='same')
    pool_proj = tf.layers.conv2d(pool_proj, filters_pool_proj, (1, 1), padding='same', activation=tf.nn.relu)

    output = tf.concat([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

    return output


# 在Z后添加ECA注意力机制
def eca_module(input_tensor, b=1, gamma=2):
    # 获取输入张量的shape
    _, h, w, c = input_tensor.get_shape().as_list()

    # 步骤1：全局平均池化
    y = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)  # shape: [batch, 1, 1, channels]

    # 步骤2：计算自适应核大小k
    k = int(abs((math.log(c, 2) + b) / gamma))
    k = max(k, 1)  # 确保k至少为1

    # 步骤3：一维卷积来捕获跨通道交互
    y = tf.reshape(y, [-1, 1, c, 1])  # reshape以适应1D卷积
    y = tf.pad(y, [[0, 0], [0, 0], [(k - 1) // 2, k // 2], [0, 0]])  # 对称填充
    y = tf.layers.conv2d(y, filters=1, kernel_size=(1, k), padding='valid', use_bias=False)

    # 步骤4：sigmoid激活函数
    y = tf.nn.sigmoid(y)
    y = tf.reshape(y, [-1, 1, 1, c])  # 恢复形状为[batch, 1, 1, channels]

    # 步骤5：通道权重与输入张量相乘
    output = input_tensor * y

    return output


# 实现CCA交叉注意力模块
# 修改后的CCA交叉注意力模块实现
# 简化的CCA模块实现，适用于TensorFlow 1.x和小尺寸特征图
def criss_cross_attention(input_tensor, reduction_ratio=8, name='cca'):
    """
    简化的CCA交叉注意力模块
    """
    # 获取输入形状
    shape = input_tensor.get_shape().as_list()
    _, h, w, c = shape

    # 通道降维
    query = tf.layers.conv2d(input_tensor, c // reduction_ratio, (1, 1), padding='same', name=name + '_query')
    key = tf.layers.conv2d(input_tensor, c // reduction_ratio, (1, 1), padding='same', name=name + '_key')
    value = tf.layers.conv2d(input_tensor, c, (1, 1), padding='same', name=name + '_value')

    # 水平方向注意力
    query_h = tf.reshape(query, [-1, h, w, c // reduction_ratio])
    key_h = tf.reshape(key, [-1, h, w, c // reduction_ratio])
    value_h = tf.reshape(value, [-1, h, w, c])

    # 为每一行计算注意力
    context_h = tf.zeros_like(value_h)
    for i in range(h):
        q_row = query_h[:, i:i + 1, :, :]  # 取第i行
        k_row = key_h[:, i:i + 1, :, :]  # 取第i行
        v_row = value_h[:, i:i + 1, :, :]  # 取第i行

        # 计算行内注意力
        q_row_flat = tf.reshape(q_row, [-1, w, c // reduction_ratio])
        k_row_flat = tf.reshape(k_row, [-1, w, c // reduction_ratio])
        v_row_flat = tf.reshape(v_row, [-1, w, c])

        # 计算行内注意力权重
        energy_h = tf.matmul(q_row_flat, tf.transpose(k_row_flat, [0, 2, 1]))
        attention_h = tf.nn.softmax(energy_h, axis=-1)

        # 应用注意力
        context_row = tf.matmul(attention_h, v_row_flat)
        context_row = tf.reshape(context_row, [-1, 1, w, c])

        # 更新水平上下文
        context_h = context_h + tf.pad(
            context_row,
            [[0, 0], [i, h - i - 1], [0, 0], [0, 0]]
        )

    # 垂直方向注意力
    context_v = tf.zeros_like(value_h)
    for j in range(w):
        q_col = query_h[:, :, j:j + 1, :]  # 取第j列
        k_col = key_h[:, :, j:j + 1, :]  # 取第j列
        v_col = value_h[:, :, j:j + 1, :]  # 取第j列

        # 计算列内注意力
        q_col_flat = tf.reshape(q_col, [-1, h, c // reduction_ratio])
        k_col_flat = tf.reshape(k_col, [-1, h, c // reduction_ratio])
        v_col_flat = tf.reshape(v_col, [-1, h, c])

        # 计算列内注意力权重
        energy_v = tf.matmul(q_col_flat, tf.transpose(k_col_flat, [0, 2, 1]))
        attention_v = tf.nn.softmax(energy_v, axis=-1)

        # 应用注意力
        context_col = tf.matmul(attention_v, v_col_flat)
        context_col = tf.reshape(context_col, [-1, h, 1, c])

        # 更新垂直上下文
        context_v = context_v + tf.pad(
            context_col,
            [[0, 0], [0, 0], [j, w - j - 1], [0, 0]]
        )

    # 合并水平和垂直上下文
    context = context_h + context_v

    # 残差连接
    output = input_tensor + context

    return output


# 递归交叉注意力模块
def rcca_module_efficient(input_tensor, reduction_ratio=8, recurrence=2, name='rcca'):
    x = input_tensor
    for i in range(recurrence):
        x = criss_cross_attention(x, reduction_ratio, name=f'{name}_{i}')
    return x


def xunlianlo(path, batch_size=68):
    x_train, y_train, h, l = read_wav_matrix(path)
    x_test, y_test, h, l = read_wav_matrix(test_path)
    # 额外特征
    x_train2, _, h2, l2 = read_wav_matrix2(path)
    x_test2, _, _, _ = read_wav_matrix2(test_path)

    # 加载验证数据 - 添加这段代码
    x_val, y_val, _, _ = read_wav_matrix(val_path)
    x_val2, _, _, _ = read_wav_matrix2(val_path)

    m, n = y_train.shape
    x = tf.placeholder(tf.float32, [None, h, l], name='x-input')
    x2 = tf.placeholder(tf.float32, [None, h2, l2], name='x2-input')
    y = tf.placeholder(tf.float32, [None, n], name='y-input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # 处理mfcc特征
    x_image = tf.reshape(x, [-1, h, l, 1], name='x_image')
    conv1 = tf.layers.conv2d(x_image, 64, (7, 7), strides=(2, 2), padding='same', activation=tf.nn.relu)

    # 处理gfcc特征
    x2_image = tf.reshape(x2, [-1, h2, l2, 1], name='x2_image')
    conv2 = tf.layers.conv2d(x2_image, 64, (7, 7), strides=(2, 2), padding='same', activation=tf.nn.relu)

    # SCDFF 特征融合
    alpha = conv1
    beta = conv2

    gamma = alpha + beta

    r = 2  # 平均池化的步长和池化大小
    K1 = weight_variable([1, 1, 64, 64], name='K1')
    K2 = weight_variable([3, 3, 64, 64], name='K2')
    K3 = weight_variable([3, 3, 64, 64], name='K3')

    T1 = gamma
    T2 = tf.nn.relu(tf.layers.batch_normalization(
        tf.nn.conv2d(tf.nn.avg_pool(gamma, ksize=[1, r, r, 1], strides=[1, r, r, 1], padding='SAME'),
                     K1, strides=[1, 1, 1, 1], padding='SAME')))
    T3 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(gamma, K2, strides=[1, 1, 1, 1], padding='SAME')))

    T2_resized = tf.image.resize(T2, tf.shape(T3)[1:3], method=tf.image.ResizeMethod.BILINEAR)
    V = T3 * tf.nn.sigmoid(T1 + T2_resized)
    U = tf.nn.sigmoid(tf.layers.batch_normalization(tf.nn.conv2d(V, K3, strides=[1, 1, 1, 1], padding='SAME')))

    ac = U
    bc = 1 - U

    Z = ac * alpha + bc * beta

    # 应用ECA注意力机制到Z
    Z_with_eca = eca_module(Z)

    pool1 = tf.layers.max_pooling2d(Z_with_eca, (3, 3), strides=(2, 2), padding='same')

    conv2 = tf.layers.conv2d(pool1, 64, (1, 1), padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, 192, (3, 3), padding='same', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, (3, 3), strides=(2, 2), padding='same')

    # MSC模块 + CCA
    MSC3a = MSC_module(pool3, 64, 96, 128, 16, 32, 32)
    MSC3a = rcca_module_efficient(MSC3a, reduction_ratio=8, recurrence=2, name='rcca_3a')

    MSC3b = MSC_module(MSC3a, 128, 128, 192, 32, 96, 64)
    MSC3b = rcca_module_efficient(MSC3b, reduction_ratio=8, recurrence=2, name='rcca_3b')

    pool4 = tf.layers.max_pooling2d(MSC3b, (3, 3), strides=(2, 2), padding='same')

    MSC4a = MSC_module(pool4, 192, 96, 208, 16, 48, 64)
    MSC4a = rcca_module_efficient(MSC4a, reduction_ratio=8, recurrence=2, name='rcca_4a')

    MSC4b = MSC_module(MSC4a, 160, 112, 224, 24, 64, 64)
    MSC4b = rcca_module_efficient(MSC4b, reduction_ratio=8, recurrence=2, name='rcca_4b')

    MSC4c = MSC_module(MSC4b, 128, 128, 256, 24, 64, 64)
    MSC4c = rcca_module_efficient(MSC4c, reduction_ratio=8, recurrence=2, name='rcca_4c')

    MSC4d = MSC_module(MSC4c, 112, 144, 288, 32, 64, 64)
    MSC4d = rcca_module_efficient(MSC4d, reduction_ratio=8, recurrence=2, name='rcca_4d')

    MSC4e = MSC_module(MSC4d, 256, 160, 320, 32, 128, 128)
    MSC4e = rcca_module_efficient(MSC4e, reduction_ratio=8, recurrence=2, name='rcca_4e')

    MSC5a = MSC_module(MSC4e, 256, 160, 320, 32, 128, 128)
    MSC5a = rcca_module_efficient(MSC5a, reduction_ratio=8, recurrence=2, name='rcca_5a')

    MSC5b = MSC_module(MSC5a, 384, 192, 384, 48, 128, 128)
    MSC5b = rcca_module_efficient(MSC5b, reduction_ratio=8, recurrence=2, name='rcca_5b')

    pool6 = tf.reduce_mean(MSC5b, axis=[1, 2])
    dropout = tf.nn.dropout(pool6, keep_prob, name='dropout')

    prediction = tf.layers.dense(dropout, n)
    tf.add_to_collection('predictions', prediction)
    p = tf.nn.softmax(prediction, name='prediction')
    tf.add_to_collection('p', p)
    tf.add_to_collection('predictions', prediction)

    # 损失函数和优化器
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')

    train_acc_list = []
    loss_list = []
    test_acc_list = []

    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    total_epochs = 120# 设置epoch总数为100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 假设每个epoch包含的迭代次数是总数据量除以批量大小
        iterations_per_epoch = len(x_train) // batch_size
        for epoch in range(total_epochs):
            for i in range(iterations_per_epoch):
                batch_x, batch_x2, batch_y = next_batch(batch_size, x_train, x_train2, y_train)
                sess.run(train_step, feed_dict={x: batch_x, x2: batch_x2, y: batch_y, keep_prob: 1.0})
            # 在每个epoch结束后计算整个训练集的准确率
            train_acc = sess.run(accuracy, feed_dict={x: x_train, x2: x_train2, y: y_train, keep_prob: 1.0})
            val_loss = sess.run(cross_entropy, feed_dict={x: x_val, x2: x_val2, y: y_val, keep_prob: 1.0})
            # 修改这一行以包含x_val2
            val_acc = sess.run(accuracy, feed_dict={x: x_val, x2: x_val2, y: y_val, keep_prob: 1.0})
            print("Epoch " + str(epoch) + " completed out of " + str(total_epochs) + ", loss: " + str(val_loss
              ) + ", training accuracy: " + str(train_acc) + ", validation accuracy: " + str(val_acc))
        saver.save(sess, 'nn.ECA-NET.100m/my_net.ckpt')
        plt.figure(1)
        plt.plot(train_acc_list, label='Train Accuracy')
        plt.plot(test_acc_list, label='Test Accuracy')
        plt.xlabel('iters')
        plt.ylabel('accuracy')
        plt.title('Train and Test Accuracy Curves')
        plt.figure(2)
        plt.plot(loss_list)
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.title('Loss curve')
        plt.show()
        coord.request_stop()
        coord.join(threads)


def test_main(test_path):
    x_test, y_test, h, l = read_wav_matrix(test_path)
    x_test2, _, _, _ = read_wav_matrix2(test_path)
    m, n = y_test.shape
    labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("nn.ECA-NET.100m/my_net.ckpt.meta")
        saver.restore(sess, 'nn.ECA-NET.100m/my_net.ckpt')
        predictions = tf.get_collection('predictions')[0]
        p = tf.get_collection('p')[0]
        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name('x-input').outputs[0]
        input_x2 = graph.get_operation_by_name('x2-input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        pred_y = []
        for i in range(m):
            feed_dict = {input_x: np.array([x_test[i]]), input_x2: np.array([x_test2[i]]), keep_prob: 1.0}
            result = sess.run(predictions, feed_dict=feed_dict)
            haha = sess.run(p, feed_dict=feed_dict)
            pred_y.append(np.argmax(result))
            print("实际 :" + str(np.argmax(y_test[i])) + " ,预测: " + str(np.argmax(result)) + " ,预测可靠度: " + str(
                np.max(haha)))
        cm = confusion_matrix(np.argmax(y_test, 1), pred_y, labels=[0, 1, 2, 3])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()


if __name__ == '__main__':
    xunlianlo(path)
    #test_main(test_path)##测试的方法