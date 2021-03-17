import tensorflow as tf
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99

ROW_NUM = 10
COL_NUM = 100
SAMPLE_NUM = 445
STEPS = 1000
BATCH_SIZE = 1  #
BIAS_VALUE = 0.1  # 初始化偏执项的值
NUM_CHANNELS = 1  # 通道数

CONV1D_KERNEL_SIZE = 3  # 一个卷积核尺寸
CONV1D_KERNEL_NUM = 3  # 第一个卷积核个数

CONV2D_KERNEL_SIZE = 3  # 第二个卷积核尺寸
CONV2D_KERNEL_NUM = 2  # 第二个卷积核个数

POOL_KERNEL_SIZE = 2
POOL_STRIDES = 2

FC_SIZE = 512  # 全连接层尺寸
OUTPUT_NODE = 1  # 输出结点个数

LEARNING_RATE = 0.01  # 学习率
REGULARIZER = 0.0001  # 正则化参数
SENTENCE_SIZE = 0  # 输入句子的有效词长度
FILTER_SIZE = [1, 2, 100]
FILTER_NUM = [[3], [2]]


def min_pool(input, ksize, strides, padding):
    res = - (tf.nn.max_pool(-input, ksize=ksize, strides=strides, padding="VALID"))
    return res


POOLINGS = [
    tf.nn.max_pool,
    tf.nn.avg_pool,
    min_pool]  # 池化方式
# WS宽度


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
