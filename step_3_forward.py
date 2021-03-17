import tensorflow.contrib.slim as slim
import tensorflow as tf
from step_2_load_data import to_matrix
from cnn_config import *
import logging
import numpy as np

logger = logging.getLogger(__name__)


# 初始化权重
def init_weight(shape, name):
    weight = tf.Variable(tf.truncated_normal(shape=shape,
                                             mean=0,
                                             stddev=1.0,
                                             dtype=tf.float32), name=name)
    # tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(REGULARIZER)(weight))
    return weight


# 初始化偏置项
def init_bias(bias_value, shape, name):
    bias = tf.constant(bias_value, shape=shape, name=name, dtype=tf.float32)
    return bias


def conv2d(vec, weight):
    return tf.nn.conv2d(vec,
                        weight,
                        strides=[1, 1, 1, 1],
                        padding='VALID')


def conv_b(x, weight, pooling, bias):
    '''
        由于要按每一维度卷积，所以将  向量 和权重 拆分成 一维
    '''
    # 降维把【1，10，100，1】的矩阵拆分成20个【1，100，1】的矩阵，
    input_unstack = tf.unstack(tf.unstack(x, axis=0), axis=1)

    # weigt是三维的，直接拆分
    w_unstack = tf.unstack(weight, axis=0)
    # b_unstack = tf.unstack(bias, axis=1)
    convs = []
    # 按行卷积池化
    for i in range(ROW_NUM):
        conv = tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID")
        conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=True)
        convs.append(conv)
    # 拼接，将结果拼接起来
    conv = tf.stack(convs, axis=1)
    # 再进行池化
    pool = pooling(conv,
                   ksize=[1, POOL_KERNEL_SIZE, POOL_KERNEL_SIZE, 1],
                   strides=[1, POOL_STRIDES, POOL_STRIDES, 1],
                   padding="VALID")
    return pool


def euclidean_distance(vec_1, vec_2):
    dis = tf.sqrt(tf.reduce_sum(tf.square(vec_1 - vec_2), 2))
    return dis


def tf_cos(vec_1, vec_2):
    # 求模
    vec_1_norm = tf.sqrt(tf.reduce_sum(tf.square(vec_1)))
    vec_2_norm = tf.sqrt(tf.reduce_sum(tf.square(vec_2)))
    # 内积
    v1_x_v2 = tf.reduce_sum(tf.multiply(vec_1, vec_2))
    # cosin = v1_x_v2 / (vec_1_m * vec_2_m)
    cos = tf.divide(v1_x_v2, tf.multiply(vec_1_norm, vec_2_norm))
    return cos


# 随便算了一下
def similarity_sentence_layer(a_1_res, b_1_res, a_2_res, b_2_res, full_c):
    # 计算 相同计算过程的 向量 cos
    result = []
    for i in range(3):
        for k in range(3):
            transform_1 = a_1_res[i][k]
            transform_2 = a_2_res[i][k]
            d = tf_cos(transform_1, transform_2)
            result.append(d)

    for i in range(2):
        for k in range(2):
            transform_1 = b_1_res[i][k]
            transform_2 = b_2_res[i][k]
            d = tf_cos(transform_1, transform_2)
            result.append(d)

    res = tf.reduce_mean(tf.multiply(result, full_c))
    logger.warning(res)
    return res


def block_a(weight, x, bias):
    # 输入文本转化为四维矩阵[batch_size, height, width, channels]
    out = []
    # 遍历池化方式
    for pooling in POOLINGS:
        pools = []  # 收集池化结果的列表
        # 有三种卷积核
        for i, ws in enumerate(FILTER_SIZE):
            logger.info(ws)
            logger.info(weight[i].shape)
            conv = tf.nn.tanh(conv2d(x, weight[i]) + BIAS_VALUE)
            logger.info(conv)
            # conv = slim.batch_norm(inputs=conv, activation_fn=tf.nn.tanh, is_training=True)
            # conv = tf.nn.relu(conv)
            if ws == 100:
                # pool = pooling(conv,
                #                ksize=[1, 1, 2, 1],
                #                strides=[1, 1, 2, 1],
                #                padding="VALID")
                pool = conv
            else:
                pool = pooling(conv,
                               ksize=[1, 1, POOL_KERNEL_SIZE, 1],
                               strides=[1, 1, POOL_STRIDES, 1],
                               padding="VALID")
            pools.append(pool)
        out.append(pools)

    return out


def block_b(weight, x, bias):
    out = []
    # 两种池化
    for pooling in POOLINGS[:-1]:
        pools = []
        # 两种卷积核不含最后一项 ∞
        for i, ws in enumerate(FILTER_SIZE[:-1]):
            # 调用多粒度卷积
            pool = conv_b(x, weight[i], pooling, bias)
            pools.append(pool)
        out.append(pools)
    return out


# 一次完整的计算过程产出res_a,res_b两个结果
def start(x, a_weight, b_weight, a_bias, b_bias):
    # block_a的权重初始化# block_a 采用conv2d 卷积核需要四个参数[卷积核高度，卷积核宽度， 图像通道数， 卷积核个数]
    input_x = tf.reshape(x, shape=(1, ROW_NUM, COL_NUM, 1))
    logger.info("reshape successfully")
    # input_x = np.array(x, dtype=np.float32).reshape(BATCH_SIZE, ROW_NUM, COL_NUM, 1)
    # 因为需要[ws = 1,2,∞] 所以循环生成对应的权重组,bolck_a 需要三种池化方式所以最后的卷积核数目定位3
    out_res_a = block_a(a_weight, input_x, a_bias)
    logger.info("out_res_a")
    out_res_b = block_b(b_weight, input_x, b_bias)
    logger.info("out_res_b")
    return out_res_a, out_res_b


def forward(vec_1, vec_2):
    # 遍历列表拿到各组数组，取其形状
    logger.info("vec.shape = (%s,%s)", ROW_NUM, COL_NUM)
    # 按照形状的不同，使用不同的权重
    with tf.variable_scope("only", reuse=tf.AUTO_REUSE):
        a_weight = [init_weight([ROW_NUM, FILTER_SIZE[i], 1, FILTER_NUM[0][0]], name="w1_%s" % (i)) for i in
                    range(3)]
        logger.info(a_weight)
        a_bias = init_bias(BIAS_VALUE, shape=FILTER_NUM[0], name="b1_0")

        b_weight = [init_weight([ROW_NUM, FILTER_SIZE[i], 1, FILTER_NUM[1][0]], name="W2_%s" % (i)) for i in
                    range(2)]
        logger.info(b_weight)
        b_bias = init_bias(BIAS_VALUE, shape=[FILTER_NUM[1][0], ROW_NUM], name="b2_0")

        full_connect = init_weight((13,), "full_c")

    a_1_res, b_1_res = start(vec_1, a_weight, b_weight, a_bias, b_bias)
    a_2_res, b_2_res = start(vec_2, a_weight, b_weight, a_bias, b_bias)

    final_res = similarity_sentence_layer(a_1_res,
                                          b_1_res,
                                          a_2_res,
                                          b_2_res,
                                          full_connect)
    return final_res
