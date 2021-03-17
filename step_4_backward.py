import tensorflow.contrib.slim as slim
import tensorflow as tf
from step_2_load_data import to_matrix
from cnn_config import *
import logging
from step_3_forward import forward

logger = logging.getLogger(__name__)


def backward(b):
    # with tf.variable_scope("only_1", reuse=tf.AUTO_REUSE):
    vec_1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, ROW_NUM, COL_NUM, 1), name="placeholder1")
    logger.info("vec_1 placeholder successfully")

    vec_2 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, ROW_NUM, COL_NUM, 1), name="placeholder2")
    logger.info("vec_2 placeholder successfully")

    label = tf.placeholder(tf.float32)
    logger.info("label placeholder successfully")

    y = forward(vec_1, vec_2)
    logger.info("forward success")

    # global_step = tf.Variable(0, trainable=False)
    # loss = tf.maximum(0.0,tf.add(tf.subtract(y, labels[i]),0.1))
    # loss = tf.reduce_mean(tf.square(tf.subtract(y, label)))
    # loss = loss + tf.add_n(tf.add_to_collection("losses",value=0.1))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

    # 指数衰减学习率
    # learning_rate = tf.train.exponential_decay(
    #     LEARNING_RATE_BASE,
    #     global_step,
    #     SAMPLE_NUM / BATCH_SIZE,
    #     LEARNING_RATE_DECAY,
    #     staircase=True
    # )

    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        logger.info("init")
        res_list = []
        vec_1_list = to_matrix()
        for i in range(SAMPLE_NUM):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            res_list.append(sess.run(y, feed_dict={vec_1: vec_1_list[start:end],
                                                   vec_2: [b]}))
            # label: labels[start:end]})
            # if i % 20 == 0:
            #     loss_v = sess.run(loss, feed_dict={vec_1: vec_1_list[start:end],
            #                                        vec_2: vec_2_list[start:end],
            #                                        label: labels[start:end]})
            #     print("第%s次，loss = %s" % (i, loss_v))
        # saver.save(sess, "./mpcnn_model", global_step=global_step)
        return res_list


if __name__ == '__main__':
    backward()
