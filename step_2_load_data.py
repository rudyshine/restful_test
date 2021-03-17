import pandas as pd
import numpy as np
import logging
from jieba import cut
from cnn_config import *
from gensim.models import word2vec

logger = logging.getLogger(__name__)


def to_matrix():
    data = pd.read_csv("D:/Projects/Python/restful_test/cut", header=None, sep="?")
    # data = np.fromfile("D:/Projects/Python/restful/cut")
    s1_array = []
    model = word2vec.Word2Vec.load("D:/Projects/Python/restful_test/static/word_to_vec.model")
    vocabulary_dict = model.wv.vocab.keys()
    # with open("D:/Projects/Python/restful/cut", "r+", encoding="utf-8") as f:
    #     for line in f:
    #         s1 = str_to_vector(line, model, vocabulary_dict)
    # sentence_1 = data
    sentence_1 = data.iloc[:, 0]
    # sentence_2 = data.iloc[:, 1]
    # similarity = data.iloc[:, 2]

    # s1_array = []
    # s2_array = []
    # s3_array = []

    range_num = len(sentence_1)
    for i in range(range_num):
        #     for line in f:
        s1 = str_to_vector(sentence_1[i], model, vocabulary_dict)
        # s2 = str_to_vector(sentence_2[i], model, vocabulary_dict)
        # s3 = np.array(similarity[i]).reshape(1)
        s1_array.append(s1)
    # s2_array.append(s2)
    # s3_array.append(s3)
    # if i % 500 == 0:
    #     logger.info("500 passed")
    # return s1_array, s2_array, s3_array

    return s1_array


# 将句子转化为word2vec向量
def str_to_vector(a_str, model, vocabulary_dict):
    words = a_str.split(" ")
    matrix_1 = np.array([np.array(model[word]) for word in words if word in vocabulary_dict], dtype=np.float32)
    matrix_1_len = len(matrix_1)
    if matrix_1_len == 10:
        pass
    elif matrix_1_len < 10:
        matrix_1 = padding(matrix_1, matrix_1_len, 10)
    else:
        matrix_1 = matrix_1[:10]
    return matrix_1.reshape(10, 100, 1)


# row小于10填充到10
def padding(matrix, matrix_len, tar_len):
    diff = tar_len - matrix_len
    add_matrix = np.zeros([diff, 100])
    matrix = matrix.reshape(-1, 100)
    res = np.row_stack((matrix, add_matrix))
    return res


if __name__ == '__main__':
    # train_model()
    model = word2vec.Word2Vec.load("word_to_vec.model")
    # vocabulary = model.wv.vocab.keys()
    # logger.info(vocabulary)
    logger.info(model["送给"])
    to_matrix()
