from __future__ import print_function
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import gensim
import pandas as pd

MAX_SEQUENCE_LENGTH = 1000 # 每篇文章选取1000个词
MAX_NB_WORDS = 10000 # 将字典设置为含有1万个词
EMBEDDING_DIM = 300 # 词向量维度，300维
VALIDATION_SPLIT = 0.2 # 测试集大小，全部数据的20%

# 目的是得到一份字典(embeddings_index)含有1万个词，每个词对应属于自己的300维向量
embeddings_index = {}

print('Indexing word vectors.')
path = '../word2vec_model'
model = gensim.models.Word2Vec.load(path)
word_vectors = model.wv
for word, vocab_obj in model.wv.vocab.items():
    if int(vocab_obj.index) < MAX_NB_WORDS:
        embeddings_index[word] = word_vectors[word]
del model, word_vectors # 删掉gensim模型释放内存
print('Found %s word vectors.' % len(embeddings_index))

# print out:
# Indexing word vectors.
# Found 10000 word vectors.


