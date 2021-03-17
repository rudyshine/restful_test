import pymysql
import jieba
from gensim.models import word2vec
from sqlalchemy import create_engine
from config import DB_URI
from models import Question, Answer
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker
from step_2_load_data import str_to_vector
from step_4_backward import backward
import numpy as np

# 第一种方法，使用ORM来实现

# 第一步：创建一个session，供后面查询使用
engine = create_engine(DB_URI)
session = sessionmaker(engine)()


# 第二步：定义一个切词，并组合成列表，做为后面的sql的in查询条件
def segmentWord(cont):
    c = []
    cut_word = jieba.cut(cont)
    for j in cut_word:
        c.append(j)
    return c


def getAnswer(question):
    b = segmentWord(question)
    # 查询包含提问关键词的原始问题关键词，原始问题与新提问关键词按照匹配的数量倒序排序取第一个
    result = session.query(func.count(Question.aid).label('number'), Question.aid).filter(
        Question.keyword.in_(b)).group_by(Question.aid).order_by(func.count(Question.aid).desc()).first()
    if result:
        # 如果有能匹配上的，就返回这个匹配上的答案
        answer = Answer.query.filter(Answer.id == result[1]).first()
        return answer.answer
    else:
        return 'noanswer'


# 第三步：第二步的条件是这一步的查询条件
def getAnswer2(question):
    model = word2vec.Word2Vec.load("D:/Projects/Python/restful_test/static/word_to_vec.model")
    vocabulary_dict = model.wv.vocab.keys()
    b = " ".join(segmentWord(question))
    b = str_to_vector(b, model, vocabulary_dict)

    res_list = backward(b)
    index = int(np.array(res_list).argmax() + 1)
    print(index)
    # 查询包含提问关键词的原始问题关键词，原始问题与新提问关键词按照匹配的数量倒序排序取第一个
    # result = session.query(func.count(Question.aid).label('number'), Question.aid).filter(
    #     Question.keyword.in_(b)).group_by(Question.aid).order_by(func.count(Question.aid).desc()).first()
    # result = Answer.query
    # result = session.query(Answer.answer).filter(Answer.id == index)

    result = session.query(Answer.answer).filter(Answer.id == index).first()[0]
    if result:
        # 如果有能匹配上的，就返回这个匹配上的答案
        # answer = Answer.query.filter(Answer.id == result[1]).first()
        return result
    else:
        return 'noanswer'

# 第二种方法：sql直接查询的方式
# def getAnswer(question):
#     connection = pymysql.connect(host='127.0.0.1',
#                                  port=3306,
#                                  user='root',
#                                  password='',
#                                  database='spider',
#                                  charset='utf8')
#     cursor = connection.cursor()
#     question_cut = jieba.cut(question)
#
#     # 拼接成原生sql的in的格式
#     b = ",".join(["'%s'" % x for x in question_cut])
#
#     # 查询匹配到提问关键词最多的问题编号，并根据这个编号查询答案
#     select_str = 'select answer from q_a where id= (select t.aid from (select count(aid) a
#     s number, aid from q_keyword where keyword in (%s) group by aid ORDER BY number desc limit 1) t)' % b
#
#     # 查询结果有多少行，也就是判断有没有能给出答案的问题
#     row_count = cursor.execute(select_str)
#     connection.commit()
#     connection.close()
#     if row_count > 0:
#         results = cursor.fetchone()[0]
#         return results
#     else:
#         return 'noanswer'
