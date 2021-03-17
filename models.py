from exts import db


# 创建一个答案模型，字段与数据库对应
class Answer(db.Model):
    __tablename__ = 'q_a'
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    finalurl = db.Column(db.Text)


# 创建一个问题关键词的模型，字段与数据库对应
class Question(db.Model):
    __tablename__ = 'q_keyword'
    aid = db.Column(db.Integer, db.ForeignKey('q_a.id'), primary_key=True, )
    keyword = db.Column(db.String(100), primary_key=True)

    author = db.relationship("Answer", backref='q_keyword')
