from flask import Flask, render_template, request
from exts import db
import config
import qa
from flask import jsonify

app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)


# 下面写自己的业务
@app.route('/', methods=['GET', 'POST'])
def first():
    if request.method == 'POST':
        question = request.get_json()
        if question:
            result = qa.getAnswer(question)
            if result != 'noanswer':
                return jsonify({"answer": result})
            else:
                return jsonify({"answer": '问题太高深,我会努力学习的'})
    else:
        return render_template('first.html')


@app.route('/second/',methods=['GET', 'POST'])
def second():
    if request.method == 'POST':
        question = request.get_json()
        if question:
            result = qa.getAnswer2(question)
            if result != 'noanswer':
                return jsonify({"answer": result})
            else:
                return jsonify({"answer": '问题太高深,我会努力学习的'})
    else:
        return render_template('second.html')


@app.route('/third/')
def third():
    # 这里写第三个方法计算的返回值，从前端获取用户输入的内容，并返回相应的结果
    return render_template('third.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=13900, debug=True)
