from flask_script import Manager
from restful import app
from flask_migrate import MigrateCommand, Migrate
from exts import db

# 下面是引用models的模型
from models import Answer, Question

# 第一步，绑定manager与app
manager = Manager(app)

# 第二步，把app和db绑定到Migrate
Migrate(app, db)

# 第三步，把MigrateCommand添加到manager
manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
