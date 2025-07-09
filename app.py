from flask import Flask
from flask_migrate import  Migrate
from config import Config
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
migrate = Migrate(app,db)

from routes import routes
app.register_blueprint(routes)

from routes import update_missing_keywords

with app.app_context():
    try:
        update_missing_keywords()
    except Exception as e:
        print(f"Error during update_missing_keywords: {e}")



if __name__ == '__main__':
    app.run(debug=True)
