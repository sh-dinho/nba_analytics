from flask import Flask
from api.routes import bp as train_bp

app = Flask(__name__)
app.register_blueprint(train_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
