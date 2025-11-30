from flask import Flask
from app.api.routes import bp as api_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app
