# app.py (Main Entry Point)
from flask import Flask
from api.routes import bp as api_bp

app = Flask(__name__)
app.register_blueprint(api_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
