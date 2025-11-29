from flask import Flask
from api.routes import bp as train_bp

app = Flask(__name__)
app.register_blueprint(train_bp)

@app.route("/health")
def health():
    from datetime import datetime
    return {"status": "ok", "time": datetime.now().strftime("%Y-%m-%d %H:%M")}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
