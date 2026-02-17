from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def handle_root():
    return "Welcome to API"

@app.route("/health")
def handle_health():
    return "ok"

@app.route("/echo", methods=["POST"])
def handle_echo():
    return request.data, 200

@app.errorhandler(404)
def handle_404(e):
    return "not found", 404

app.run(port=8080)
