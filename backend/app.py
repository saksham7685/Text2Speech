from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import predict_params, detect_language
from flask import send_from_directory

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route("/")
def index():
    # Serve the original UI unchanged
    return render_template('index.html')

@app.route('/style.css')
def style_css():
    # Keep href="style.css" working without changing HTML
    return send_from_directory('templates', 'style.css')

@app.route("/nn_params", methods=["POST"])
def nn_params():
    data = request.get_json() or {}
    text = data.get("text", "")
    pitch, rate = predict_params(text)
    return jsonify({"pitch": pitch, "rate": rate})

@app.route("/nn_language", methods=["POST"])
def nn_language():
    data = request.get_json() or {}
    text = data.get("text", "")
    lang, scores = detect_language(text)
    return jsonify({"language": lang, "scores": scores})

if __name__ == "__main__":
    app.run(debug=True)
