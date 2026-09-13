from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/predict/fused', methods=['POST'])
def predict_fused():
    # Placeholder for fusion API
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
