from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('ridge_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return 'Welcome to the API!'

@app.route('/run-model', methods=['POST'])
def run_model():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        predictions = model.predict(input_df)
        response = {'predictions': predictions.tolist()}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
