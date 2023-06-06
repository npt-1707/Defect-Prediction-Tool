from flask import Flask, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

@app.route('/api/lapredict', methods=['POST'])
def template():
    # Get request_data from main.py
    '''
        {
            "id": string,
            "input": 14 features
        }
    '''
    request_data = request.get_json()

    # get input
    input = {"la":[request_data["input"]["la"]]}
    input = pd.DataFrame(input)

    # get model
    model_path = os.path.join(os.getcwd(), "model")
    files = os.listdir(model_path)
    predict = []
    for file in files:
        with open(os.path.join(model_path, file), "rb") as f:
            model = pickle.load(f) 
        predict.append(model.predict_proba(input)[:, 1][0])
    
    if app.debug:
        print(input)
        print(predict)
    # Create response like form below:
    '''
        {
            "id": string,
            "output": float
        }
    '''
    return {
        'id': request_data["id"],
        'output': 1/6 * sum(predict)
    }

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8001)