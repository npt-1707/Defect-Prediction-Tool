from flask import Flask, request
import pandas as pd
import os, json
from .model import JITLine

app = Flask(__name__)

@app.route('/api/jitline', methods=['POST'])
def template():
    # Get request_data from main.py
    '''
        {
            "id": string,
            "input": 14 features
        }
    '''
    request_data = request.get_json()
    if app.debug:
        print("Request: ", json.dumps(request_data, indent=4))
    columns = ["_id", "ns", "nd", "nf", "entrophy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    input = request_data["input"]
    features = input["feature"]
    features = {key:[features[key]] for key in columns}
    features = pd.DataFrame(features)
    code_change = input["commit"]
    
    if app.debug:
        print(json.dumps(code_change, indent=4))

    # get model
    model_path = os.path.join(os.getcwd(), "pretrained_models")
    files = os.listdir(model_path)
    predict = []
    for file in files:
        model = JITLine(load_path=os.path.join(model_path, file))
        predict.append(model.predict_proba(features, code_change)[:, 1][0])
    
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
    app.run(debug=True, host="0.0.0.0", port=5006)