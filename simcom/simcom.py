from flask import Flask, request
from model import DeepJIT
import json, torch, os, pickle, numpy as np, pandas as pd

app = Flask(__name__)

@app.route('/api/simcom', methods=['POST'])
def template():
    # Get request_data from main.py
    '''
        {
            "id": string,
            "parameters": dictionary,
            "input": input_type,
            "device": string
        }
    '''
    request_data = request.get_json()
    # if app.debug:
    #     print(request_data["parameters"])

    #----------- Sim ----------------------
    features = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    input = request_data["input"]["feature"]
    input = {key:[input[key]] for key in features}
    input = pd.DataFrame(input)
    sim_models_path = os.path.join(os.getcwd(), "sim_model")
    files = os.listdir(sim_models_path)
    sim_predict = []
    for file in files:
        with open(os.path.join(sim_models_path, file), "rb") as f:
            model = pickle.load(f)
        sim_predict.append(model.predict_proba(input)[:, 1][0])
    sim_predict = np.mean(sim_predict)
    #--------------------------------------------

    #----------- Com ----------------------
    # Split input
    code_loader, dict_msg, dict_code = request_data["input"]["commit"]

    # Load parameters
    params = request_data["parameters"]

    # Set up param
    params["filter_sizes"] = [int(k) for k in params["filter_sizes"].split(',')]
    params["vocab_msg"], params["vocab_code"] = len(dict_msg), len(dict_code)
    params["class_num"] = 1

    # Create model and Load pretrain
    model = DeepJIT(params).to(device=request_data["device"])
    model.load_state_dict(torch.load(params["pretrained_model"]))

    # Forward
    model.eval()
    with torch.no_grad():
        # Extract data from DataLoader
        code = torch.tensor(code_loader["code"], device=request_data["device"])
        message = torch.tensor(code_loader["message"], device=request_data["device"])

        # Forward
        com_predict = model(message, code)
        com_predict = com_predict.item()

    #---------------------------------

    # Combine predict
    predict = (com_predict + sim_predict) / 2

    # Create response like form below:
    '''
        {
            "id": string,
            "output": float
        }
    '''
    return {
        'id': request_data["id"],
        'output': predict
    }

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5004)