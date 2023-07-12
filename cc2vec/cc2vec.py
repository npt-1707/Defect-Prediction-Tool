from flask import Flask, request
from model import HierachicalRNN, DeepJITExtended
import torch
import numpy as np

app = Flask(__name__)

@app.route('/api/cc2vec', methods=['POST'])
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

    # Split input
    code_loader, dict_msg, dict_code = request_data["input"]

    # Load parameters
    params = request_data["parameters"]

    # Set up param
    params["filter_sizes"] = [int(k) for k in params["filter_sizes"].split(',')]
    params["vocab_msg"], params["vocab_code"] = len(dict_msg), len(dict_code)
    params["cc2vec_class_num"] = len(dict_msg)
    params["deepjit_class_num"] = 1

    cc2vec = HierachicalRNN(params).to(device=request_data["device"])
    cc2vec.load_state_dict(torch.load(params["pretrained_cc2vec"], map_location=request_data["device"]))
    cc2vec.eval()
    with torch.no_grad():
        # Extract data from DataLoader
        added_code = np.array(code_loader["added_code"])
        removed_code = np.array(code_loader["removed_code"])

        # Forward
        state_word = cc2vec.init_hidden_word(request_data['device'])
        state_sent = cc2vec.init_hidden_sent(request_data['device'])
        state_hunk = cc2vec.init_hidden_hunk(request_data['device'])
        feature = cc2vec(added_code, removed_code, state_hunk, state_sent, state_word, request_data['device'])

    params["embedding_feature"] = feature.shape[1]

    deepjit = DeepJITExtended(params).to(device=request_data["device"])
    deepjit.load_state_dict(torch.load(params["pretrained_deepjit"], map_location=request_data["device"]))
    deepjit.eval()
    with torch.no_grad():
        code = torch.tensor(code_loader["code"], device=request_data["device"])
        message = torch.tensor(code_loader["message"], device=request_data["device"])

        predict = deepjit.forward(feature, message, code)

    # Create response like form below:
    '''
        {
            "id": string,
            "output": float
        }
    '''
    return {
        'id': request_data["id"],
        'output': predict.item()
    }

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5002)