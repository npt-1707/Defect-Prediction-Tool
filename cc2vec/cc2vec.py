from flask import Flask, request
from model import HierachicalRNN, DeepJITExtended
import torch

app = Flask(__name__)

@app.route('/api/cc2vec', methods=['POST'])
def template():
    # Get request_data from main.py
    '''
        {
            "id": string,
            "parameters": dictionary,
            "input": input_type
        }
    '''
    request_data = request.get_json()
    if app.debug:
        print(request_data)

    # Split input
    code_loader, dict_msg, dict_code = request_data["input"]

    # Load parameters
    params = request_data["parameters"]

    # Set up param
    params["filter_sizes"] = [int(k) for k in params["filter_sizes"].split(',')]
    params["vocab_msg"], params["vocab_code"] = len(dict_msg), len(dict_code)
    params["cc2vec_class_num"] = len(code_loader["message"])
    params["deepjit_class_num"] = 1

    # Create model and Load pretrain
    cc2vec = HierachicalRNN(params).to(device=params["device"])
    cc2vec.load_state_dict(torch.load(params["pretrained_cc2vec"]))
    deepjit = DeepJITExtended(params).to(device=params["device"])
    deepjit.load_state_dict(torch.load(params["pretrained_deepjit"]))

    # Forward
    cc2vec.eval()
    deepjit.eval()
    with torch.no_grad():
        # Extract data from DataLoader
        added_code = torch.tensor(code_loader["added_code"], device=params["device"])
        removed_code = torch.tensor(code_loader["removed_code"], device=params["device"])
        code = torch.tensor(code_loader["code"], device=params["device"])
        message = torch.tensor(code_loader["message"], device=params["device"])

        # Forward
        feature = cc2vec(added_code, removed_code)
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
    app.run(debug=True, host="0.0.0.0", port=5001)