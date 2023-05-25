from flask import Flask, request
from model import DeepJIT
import json, torch

app = Flask(__name__)

@app.route('/api/model_template', methods=['POST'])
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
    params["class_num"] = 1

    # Create model and Load pretrain
    model = DeepJIT(params).to(device=params["device"])
    model.load_state_dict(torch.load(params["pretrained_model"]))

    # Forward
    model.eval()
    with torch.no_grad():
        # Extract data from DataLoader
        code = torch.tensor(code_loader["code"], device=params["device"])
        message = torch.tensor(code_loader["message"], device=params["device"])
        print(code.size())
        print(message.size())

        # Forward
        predict = model(message, code)

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