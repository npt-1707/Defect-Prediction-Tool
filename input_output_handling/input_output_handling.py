from flask import Flask, request
import requests, json
from utils import extract_info_from_commit_link
from preprocess.deepjit.preprocess import deepjit_preprocess
from preprocess.cc2vec.preprocess import cc2vec_preprocess

# Dictionary mapping model_name and model_preprocessing
preprocess_data = {
    'deepjit': deepjit_preprocess,
    'cc2vec': cc2vec_preprocess
}

api_lists = {
    'deepjit': "http://localhost:5001/api/deepjit",
    'cc2vec': "http://localhost:5002/api/cc2vec",
    'lapredict': "http://localhost:8001/api/lapredict"
}

app = Flask(__name__)

@app.route('/api/input_output', methods=['POST'])
def template():
    # Get request_data from main.py 
    request_data = request.get_json()
    # if app.debug:
    #     print(request_data)

    #----- Handling request -------------------#
    '''
        request {
            "id": string,
            "commit_info": dict,
            "features": .csv file,
            "link_commit": string,
            "access_token": string,
            "ensemble": boolean,
            "deep_models": list of string,
            "traditional_models": list of string,
            "number_models": int
        }
    '''
    ## Handling feature if any
    if 'features' in request_data:
        features = request_data["features"]
    ## Handling link_commit if any
    if 'link_commit' in request_data:
        commit_info = extract_info_from_commit_link(request_data['link_commit'], request_data['access_token'])
    ## Handling commit_info if any
    if 'commit_info' in request_data:
        commit_info = request_data["commit_info"]

    # Data Preprocessing
    ## Must return a model_request like below
    '''
        {
            "id": string,
            "input": input_type
        }
    '''
    model_name_to_model_input = {}
    ## Preprocessing data for deep models
    for model in request_data['traditional_models']:
        model_name_to_model_input[model] = {
            'id': request_data['id'],
            "input": features
        }
    ## Preprocessing data for deep models
    for model in request_data['deep_models']:
        model_name_to_model_input[model] = {
            'id': request_data['id']
        }
        with open(f"model_parameters/{model}.json", 'r') as file:
            params = json.load(file)
        input = preprocess_data[model](commit_info, params)
        model_name_to_model_input[model]['input'] = input
        model_name_to_model_input[model]['parameters'] = params

    # for model in request_data['deep_models']:
    #     print(model)
    #     print(model_name_to_model_input[model]['parameters'])

    # Forward to model
    output = {}
    for model in request_data["traditional_models"] + request_data["deep_models"]:
        send_message = model_name_to_model_input[model]
        print(send_message)
        model_response = requests.post(api_lists[model], json=send_message)
        if model_response.status_code == 200:
            model_response = model_response.json()
            # if app.debug:
            #     print(model_response)
            output[model] = model_response['output']
        else:
            print('Error:', model_response.status_code)

    # Create response like form below:
    '''
        {
            "id": string,
            "output": dictionary
        }
    '''
    return {
        'id': request_data['id'],
        'output': output
    }

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")