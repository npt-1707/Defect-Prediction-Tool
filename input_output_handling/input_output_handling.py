from flask import Flask, request
import requests, json
from utils import extract_info_from_commit_link
from preprocess.deepjit.preprocess import deepjit_preprocess

# Dictionary mapping model_name and model_preprocessing
preprocess_data = {
    'deepjit': deepjit_preprocess
}

app = Flask(__name__)

@app.route('/api/input_output', methods=['POST'])
def template():
    # Get request_data from main.py 
    request_data = request.get_json()
    if app.debug:
        print(request_data)

    #----- Handling request -------------------#
    '''
        request {
            "id": string,
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
        # THANH'S CODE
        pass
    ## Handling link_commit if any
    if 'link_commit' in request_data:
        commit_info = extract_info_from_commit_link(request_data['link_commit'], request_data['access_token'])

    # Data Preprocessing
    ## Must return a model_request like below
    '''
        {
            "id": string,
            "input": input_type
        }
    '''
    ## A dict mapping model_name and model_input
    model_input = {
        'id': request_data['id']
    }
    model_name_to_model_input = {}
    ## Preprocessing data for deep models
    for model in request_data['traditional_models']:
        # THANH'S CODE
        pass
    ## Preprocessing data for deep models
    for model in request_data['deep_models']:
        with open(f"model_parameters/{model}.json", 'r') as file:
            params = json.load(file)
        input = preprocess_data[model](commit_info, params)
        model_input['input'] = input
        model_input['parameters'] = params 
        model_name_to_model_input[model] = model_input

    # Forward to model
    output = {}
    for model in request_data["traditional_models"] + request_data["deep_models"]:
        model_response = requests.post(f'http://localhost:5001/api/{model}', json=model_name_to_model_input[model])
        if model_response.status_code == 200:
            model_response = model_response.json()
            if app.debug:
                print(model_response)
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