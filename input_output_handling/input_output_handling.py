from flask import Flask, request
import requests, json
from utils import extract_info_from_commit_link
from preprocess.preprocess import deep_preprocess

# Dictionary mapping model_name and model_preprocessing
def model_template_preprocess(data):
    return 0

preprocess_data = {
    'model_template': deep_preprocess
}

app = Flask(__name__)

@app.route('/api/template', methods=['POST'])
def template():
    # Get request_data from main.py
    request_data = request.get_json()
    if app.debug:
        print(request_data)

    # Handling message
    '''
        {
            "id": string,
            "features": .csv file,
            "link_commit": string,
            "ensemble": boolean,
            "deep_models": list of string,
            "traditional_models": list of string,
            "number_models": int
        }
    '''
    ## Handling feature if any
    if 'features' in request_data:
        '''
            to THANH: em muon xu ly cai file .csv nhu nao thi lam o day
            define module ben utils roi keo sang day nhe
        '''
        pass
    ## Handling link_commit if any
    if 'link_commit' in request_data:
        commit_info = extract_info_from_commit_link(request_data['link_commit'])

    # Data Preprocessing
    ## Must return a model_request like below
    '''
        {
            "id": string,
            "input": input_type
        }
    '''
    ## Create a dict mapping model_name and model_input
    model_input = {
        'id': request_data['id']
    }
    model_name_to_model_input = {}
    ## Preprocessing data
    '''
        for model in request_data['traditional_models'] + request_data['deep_models']:
            input = preprocess_data[model](features)
            model_name_to_model_input[model] = input
    '''
    for model in request_data['deep_models']:
        # Load parameters
        with open("model_parameters/model_parameter.json", 'r') as file:
            params = json.load(file)
        input = preprocess_data[model](commit_info, params)
        model_input['input'] = input
        model_name_to_model_input[model] = model_input

    # Forward to model
    output = {}
    for model in request_data["deep_models"]:
        if app.debug:
            print(model_name_to_model_input[model])
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