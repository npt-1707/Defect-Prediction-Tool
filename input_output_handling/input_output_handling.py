from flask import Flask, request
import requests
from utils import extract_info_from_commit_link

# Dictionary mapping model_name and model_preprocessing
preprocess_data = {}

app = Flask(__name__)

@app.route('/api/template', methods=['POST'])
def template():
    # Get request_data from main.py
    request_data = request.get_json()
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
            "vectorized_feature": vectorized_feature
        }
    '''

    ## Create a dict mapping model_name and model_input
    model_name_to_model_input = {}

    ## 
    if app.debug:
        print(request_data['traditional_models'] + request_data['deep_models'])
    for model in request_data['traditional_models'] + request_data['deep_models']:
        input = preprocess_data[model](commit_info)
        model_name_to_model_input[model] = input
        
    # Forward to model
    # for model in request_data["model"]:
    #     model_response = requests.post(f'http://localhost:5000/api/{model}', json=model_request.get_json())
    #     if model_response.status_code == 200:
    #         print(model_response.json())
    #     else:
    #         print('Error:', model_response.status_code)

    # Create response like form below:
    '''
        {
            "id": string,
            "output": dictionary
        }
    '''
    output = {'result': 0}
    return {
        'id': request_data['id'],
        'output': output
    }

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")