from flask import Flask, request
import requests

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

    # Data Preprocessing
    # Must return a model_request like below:
    '''
        {
            "id": string,
            "number_models": int,
            "vectorized_feature": vectorized_feature
        }
    '''

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