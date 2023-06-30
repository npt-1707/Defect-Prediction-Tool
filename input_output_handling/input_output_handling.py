from flask import Flask, request
import requests, json
from github import Github
import os
import asyncio
import aiohttp
from utils import extract_owner_and_repo, commit_to_info
from preprocess.deepjit.preprocess import deepjit_preprocess
from preprocess.cc2vec.preprocess import cc2vec_preprocess
from auto_extract.RepositoryExtractor import RepositoryExtractor

# Dictionary mapping model_name and model_preprocessing
preprocess_data = {
    'deepjit': deepjit_preprocess,
    'cc2vec': cc2vec_preprocess,
    'simcom': deepjit_preprocess
}

api_lists = {
    'deepjit': "http://localhost:5001/api/deepjit",
    'cc2vec': "http://localhost:5002/api/cc2vec",
    'lapredict': "http://localhost:5003/api/lapredict"
}

ensemble_methods = {
    'average': lambda results: sum(results) / len(results),
    'max': lambda results: 1 if 1 in results else 0,
    'majority': lambda results: 1 if results.count(1) > results.count(0) else 0
}

app = Flask(__name__)

async def send_request(session, model, send_message, api_url):
    async with session.post(api_url, json=send_message) as response:
        if response.status == 200:
            model_response = await response.json()
            return model, model_response['output']
        else:
            return model, -1
        
async def send_requests(request_data, model_name_to_model_input):
    output = {}
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for model in request_data["traditional_models"] + request_data["deep_models"]:
            send_message = model_name_to_model_input[model]
            api_url = api_lists[model]
            task = asyncio.ensure_future(send_request(session, model, send_message, api_url))
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        for model, model_response in responses:
            output[model] = model_response
        
        return output

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
            "features": dict,
            "link_commit": string,
            "access_token": string,
            "ensemble": boolean,
            "deep_models": list of string,
            "traditional_models": list of string,
            "number_models": int
        }
    '''

    ## Handling link_commit if any
    if 'link_commit' in request_data:
        owner, name, commit_hash = extract_owner_and_repo(request_data['link_commit'])
        g = Github(request_data['access_token'])
        current_path = os.getcwd()
        extractor = RepositoryExtractor(g, owner, name, current_path)
        if len(request_data["traditional_models"]) > 0:
            extractor.get_repo_commits_info(main_language_only=True)
            extractor.extract_repo_k_features()
            features = extractor.features[commit_hash]
            if len(request_data["deep_models"]) > 0:
                commit = extractor.commits[commit_hash]
                commit_info = commit_to_info(commit)
        else:
            commit = extractor.get_commit_info(commit_hash, extractor.language)
            commit_info = commit_to_info(commit)
            
        os.chdir(current_path)
    ## Handling commit_info & features if any
    if 'commit_info' in request_data:
        commit_info = request_data["commit_info"]
    if 'features' in request_data:    
        features = request_data["features"]

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
            'id': request_data['id'],
            'device': request_data['device']
        }
        with open(f"model_parameters/{model}.json", 'r') as file:
            params = json.load(file)
        input = preprocess_data[model](commit_info, params)
        if model != 'simcom':
            model_name_to_model_input[model]['input'] = input
        else:
            model_name_to_model_input[model]['input'] = {}
            model_name_to_model_input[model]['input']['commit'] = input
            model_name_to_model_input[model]['input']['feature'] = features
        model_name_to_model_input[model]['parameters'] = params

    # for model in request_data['deep_models']:
    #     print(model)
    #     print(model_name_to_model_input[model]['parameters'])

    # Forward to model
    ## Using for loop
    # output = {}
    # for model in request_data["traditional_models"] + request_data["deep_models"]:
    #     send_message = model_name_to_model_input[model]
    #     model_response = requests.post(api_lists[model], json=send_message)
    #     if model_response.status_code == 200:
    #         model_response = model_response.json()
    #         # if app.debug:
    #         #     print(model_response)
    #         output[model] = model_response['output']
    #     else:
    #         output[model] = -1

    ## Using async
    output = asyncio.run(send_requests(request_data, model_name_to_model_input))

    # If ensemble learning is True
    if len(request_data['ensemble']) >= 0:
        results = list(output.values())
        results = [x for x in results if x != -1]
        if len(results) >= 2:
            output['emsemble_result'] = {}
            for method in request_data['ensemble']:
                if method != 'average':
                    results = [1 if prob >= request_data['threshold'] else 0 for prob in results]
                output['emsemble_result'][method] = ensemble_methods[method](results)
        else:
            output['emsemble_result'] = "Can not ensemble on 1 results."

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