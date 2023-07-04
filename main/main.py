from urllib.parse import urlparse
import json
import argparse
import time
import os
import requests
from auto_extract.RepositoryExtractor import RepositoryExtractor

def extract_diff(diff):
    num_added_lines = 0
    list_file_changes = []
    for file_elem in list(diff.items()):
        file_path = file_elem[0]
        file_val = file_elem[1]
            
        file = {"file_name": file_path, "code_changes":[]}
        for ab in file_val["content"]:
            if "ab" in ab:
                continue
            hunk = {"added_code":[], "removed_code":[]}
            if "a" in ab:
                hunk["removed_code"] += [line.strip() for line in ab["a"]]
            if "b" in ab:
                hunk["added_code"] += [line.strip() for line in ab["b"]]
                num_added_lines += len(ab["b"])
            hunk["added_code"] = "\n".join(hunk["added_code"])
            hunk["removed_code"] = "\n".join(hunk["removed_code"])
            file["code_changes"].append(hunk)
        list_file_changes.append(file)
    return list_file_changes, num_added_lines

def commit_to_info(commit):
    list_file_changes, num_added_lines = extract_diff(commit["diff"])
    
    return {
            'commit_hash': commit["commit_id"],
            'commit_message': commit["commit_msg"],
            'main_language_file_changes': list_file_changes,
            'num_added_lines_in_main_language': num_added_lines,
        }

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-repo', type=str, default='', help='path to git repo')
    parser.add_argument('-commit_hash', type=str, default='HEAD', help='commit hash')
    available_languages = ["Python", "Java", "C++", "C", "C#", "JavaScript", "TypeScript", "Ruby", "PHP", "Go", "Swift"]
    parser.add_argument('-main_language', type=str, default='', choices=available_languages, help='Main language of repo')
    
    parser.add_argument('-commit', type=str, default='', help='commit link')
    parser.add_argument('-access_token', type=str, default='', help='user access token')
    
    # parser.add_argument('-ensemble', action='store_true', help='enable ensemble')
    available_ensemble = ['average', 'max', 'majority']
    parser.add_argument('-ensemble', nargs='+', type=str, default=[], choices=available_ensemble, help='list of deep learning models')
    parser.add_argument('-threshold', type=int, default=0.5, help='threshold')

    available_deep_models = ['deepjit', 'cc2vec', 'simcom', 'codebert_cc2vec']
    available_traditional_models = ['lapredict', 'earl', 'tler', 'jitline']
    parser.add_argument('-deep', nargs='+', type=str, default=[], choices=available_deep_models, help='list of deep learning models')
    parser.add_argument('-traditional', nargs='+', type=str, default=[], choices=available_traditional_models, help='list of machine learning models')

    parser.add_argument('-device', type=str, default='cpu', help='Ex: cpu, cuda, cuda:1')

    parser.add_argument('-debug', action='store_true', help='allow debug print')

    return parser

if __name__ == '__main__':
    # Start the timer
    start_time = time.time()
    
    params = read_args().parse_args()

    # if params.debug:
    #     print("Repo: ", params.repo)
    #     print("Commit hash: ", params.commit_hash)
    #     print("Commit link: ", params.commit)
    #     print("Access token: ", params.access_token)
    #     print("Feature:", params.feature)
    #     print("Ensemble:", params.ensemble)
    #     print("DL models:", params.deep)
    #     print("ML models:", params.traditional)

    request = {
        "id": "main-" + str(int(time.time())),
        'ensemble': params.ensemble,
        'threshold': params.threshold,
        "deep_models": params.deep,
        "traditional_models": params.traditional,
        "device": params.device
    }
    
    if params.commit == '' and params.repo == '':
        raise Exception("-commit, -repo, atleast one of these is required")

    if params.commit != '':
        if len(params.deep) == 0:
            raise Exception(f'Atleast 1 deep learning model is required')
        if params.access_token == '':
            raise Exception(f'Github access token is required')
        parsed_url = urlparse(params.commit)
        if parsed_url.hostname == 'github.com' and '/commit/' in parsed_url.path:
            # if params.debug:
            #     print(f'{params.commit} is a GitHub commit link')
            request['link_commit'] = params.commit
            request['access_token'] = params.access_token
        else:
            raise Exception(f'{params.commit} not a GitHub commit link')

    if params.repo != '':
        # commit_info = extract_info_from_repo_path(params.repo, params.commit_hash)
        if params.commit_hash == '':
            raise Exception(f'Commit hash is required')
        if params.main_language == '':
            raise Exception(f"Repository's main language is required")
        current_dir = os.getcwd()
        extractor = RepositoryExtractor(params.repo, current_dir, params.main_language)
        if len(params.traditional) == 0:
            commit = extractor.get_commit_info(params.commit_hash, [params.main_language])
            request['commit_info'] = commit_to_info(commit)
        else:
            extractor.get_repo_commits_info(main_language_only=True)
            extractor.extract_repo_k_features()
            feature = extractor.features[params.commit_hash]
            request["features"] = feature
            if len(params.deep) > 0:
                commit = extractor.commits[params.commit_hash]
                request['commit_info'] = commit_to_info(commit)
                
    if params.debug:
        print("Request: ", json.dumps(request, indent=4))

    response = requests.post('http://localhost:5000/api/input_output', json=request)
    if response.status_code == 200:
        print("Response: ", json.dumps(response.json(), indent=4))
    else:
        raise Exception(response.status_code)
    
    # End the timer
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Print the result
    print(f"The code took {execution_time:.2f} seconds to run.")