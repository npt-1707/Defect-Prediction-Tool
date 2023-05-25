import argparse, os, time, requests
from urllib.parse import urlparse
import subprocess
from git import Repo

def get_file_changes_from_diff(diff, file_format):
    # sourcery skip: extract-duplicate-method
    lines = diff.split("\n")
    num_added_lines = 0
    list_file_changes = []
    file = None
    for i in range(len(lines)):
        line = lines[i]            
        if (line.startswith("+++") or line.startswith("---")):  # new file had been changed
            file_name = line[6:]
            if type(file) is dict:
                for hunk in file["code_changes"]:
                    hunk["added_code"] = "\n".join(hunk["added_code"])
                    hunk["removed_code"] = "\n".join(hunk["removed_code"])
                    list_file_changes.append(file)

            file = None if file_format not in line else {'file_name': file_name, 'code_changes': []}
        elif file is None:
            continue
        elif line.startswith("@"): # a file_changes in changed file
            file['code_changes'].append({"added_code":[], "removed_code":[]})
        elif line.startswith("+"):
            file['code_changes'][len(file['code_changes'])-1]["added_code"].append(line[1:].strip())
            num_added_lines += 1
        elif line.startswith("-"):
            file['code_changes'][len(file['code_changes'])-1]["removed_code"].append(line[1:].strip())
        
        if i == len(lines) - 1:
            try:
                for hunk in file['code_changes']:
                    hunk["added_code"] = "\n".join(hunk["added_code"])
                    hunk["removed_code"] = "\n".join(hunk["removed_code"])
                    list_file_changes.append(file)
            except Exception:
                pass
        
    return list_file_changes, num_added_lines

def extract_repo_name(path: str) -> str:
    # Split the link by '/' character
    parts = path.split('/')

    # Get repository name from the parts
    repo_name = parts[-1]

    # Return the owner and repository name
    return repo_name

def extract_info_from_repo_path(path: str, owner: str) -> dict:
    repo_name = extract_repo_name(path)

    repo = Repo(f'{repo_name}') 

    commit = repo.commit('HEAD')

    api_url = f'https://api.github.com/repos/{owner}/{repo_name}'

    # Send a GET request to the GitHub API
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the JSON response data
        data = response.json()
        # Extract the main programming language from the response
        main_language = data['language']
        print(f'The main programming language of the repository is: {main_language}')
    else:
        print(f'Failed to get repository details. Status code: {response.status_code}')

    commit = repo.commit('HEAD')
    # Define the command to be executed
    command = f"cd {repo_name} && git show HEAD"

    # Run the command and capture the output
    output = subprocess.check_output(command, shell=True, text=True)

    match main_language:
        case 'Python':
            file_format = '.py'
        case 'C++':
            file_format = '.cpp'
        case 'C':
            file_format = '.c'
        case 'Java':
            file_format = '.java'
        case 'JavaScript':
            file_format = '.js'

    file_changes, num_added_lines = get_file_changes_from_diff(output, file_format)
    return {
        'commit_hash': 'HEAD',
        'commit_message': commit.message,
        'main_language_file_changes': file_changes,
        'num_added_lines_in_main_language': num_added_lines,
    }

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-repo', type=str, default='', help='path to git repo')
    parser.add_argument('-owner', type=str, default='', help='repo owner')
    parser.add_argument('-commit', type=str, default='', help='commit link')
    parser.add_argument('-access_token', type=str, default='', help='user access token')
    parser.add_argument('-feature', type=str, default='', help='.csv file path')

    parser.add_argument('-ensemble', action='store_true', help='enable ensemble')

    available_deep_models = ['deepjit', 'cc2vec', 'simcom', 'codebert_cc2vec']
    available_traditional_models = ['lapredict', 'earl', 'tler', 'jitline']
    model = parser.add_mutually_exclusive_group()
    model.add_argument('-deep', nargs='+', type=str, default=[], choices=available_deep_models, help='list of deep learning models')
    model.add_argument('-traditional', nargs='+', type=str, default=[], choices=available_traditional_models, help='list of machine learning models')

    parser.add_argument('-debug', action='store_true', help='allow debug print')

    return parser

# sourcery skip: raise-specific-error
if __name__ == '__main__':
    params = read_args().parse_args()

    if params.debug:
        print("Commit link: ", params.commit)
        print("Access token: ", params.access_token)
        print("Feature:", params.feature)
        print("Ensemble:", params.ensemble)
        print("DL models:", params.deep)
        print("ML models:", params.traditional)

    request = {
        "id": "main-" + str(int(time.time())),
        'ensemble': params.ensemble,
        "deep_models": params.deep,
        "traditional_models": params.traditional,
        "number_models": len(params.deep) + len(params.traditional)
    }
    
    if params.feature == '' and params.commit == '' and params.repo == '':
        raise Exception("-commit, -feature, -repo, atleast one of these is required")

    if params.feature != '':
        if len(params.traditional) == 0:
            raise Exception(f'Atleast 1 traditional model is required')
        # Check if file exists
        if not os.path.isfile(params.feature):
            raise Exception(f'{params.feature} does not exist')
        else:
            # Read file contents
            with open(params.feature, 'r') as f:
                file_contents = f.read()
                if params.debug:
                    print(file_contents)
                request['features'] = file_contents

    if params.commit != '':
        if len(params.deep) == 0:
            raise Exception(f'Atleast 1 deep learning model is required')
        if params.access_token == '':
            raise Exception(f'Github access token is required')
        parsed_url = urlparse(params.commit)
        if parsed_url.hostname == 'github.com' and '/commit/' in parsed_url.path:
            if params.debug:
                print(f'{params.commit} is a GitHub commit link')
            request['link_commit'] = params.commit
            request['access_token'] = params.access_token
        else:
            raise Exception(f'{params.commit} not a GitHub commit link')

    if params.repo != '':
        if params.owner == '':
            raise Exception(f"Reporisoty's owner is required")
        commit_info = extract_info_from_repo_path(params.repo, params.owner)
        request['commit_info'] = commit_info
        
    if params.debug:
        print(request)

    response = requests.post('http://localhost:5000/api/input_output', json=request)
    if response.status_code == 200:
        print(response.json())  # {'message': 'accepted'}
    else:
        raise Exception(response.status_code)