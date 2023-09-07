from urllib.parse import urlparse
import json, argparse, time, os, requests, logging
from auto_extract.RepositoryExtractor import RepositoryExtractor

# Configure the logging
logging.basicConfig(
    filename='main_log.log',
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
    available_traditional_models = ['lapredict', 'earl', 'tlel', 'jitline']
    parser.add_argument('-deep', nargs='+', type=str, default=[], choices=available_deep_models, help='list of deep learning models')
    parser.add_argument('-traditional', nargs='+', type=str, default=[], choices=available_traditional_models, help='list of machine learning models')

    parser.add_argument('-device', type=str, default='cpu', help='Ex: cpu, cuda, cuda:1')

    parser.add_argument('-debug', action='store_true', help='allow debug print')

    return parser

if __name__ == '__main__':
    # Start the timer
    start_time = time.time()
    
    params = read_args().parse_args()

    if params.debug:
        print("Repo: ", params.repo)
        print("Commit hash: ", params.commit_hash)
        print("Commit link: ", params.commit)
        print("Access token: ", params.access_token)
        print("Feature:", params.feature)
        print("Ensemble:", params.ensemble)
        print("DL models:", params.deep)
        print("ML models:", params.traditional)

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
        if params.access_token == '':
            raise Exception(f'Github access token is required')
        parsed_url = urlparse(params.commit)
        if parsed_url.hostname == 'github.com' and '/commit/' in parsed_url.path:
            if params.debug:
                print(f'{params.commit} is a GitHub commit link')
            request['link_commit'] = params.commit
            if not os.path.exists("github_access_token.txt"):
                raise Exception(f'github_access_token.txt not found. Please create one and put your GitHub access token in it')
            else:
                with open("github_access_token.txt", "r") as f:
                    request['access_token'] = f.read().strip()
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
        extractor.get_repo_commits_info(main_language_only=True)
        extractor.extract_repo_k_features()
        request["features"] = extractor.features[params.commit_hash]
        request['commit_info'] = extractor.commits[params.commit_hash]
                
    if params.debug:
        # Create a copy of the dictionary
        dict_without_token = request.copy()

        # Remove the "access_token" key from the copy
        dict_without_token.pop("access_token", None)
        dict_without_token.pop("commit_info", None)
        print("Request: ", json.dumps(dict_without_token, indent=4))

    # response = requests.post('http://35.78.205.195:5000/api/input_output', json=request)
    # if response.status_code == 200:
    #     print("Response: ", json.dumps(response.json(), indent=4))
    # else:
    #     raise Exception(response.status_code)
    
    # End the timer
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Print the result
    print(f"The code took {execution_time:.2f} seconds to run.")