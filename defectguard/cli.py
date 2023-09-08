import argparse, json, os, getpass
from urllib.parse import urlparse

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

def main():
    params = read_args().parse_args()

    user_input = {
        'ensemble': params.ensemble,
        'threshold': params.threshold,
        "deep_models": params.deep,
        "traditional_models": params.traditional,
        "device": params.device
    }
    
    if params.commit == '' and params.repo == '':
        raise Exception("-commit, -repo, atleast one of these is required")

    if params.commit != '':
        parsed_url = urlparse(params.commit)
        if parsed_url.hostname == 'github.com' and '/commit/' in parsed_url.path:
            user_input['link_commit'] = params.commit
            if not os.path.exists("github_access_token.txt"):
                password = getpass.getpass("Enter your GitHub access token: ")
                file_name = "github_access_token.txt"
                with open(file_name, "w") as file:
                    file.write(password)
                print(f"GitHub access token has been saved to {file_name}")
            else:
                with open("github_access_token.txt", "r") as f:
                    user_input['access_token'] = f.read().strip()
        else:
            raise Exception(f'{params.commit} not a GitHub commit link')

    if params.repo != '':
        if params.commit_hash == '':
            raise Exception(f'Commit hash is required')
        if params.main_language == '':
            raise Exception(f"Repository's main language is required")
                
    if params.debug:
        dict_without_token = user_input.copy()
        dict_without_token.pop("access_token", None)
        dict_without_token.pop("commit_info", None)
        print("User's input: ", json.dumps(dict_without_token, indent=4))