import argparse, os, getpass, sys
from urllib.parse import urlparse
from .extractor.RepositoryExtractor import RepositoryExtractor
from .utils.utils import commit_to_info
from .models.deepjit.handler import DeepJIT
from .models.cc2vec.handler import CC2Vec
from .models.simcom.handler import SimCom
from .models.lapredict.handler import LAPredict
from .models.tlel.handler import TLEL
from .models.jitline.handler import JITLine
from argparse import Namespace
from .utils.logger import logger

__version__ = "0.1.0"

def read_args():
    available_languages = ["Python", "Java", "C++", "C", "C#", "JavaScript", "TypeScript", "Ruby", "PHP", "Go", "Swift"]
    models = ['deepjit', 'cc2vec', 'simcom', 'lapredict', 'tlel', 'jitline']
    dataset = ['gerrit', 'go', 'platform', 'jdt', 'qt', 'openstack']
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)

    parser.add_argument('-repo', type=str, default='', help='Path to git repository')
    parser.add_argument('-commit_hash', nargs='+', type=str, default=['HEAD'], help='List of commit hashes')
    parser.add_argument('-main_language', type=str, default='', choices=available_languages, help='Main language of repo')
    
    parser.add_argument('-github_link', type=str, default='', help='GitHub link')

    parser.add_argument('-models', nargs='+', type=str, default=[], choices=models, help='List of deep learning models')
    parser.add_argument('-dataset', type=str, default='openstack', choices=dataset, help='Dataset\'s name')
    parser.add_argument('-cross', action='store_true', help='Cross project')

    parser.add_argument('-device', type=str, default='cpu', help='Eg: cpu, cuda, cuda:1')

    return parser

def init_model(model_name, dataset, cross, device):
    project = 'cross' if cross else 'within'
    match model_name:
        case 'deepjit':
            return DeepJIT(
                dataset=dataset,
                project=project,
                device=device
            )
        case 'cc2vec':
            return CC2Vec(
                dataset=dataset,
                project=project,
                device=device
            )
        case 'simcom':
            return SimCom(
                dataset=dataset,
                project=project,
                device=device
            )
        case 'lapredict':
            return LAPredict(
                dataset=dataset,
                project=project,
                device=device
            )
        case 'tlel':
            return TLEL(
                dataset=dataset,
                project=project,
                device=device
            )
        case 'jitline':
            return JITLine(
                dataset=dataset,
                project=project,
                device=device
            )
        case _:
            raise Exception('No such model')


def main():
    params = read_args().parse_args()

    user_input = {
        "models": params.models,
        'dataset': params.dataset,
        'cross': params.cross,
        "device": params.device
    }
    
    # User's input handling
    if params.github_link == '' and params.repo == '':
        raise Exception("-commit, -repo, atleast one of these is required")

    if params.github_link != '':
        parsed_url = urlparse(params.github_link)
        if parsed_url.hostname == 'github.com' and '/commit/' in parsed_url.path:
            user_input['link_commit'] = params.github_link
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
            raise Exception(f'{params.github_link} not a GitHub commit link')
        
        owner, repo_name = parsed_url.path[:parsed_url.path.find('/commit/')].split('/')[1:3]
        
        extract_config = {
            "mode": "online",
            "github_token_path": os.path.join(sys.path[0], "github_access_token.txt"),
            "github_owner": owner,
            "github_repo": repo_name,
        }

    if params.repo != '':
        if params.commit_hash == '':
            raise Exception(f'Commit hash is required')
        if params.main_language == '':
            raise Exception(f"Repository's main language is required")
        
        extract_config = {
            "mode": "local",
            "local_repo_path": params.repo,
            "main_language": params.main_language,
        }

    # Extract info from user's repo

    #-----THANH-------
    save_path = os.path.join(sys.path[0], "auto_extract/save")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    extract_config["save_path"]=save_path
    extract_config["to_csv"]=False        
        
    extractor = RepositoryExtractor()
    extractor.config_repo(Namespace(**extract_config))
    commits, features = extractor.get_commits(params.commit_hash)
    user_input["features"] = features
    user_input["commit_info"] = []
    for i in range(len(params.commit_hash)):
        user_input["commit_info"].append(commit_to_info(commits[i]))
    #-----THANH-------

    # Load Model
    model_list = {}
    for model in params.models:
        model_list[model] = init_model(model, params.dataset, params.cross, params.device)

    # Inference
    outputs = {}
    for model in model_list.keys():
        outputs[model] = model_list[model].handle(user_input)

    print(outputs)