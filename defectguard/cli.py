import argparse, os, getpass, time, json
from urllib.parse import urlparse
from .extractor.RepositoryExtractor import RepositoryExtractor
from .utils.utils import commit_to_info, SRC_PATH, sort_by_predict
from .models.deepjit.warper import DeepJIT
from .models.cc2vec.warper import CC2Vec
from .models.simcom.warper import SimCom
from .models.lapredict.warper import LAPredict
from .models.tlel.warper import TLEL
from .models.jitline.warper import JITLine
from argparse import Namespace
from .utils.logger import ic
from datetime import datetime

__version__ = "0.1.0"


def read_args():
    available_languages = [
        "Python",
        "Java",
        "C++",
        "C",
        "C#",
        "JavaScript",
        "TypeScript",
        "Ruby",
        "PHP",
        "Go",
        "Swift",
    ]
    models = ["deepjit", "cc2vec", "simcom", "lapredict", "tlel", "jitline"]
    dataset = ["gerrit", "go", "platform", "jdt", "qt", "openstack"]
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )

    parser.add_argument("-repo", type=str, default="", help="Path to git repository")
    parser.add_argument(
        "-commit_hash", nargs="+", type=str, default=[], help="List of commit hashes"
    )
    parser.add_argument("-top", type=int, default=0, help="Number of top commits")
    parser.add_argument(
        "-main_language",
        type=str,
        default="",
        choices=available_languages,
        help="Main language of repo",
    )

    parser.add_argument("-github_link", type=str, default="", help="GitHub link")

    parser.add_argument(
        "-models",
        nargs="+",
        type=str,
        default=[],
        choices=models,
        help="List of deep learning models",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default="openstack",
        choices=dataset,
        help="Dataset's name",
    )
    parser.add_argument("-cross", action="store_true", help="Cross project")
    parser.add_argument("-uncommit", action="store_true", help="Include uncommit in list when using -top")

    parser.add_argument(
        "-device", type=str, default="cpu", help="Eg: cpu, cuda, cuda:1"
    )

    parser.add_argument("-sort", action="store_true", help="Sort output of model by predict score")
    parser.add_argument("-debug", action="store_true", help="Turn on system debug print")
    parser.add_argument("-log_to_file", action="store_true", help="Logging to file instead of stdout")

    return parser


def init_model(model_name, dataset, cross, device):
    project = "cross" if cross else "within"
    match model_name:
        case "deepjit":
            return DeepJIT(dataset=dataset, project=project, device=device)
        case "cc2vec":
            return CC2Vec(dataset=dataset, project=project, device=device)
        case "simcom":
            return SimCom(dataset=dataset, project=project, device=device)
        case "lapredict":
            return LAPredict(dataset=dataset, project=project, device=device)
        case "tlel":
            return TLEL(dataset=dataset, project=project, device=device)
        case "jitline":
            return JITLine(dataset=dataset, project=project, device=device)
        case _:
            raise Exception("No such model")


def main():
    params = read_args().parse_args()

    if not params.debug:
        ic.disable()
    
    if params.log_to_file:
        # Create a folder named 'logs' if it doesn't exist
        if not os.path.exists(f'{SRC_PATH}/logs'):
            os.makedirs(f'{SRC_PATH}/logs')

        # Define a file to log IceCream output
        log_file_path = os.path.join(f'{SRC_PATH}/logs', 'logs.log')

        # Replace logging configuration with IceCream configuration
        ic.configureOutput(prefix=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | ', outputFunction=lambda x: open(log_file_path, 'a').write(x + '\n'))

    ic("Start DefectGuard")
    start_whole_process_time = time.time()

    user_input = {
        "models": params.models,
        "dataset": params.dataset,
        "cross": params.cross,
        "device": params.device,
    }

    ic(user_input)

    # User's input handling
    if params.github_link != "":
        parsed_url = urlparse(params.github_link)
        if parsed_url.hostname == "github.com" and "/commit/" in parsed_url.path:
            user_input["link_commit"] = params.github_link
            if not os.path.exists("github_access_token.txt"):
                password = getpass.getpass("Enter your GitHub access token: ")
                file_name = "github_access_token.txt"
                with open(file_name, "w") as file:
                    file.write(password)
                print(f"GitHub access token has been saved to {file_name}")
            else:
                with open("github_access_token.txt", "r") as f:
                    user_input["access_token"] = f.read().strip()
        else:
            raise Exception(f"{params.github_link} not a GitHub commit link")

        owner, repo_name = parsed_url.path[: parsed_url.path.find("/commit/")].split(
            "/"
        )[1:3]

        extract_config = {
            "mode": "online",
            "github_token_path": os.path.join(params.repo, "github_access_token.txt"),
            "github_owner": owner,
            "github_repo": repo_name,
        }

    if params.repo != "":
        if params.main_language == "":
            raise Exception(f"Repository's main language is required")
        if params.top == 0 and len(params.commit_hash) == 0:
            raise Exception(f"-top or -commit_hash is required")
        if params.top > 0 and len(params.commit_hash) > 0:
            raise Exception(f"-top and -commit_hash cannot be used at the same time")
        extract_config = {
            "mode": "local",
            "local_repo_path": params.repo,
            "main_language": params.main_language,
        }

        # Extract info from user's repo

    # -----THANH-------
    start_extract_time = time.time()

    save_path = os.path.join(params.repo, "extractor")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    extract_config["save_path"] = save_path
    extract_config["to_csv"] = False

    extractor = RepositoryExtractor()
    extractor.config_repo(Namespace(**extract_config))
    if len(params.commit_hash) > 0:
        commits, features, not_found_ids = extractor.get_commits(params.commit_hash)
    elif params.uncommit:
        params.commit_hash = extractor.get_top_commits(params.repo, params.top, params.uncommit)
        commits, features, not_found_ids = extractor.get_commits(params.commit_hash)
    else:
        params.commit_hash = extractor.get_top_commits(params.repo, params.top)
        commits, features, not_found_ids = extractor.get_commits(params.commit_hash)

    user_input["commit_hashes"] = [id for id in params.commit_hash if id not in not_found_ids]
    user_input["features"] = features
    user_input["commit_info"] = []
    for i in range(len(user_input["commit_hashes"])):
        user_input["commit_info"].append(commit_to_info(commits[i]))

    end_extract_time = time.time()
    # -----THANH-------

    if len(user_input["commit_info"]) > 0:
        # Load Model
        model_list = {}
        for model in params.models:
            model_list[model] = init_model(
                model, params.dataset, params.cross, params.device
            )

        # Inference
        outputs = {'no_code_change_commit': not_found_ids}
        for model in model_list.keys():
            start_inference_time = time.time()

            outputs[model] = sort_by_predict(model_list[model].handle(user_input)) if params.sort else model_list[model].handle(user_input)

            end_inference_time = time.time()

            ic(
                f"Inference time of {model}: {end_inference_time - start_inference_time}"
            )

        print(json.dumps(outputs, indent=2))

    end_whole_process_time = time.time()

    ic(f"Extract features time: {end_extract_time - start_extract_time}")
    ic(
        f"Whole process time: {end_whole_process_time - start_whole_process_time}"
    )
