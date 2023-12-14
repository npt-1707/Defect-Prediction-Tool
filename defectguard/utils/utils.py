from importlib.resources import files
import os, gdown

SRC_PATH = str(files('defectguard'))

IDS = {
    'deepjit': {
        'hyperparameters': '1US-qs1Ly9wfRADcEMLBtTa8Ao91wNwOv',
        'platform_within': '11Qjj84btTuqbYGpphmin0spMuGgJerNa',
        'platform_cross': '1BTo26TU2G58OsBxoM-EidyijfnQZXuc4',
        'platform_dictionary_within': '1C6nVSr0wLS8i8bH_IptCUKdqrdiSngcv',
        'platform_dictionary_cross': '1XY4J3bCKo7IWMXcA2DJqVzzAD8XOZi-b',
    },
    'cc2vec': {
        'qt_dictionary': '1GTgkEcZdwVDzp0Tq86Uch_j5f4assfSU',
        'dextended_qt_within': '1uuSeYee40Azw1jWD2ln287GZ49ApTMxL',
        'hyperparameters': '1Zim5j4eKfl84r4mGDmRELwAwg8oVQ5uJ',
        'cc2vec_qt_within': '1-ZQjygr6myPj4ml-VyyiyrGKiL0HV2Td',
    },
    'simcom': {
        'hyperparameters': '1Y9pt5EShp5Z2Q2Ff6EjHp0fxByXWViw6',
        'sim_qt_within': '1zee0mzb1bjnnim-WMer2K6e3WlWMNC0D',
        'sim_platform_within': '1SJ8UnaMQlaB58E7VsQWbHFmh2ms0QFg_',
        'sim_openstack_within': '1iJDpDLL19d_dp7mdjxu0ADqN25Hgyxuk',
        'sim_jdt_within': '1PPz385vq3cuuTf5pqM4k4c018rXoN1If',
        'sim_go_within': '1nknqQPbgJJXXCJ5pa4G27ymcEY2goxBq',
        'sim_gerrit_within': '1CmsiNXe5qXtEw6rG7IXLq2KVLslhOcij',
        'platform_dictionary': '19h6kUCiHXTsijXUEArxSx4afS4hdKrvx',
        'com_platform_within': '1KmUkYFVaH34kBA4pW8qXgv1JV9qCRtkx',
    },
    'lapredict': {
        'qt_within': '1HG-cscwvWAjWXlovqoyba1k5Do5NJh6b',
        'platform_within': '1kcWcD1PUDSksX7p_vKBlpVV20S3pcVQ8',
        'openstack_within': '1Y3bGUGoDEaQyUJAJ1x2-rdvLbcbdq-nz',
        'jdt_within': '1vjH9u7ObFPXuTtAdqNZAM47eeuDhl1B-',
        'go_within': '1r1mSvWvt4S93cZPI_j_bOKppBI-laLlI',
        'gerrit_within': '1484sBLghCpPd3XCpHt9Gqd_hvdq7TPP1',
    },
    'tlel': {
        'qt_within': '1ZvwEQ6lbb_43_JBgEB6VnRxR7HNQZlOk',
        'platform_within': '1vS26ng_kZ5gdYESyWrfciMacXz74AzhZ',
        'openstack_within': '1yCOI_5inFnxH1EDS2JpA282UN7Zc1AXV',
        'jdt_within': '1GUEC7kFCybuoEetr-1Tis_6EmaWXgWwG',
        'go_within': '1siGmkBSq5qcuoxnhxo2Gc2_IhrVnLmWh',
        'gerrit_within': '1CI326L7vwokRXxwRdufzOvKtciPUj_TX',
    },
    'jitline': '',
}

def sort_by_predict(commit_list):
    # Sort the list of dictionaries based on the "predict" value in descending order
    sorted_list = sorted(commit_list, key=lambda x: x['predict'], reverse=True)
    return sorted_list

def vsc_output(data):
    # Extract the commit hashes from "no_code_change_commit"
    no_code_change_commits = data.get("no_code_change_commit", [])
    
    # Extract the "deepjit" list
    deepjit_list = data.get("deepjit", [])
    
    # Create a dictionary with keys from "no_code_change_commit" and values as -1
    new_dict = [{'commit_hash': commit, 'predict': -1} for commit in no_code_change_commits]
    
    # Append the new dictionary to the "deepjit" list
    deepjit_list += (new_dict)
    
    # Update the "deepjit" key in the original data
    data["deepjit"] = deepjit_list

    return data

def create_download_list(model_name, dataset, project):
    download_list = []
    dictionary = f'{dataset}_dictionary_{project}'
    version = f'{dataset}_{project}'

    if model_name == 'simcom':
        sim_dataset = f'sim_{version}'
        com_dataset = f'com_{version}'
        download_list.append(sim_dataset)
        download_list.append(com_dataset)
        download_list.append('hyperparameters')
        download_list.append(dictionary)
    elif model_name == 'cc2vec':
        cc2vec_dataset = f'cc2vec_{version}'
        dextended_dataset = f'dextended_{version}'
        download_list.append(cc2vec_dataset)
        download_list.append(dextended_dataset)
        download_list.append('hyperparameters')
        download_list.append(dictionary)
    elif model_name == 'deepjit':
        download_list.append(version)
        download_list.append('hyperparameters')
        download_list.append(dictionary)
    else:
        download_list.append(version)
    
    return download_list

def download_file(file_id, folder_path):
    if not os.path.isfile(folder_path):
        gdown.download(
            f'https://drive.google.com/uc?/export=download&id={file_id}',
            output=folder_path
            )

def download_folder(model_name, dataset, project):
    # Check if the file exists locally
    folder_path = f'{SRC_PATH}/models/metadata/{model_name}'

    if not os.path.exists(folder_path):
        # File doesn't exist, download it
        # Create the directory if it doesn't exist
        print(f"Directory: {folder_path}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Download model's metadata
    download_list = create_download_list(model_name, dataset, project)
    for item in download_list:
        download_file(IDS[model_name][item], f'{folder_path}/{item}')

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
    if commit:
        list_file_changes, num_added_lines = extract_diff(commit["diff"])
        
        return {
                'commit_hash': commit["commit_id"],
                'commit_message': commit['msg'],
                'main_language_file_changes': list_file_changes,
                'num_added_lines_in_main_language': num_added_lines,
            }
    else:
        return {}