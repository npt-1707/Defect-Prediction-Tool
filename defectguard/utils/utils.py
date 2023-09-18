from importlib.resources import files
import os, gdown

SRC_PATH = str(files('defectguard'))

IDS = {
    'deepjit': {
        'platform_within': '1I4y2i_dWsgqWtj8ytAPjAL3mGIROYjY3',
        'hyperparameters': '1US-qs1Ly9wfRADcEMLBtTa8Ao91wNwOv',
        'platform_dictionary': '1XUq2KUwf6yT3zskeB-E8otcNnxvb4gFK',
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
    'lapredict': '',
    'tlel': '',
    'jitline': '',
}

def create_download_list(model_name, version, dictionary):
    download_list = ['hyperparameters']

    if model_name == 'simcom':
        sim_version = f'sim_{version}'
        com_version = f'com_{version}'
        download_list.append(sim_version)
        download_list.append(com_version)
    elif model_name == 'cc2vec':
        cc2vec_version = f'cc2vec_{version}'
        dextended_version = f'dextended_{version}'
        download_list.append(cc2vec_version)
        download_list.append(dextended_version)
    else:
        download_list.append(version)

    if dictionary is not None:
        download_list.append(f'{dictionary}_dictionary')
    
    return download_list

def download_file(file_id, folder_path):
    if not os.path.isfile(folder_path):
        gdown.download(
            f'https://drive.google.com/uc?/export=download&id={file_id}',
            output=folder_path
            )

def download_folder(model_name, version, dictionary=None):
    # Check if the file exists locally
    folder_path = f'{SRC_PATH}/models/{model_name}'
    print(f"Folder's path: {folder_path}")

    if not os.path.exists(folder_path):
        # File doesn't exist, download it
        print(f"'{model_name}' does not exist locally. Downloading...")

        # Create the directory if it doesn't exist
        print(f"Directory: {folder_path}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Download model's metadata
    download_list = create_download_list(model_name, version, dictionary)
    print(f"Download list: {download_list}")
    for item in download_list:
        download_file(IDS[model_name][item], f'{folder_path}/{item}')

    print(f"'{model_name}' downloaded.")