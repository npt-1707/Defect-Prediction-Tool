from importlib.resources import files
import dvc.api, os, pickle

config_path = str(files('utils').joinpath('colab-385406-e848129cc804.json'))

CONFIG = {
    'remote': {
        'storage': {
            'gdrive_client_id': '515220301377-6015nql03h01qkg5maahiusa1int0b54.apps.googleusercontent.com',
            'gdrive_client_secret': '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',
            'gdrive_use_service_account': True,
            'gdrive_service_account_user_email': 'defectguard@colab-385406.iam.gserviceaccount.com',
            'gdrive_service_account_json_file_path': config_path,
        },
    },
}

MODELS = {
    'deepjit': {
        'version': {
            'platform_within': 'platform_within.pt',
        },
        'dictionary': {
            'platform': 'platform_dict.pkl'
        },
        'parameters': 'deepjit.json',
    },
}

def save_file(file_contents, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(file_contents, file)

def load_file(file_path):
    with open(file_path, 'rb') as file:
        loaded_binary_data = pickle.load(file)
    return loaded_binary_data

def download(model_name, file_name, cache):
    # Check if the file exists locally
    file_path = str(files('defectguard').joinpath(f'models/{model_name}/{file_name}'))
    print(f"File's path: {file_path}")
    if not os.path.isfile(file_path):
        # File doesn't exist, download it
        print(f"File '{file_name}' does not exist locally. Downloading...")

        file_contents = dvc.api.read(
            f'models/{model_name}/{file_name}',
            repo="https://github.com/manhlamabc123/DefectGuard",
            mode='rb',
            config=CONFIG
        )

        if cache:
            save_file(file_contents, file_path)

        print(f"File '{file_name}' downloaded.")

        return file_contents
    else:
        print(f"Load file '{file_name}' from local.")

        file_contents = load_file(file_path)

        return file_contents

def load_metadata(model_name, version, dictionary=None, cache=True):
    model_metadata = []
    model = MODELS[model_name]

    model_version = download(model_name, model['version'][version], cache)
    model_metadata.append(model_version)

    if 'parameters' in model:
        model_parameters = download(model_name, model['parameters'], cache)
        model_metadata.append(model_parameters)

    if 'dictionary' in model:
        model_dictionary = download(model_name, model['dictionary'][dictionary], cache)
        model_metadata.append(model_dictionary)

    return tuple(model_metadata)