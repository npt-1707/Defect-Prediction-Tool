from dvc.api import DVCFileSystem
import os

def download(model_name, destination_path='models/'):
    # Check if the folder already exists locally
    local_folder_path = os.path.join(destination_path, model_name)
    if os.path.exists(local_folder_path):
        print(f"Folder '{model_name}' already exists locally at '{local_folder_path}'")
        return

    fs = DVCFileSystem()
    fs.get(local_folder_path, local_folder_path, recursive=True)

    print(f"Folder '{model_name}' downloaded to '{local_folder_path}'")