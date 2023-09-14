import os, subprocess

def download(model_name, destination_path='models/'):
    # Check if the folder already exists locally
    local_folder_path = os.path.join(destination_path, model_name)
    if os.path.exists(local_folder_path):
        print(f"Folder '{model_name}' already exists locally at '{local_folder_path}'")
        return

    # Define the DVC command to pull the folder
    dvc_pull_command = f"dvc pull {local_folder_path}"

    # Run the DVC pull command
    try:
        subprocess.check_call(dvc_pull_command, shell=True)
        print("Folder pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling folder: {e}")