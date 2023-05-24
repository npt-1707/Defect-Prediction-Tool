import requests, os
from github import Github
import utils, subprocess
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


def extract_owner_and_repo(commit_link):
    # Split the link by '/' character
    parts = commit_link.split('/')

    # Get the owner and repository name from the parts
    owner = parts[3]
    repo_name = parts[4]
    commit_hash = parts[6]

    # Return the owner and repository name
    return owner, repo_name, commit_hash

def extract_info_from_commit_link(link: str) -> dict:

    # # Replace with your GitHub access token
    # access_token = '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'

    # # Create a GitHub instance
    # g = Github(access_token)

    owner, repo_name, commit_hash = extract_owner_and_repo(link)

    # # Get the repository
    # repo = g.get_repo(f'{owner}/{repo_name}')

    # if not os.path.exists(f'repo/{repo_name}'):
    #     clone_url = repo.clone_url
    #     utils.Repo.clone_from(clone_url, f'repo/{repo_name}')

    repo = Repo(f'repo/{repo_name}') 

    commit = repo.commit(commit_hash)

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

    commit = repo.commit(commit_hash)
    # Define the command to be executed
    command = f"cd repo/{repo_name} && git show {commit_hash}"

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
        'commit_hash': commit_hash,
        'commit_message': commit.message,
        'main_language_file_changes': file_changes,
        'num_added_lines_in_main_language': num_added_lines,
    }