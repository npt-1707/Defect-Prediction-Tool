
def deep_preprocess(commit_info):
    # Extract commit message
    commit_message = commit_info['commit_message']
    commit = commit_info['main_language_file_changes']
    