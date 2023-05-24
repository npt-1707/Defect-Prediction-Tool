import pickle, torch
from preprocess.padding import padding_data

def deep_preprocess(commit_info, params):
    # Extract commit message
    commit_message = commit_info['commit_message']
    commit = commit_info['main_language_file_changes']
    
    dictionary = pickle.load(open("preprocess/dictionary/platform_dict.pkl", 'rb'))   
    dict_msg, dict_code = dictionary

    pad_msg = padding_data(data=commit_message, dictionary=dict_msg, params=params, type='msg')        
    pad_code = padding_data(data=commit, dictionary=dict_code, params=params, type='code')

    # Using Pytorch Dataset and DataLoader
    code = {
        "code": pad_code.tolist(),
        "message": pad_msg.tolist()
    }
    
    return (code, dict_msg, dict_code)