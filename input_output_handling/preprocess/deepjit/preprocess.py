import pickle
from preprocess.deepjit.padding import padding_data
from utils import hunks_to_code

def deepjit_preprocess(commit_info, params):
    # Extract commit message
    commit_message = commit_info['commit_message']
    commit = commit_info['main_language_file_changes']

    code = hunks_to_code(commit)
    
    dictionary = pickle.load(open("preprocess/dictionary/platform_dict.pkl", 'rb'))   
    dict_msg, dict_code = dictionary

    pad_msg = padding_data(data=[commit_message], dictionary=dict_msg, params=params, type='msg')        
    pad_code = padding_data(data=[code], dictionary=dict_code, params=params, type='code')

    print(pad_msg.shape)
    print(pad_code.shape)

    # Using Pytorch Dataset and DataLoader
    code = {
        "code": pad_code.tolist(),
        "message": pad_msg.tolist()
    }
    
    return (code, dict_msg, dict_code)