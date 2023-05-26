import pickle
from preprocess.cc2vec.padding import *
from utils import hunks_to_code

def cc2vec_preprocess(commit_info, params):
    # Extract commit message
    commit_message = commit_info['commit_message']
    commit = commit_info['main_language_file_changes']

    # DeepJIT
    code = hunks_to_code(commit)
    
    dictionary = pickle.load(open("preprocess/cc2vec/dictionary/qt_dict.pkl", 'rb'))   
    dict_msg, dict_code = dictionary

    pad_msg = padding_data(data=[commit_message], dictionary=dict_msg, params=params, type='msg')        
    pad_code = padding_data(data=[code], dictionary=dict_code, params=params, type='code')

    # CC2Vec
    added_code, removed_code = clean_and_reformat_code(commit)
    pad_added_code = padding_commit_code(data=added_code, max_file=params['code_file'], max_line=params['code_line'], max_length=params['cc2vec_code_length'])
    pad_removed_code = padding_commit_code(data=removed_code, max_file=params['code_file'], max_line=params['code_line'], max_length=params['cc2vec_code_length'])
    pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
    pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)

    # Using Pytorch Dataset and DataLoader
    code = {
        "code": pad_code.tolist(),
        "message": pad_msg.tolist(),
        "added_code": pad_added_code.tolist(),
        "removed_code": pad_removed_code.tolist()
    }
    
    return (code, dict_msg, dict_code)