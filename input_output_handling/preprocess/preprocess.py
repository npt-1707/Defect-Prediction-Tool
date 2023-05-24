import pickle, torch
from torch.utils.data import Dataset, DataLoader
from padding import padding_data

class DeepJITDataset(Dataset):
    def __init__(self, code, message):
        self.code = code
        self.message = message
    
    def __len__(self):
        return len(self.code)
    
    def __getitem__(self, idx):
        code = self.code[idx]
        message = self.message[idx]
        code = torch.tensor(code)
        message = torch.tensor(message)

        return {
            'code': code,
            'message': message
        }

def deep_preprocess(commit_info, params):
    # Extract commit message
    commit_message = commit_info['commit_message']
    commit = commit_info['main_language_file_changes']
    
    dictionary = pickle.load(open("params.predict_data", 'rb'))   
    dict_msg, dict_code = dictionary

    pad_msg = padding_data(data=commit_message, dictionary=dict_msg, params=params, type='msg')        
    pad_code = padding_data(data=commit, dictionary=dict_code, params=params, type='code')

    # Using Pytorch Dataset and DataLoader
    code_dataset = DeepJITDataset(pad_code, pad_msg)
    code_dataloader = DataLoader(code_dataset, batch_size=params["batch_size"])

    return (code_dataloader, dict_msg, dict_code)