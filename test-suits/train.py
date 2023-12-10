from defectguard import DeepJIT
import pickle, torch
from padding import padding_data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class CustomDataset(Dataset):
    def __init__(self, ids, code, message, labels):
        self.ids = ids
        self.code = code
        self.message = message
        self.labels = labels
    
    def __len__(self):
        return len(self.code)
    
    def __getitem__(self, idx):
        commit_hash = self.ids[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        code = self.code[idx]
        message = self.message[idx]
        code = torch.tensor(code)
        message = torch.tensor(message)

        return {
            'commit_hash': commit_hash,
            'code': code,
            'message': message,
            'labels': labels
        }

with open('/home/manh/Documents/DefectGuard/Data/bootstrap_part_5_dextend.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

ids, messages, commits, labels = loaded_data

dictionary = pickle.load(open('/home/manh/Documents/DefectGuard/Data/bootstrap_part_5_dict.pkl', 'rb'))   
dict_msg, dict_code = dictionary

# ---------------------- DefectGuard -------------------------------
model = DeepJIT(device='cuda')
model.initialize()
# ------------------------------------------------------------------

pad_msg = padding_data(data=messages, dictionary=dict_msg, params=model.parameters, type='msg')        
pad_code = padding_data(data=commits, dictionary=dict_code, params=model.parameters, type='code')

code_dataset = CustomDataset(ids, pad_code, pad_msg, labels)
code_dataloader = DataLoader(code_dataset, batch_size=model.parameters['batch_size'])

optimizer = torch.optim.Adam(model.model.parameters(), lr=5e-5)
criterion = nn.BCELoss()

for epoch in range(1, 10 + 1):
    total_loss = 0
    for batch in code_dataloader:
        # Extract data from DataLoader
        code = batch["code"].to(model.device)
        message = batch["message"].to(model.device)
        labels = batch["labels"].to(model.device)
        
        optimizer.zero_grad()

        # ---------------------- DefectGuard -------------------------------
        predict = model(message, code)
        # ------------------------------------------------------------------
        
        loss = criterion(predict, labels)

        loss.backward()

        total_loss += loss

        optimizer.step()

    print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, 10, total_loss))