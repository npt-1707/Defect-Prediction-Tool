from defectguard.models.BaseHandler import BaseHandler
import pickle, json, torch
from .model import DeepJITModel
from defectguard.utils.utils import download_folder, SRC_PATH
from .utils import *

class DeepJIT(BaseHandler):
    def __init__(self, dataset='platform', project='within', device="cpu"):
        self.model_name = 'deepjit'
        self.dataset = dataset
        self.project = project
        self.initialized = False
        self.model = None
        self.device = device
        self.message_dictionary = None
        self.code_dictionary = None
        self.parameters = None
        download_folder(self.model_name, self.dataset, self.project)

    def __call__(self, message, code):
        return self.model(message, code)
    
    def initialize(self):
        # Load dictionary
        dictionary = pickle.load(open(f"{SRC_PATH}/models/metadata/{self.model_name}/{self.dataset}_dictionary_{self.project}", 'rb'))
        self.message_dictionary, self.code_dictionary = dictionary

        # Load parameters
        with open(f"{SRC_PATH}/models/metadata/{self.model_name}/hyperparameters", 'r') as file:
            self.parameters = json.load(file)

        # Set up param
        self.parameters["filter_sizes"] = [int(k) for k in self.parameters["filter_sizes"].split(',')]
        self.parameters["vocab_msg"], self.parameters["vocab_code"] = len(self.message_dictionary), len(self.code_dictionary)
        self.parameters["class_num"] = 1

        # Create model and Load pretrain
        self.model = DeepJITModel(self.parameters).to(device=self.device)
        self.model.load_state_dict(torch.load(f"{SRC_PATH}/models/metadata/{self.model_name}/{self.dataset}_{self.project}", map_location=self.device))

        # Set initialized to True
        self.initialized = True

    def preprocess(self, data):
        if not self.initialized:
            self.initialize()

        commit_info = data['commit_info']
        commit_hashes = []
        commit_messages = []
        codes = []

        for commit in commit_info:
            commit_hashes.append(commit['commit_hash'])

            # Extract commit message
            commit_message = commit['commit_message'].strip()
            commit_message = split_sentence(commit_message)
            commit_message = ' '.join(commit_message.split(' ')).lower()
            
            commit = commit['main_language_file_changes']

            code = hunks_to_code(commit)

            commit_messages.append(commit_message)
            codes.append(code)

        pad_msg = padding_data(data=commit_messages, dictionary=self.message_dictionary, params=self.parameters, type='msg')        
        pad_code = padding_data(data=codes, dictionary=self.code_dictionary, params=self.parameters, type='code')

        # Using Pytorch Dataset and DataLoader
        code = {
            "code": pad_code.tolist(),
            "message": pad_msg.tolist()
        }
        
        return commit_hashes, code

    def inference(self, model_input):
        if not self.initialized:
            self.initialize()

        # Forward
        self.model.eval()
        with torch.no_grad():
            # Extract data from DataLoader
            code = torch.tensor(model_input["code"], device=self.device)
            message = torch.tensor(model_input["message"], device=self.device)

            # Forward
            predict = self.model(message, code)
        
        return predict

    def postprocess(self, commit_hashes, inference_output):
        if not self.initialized:
            self.initialize()

        inference_output = inference_output.tolist()

        return [{'commit_hash': commit_hashes[i], 'predict': inference_output[i]} for i in range(len(commit_hashes))]

    def handle(self, data):
        if not self.initialized:
            self.initialize()
            
        commit_hashes, preprocessed_data = self.preprocess(data)
        model_output = self.inference(preprocessed_data)
        final_prediction = self.postprocess(commit_hashes, model_output)

        return final_prediction