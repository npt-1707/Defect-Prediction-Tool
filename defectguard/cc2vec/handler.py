from defectguard.BaseHandler import BaseHandler
import pickle, json, torch
from defectguard.cc2vec.model import HierachicalRNN, DeepJITExtended
from defectguard.utils.utils import download_folder, SRC_PATH

class CC2Vec(BaseHandler):
    def __init__(self, version='qt_within', dictionary='qt', device="cpu"):
        self.model_name = 'cc2vec'
        self.version = version
        self.dictionary = dictionary
        self.initialized = False
        self.cc2vec = None
        self.deepjit_extended = None
        self.device = device
        download_folder(self.model_name, self.version, self.dictionary)
        
    def initialize(self):
        # Load dictionary
        dictionary = pickle.load(open(f"{SRC_PATH}/models/{self.model_name}/{self.dictionary}_dictionary", 'rb'))
        dict_msg, dict_code = dictionary

        # Load parameters
        with open(f"{SRC_PATH}/models/{self.model_name}/hyperparameters", 'r') as file:
            params = json.load(file)

        # Set up param
        params["filter_sizes"] = [int(k) for k in params["filter_sizes"].split(',')]
        params["vocab_msg"], params["vocab_code"] = len(dict_msg), len(dict_code)
        params["cc2vec_class_num"] = len(dict_msg)
        params["deepjit_class_num"] = 1
        params["embedding_feature"] = params['embedding_size'] * 3 + 2 + 2

        # Initialize model
        self.cc2vec = HierachicalRNN(params).to(device=self.device)
        self.cc2vec.load_state_dict(torch.load(f"{SRC_PATH}/models/{self.model_name}/cc2vec_{self.version}", map_location=self.device))

        self.deepjit_extended = DeepJITExtended(params).to(device=self.device)
        self.deepjit_extended.load_state_dict(torch.load(f"{SRC_PATH}/models/{self.model_name}/dextended_{self.version}", map_location=self.device))

        # Set initialized to True
        self.initialized = True

    def preprocess(self, data):
        if not self.initialized:
            self.initialize()
        print("Preprocessing...")

    def inference(self, model_input):
        if not self.initialized:
            self.initialize()
        print("Inferencing...")

    def postprocess(self, inference_output):
        if not self.initialized:
            self.initialize()
        print("Postprocessing...")

    def handle(self, data):
        if not self.initialized:
            self.initialize()
        print("Handling...")
        preprocessed_data = self.preprocess(data)
        model_output = self.inference(preprocessed_data)
        final_prediction = self.postprocess(model_output)