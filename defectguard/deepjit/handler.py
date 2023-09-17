from defectguard.BaseHandler import BaseHandler
import pickle, json, torch, io
from defectguard.deepjit.model import DeepJITModel
from defectguard.utils.utils import load_metadata

class DeepJIT(BaseHandler):
    def __init__(self, model='deepjit', version='platform_within', dictionary='platform', device="cpu"):
        self.initialized = False
        # self.model_pt = dvc.api.read(
        #     f'models/deepjit/{model}.pt',
        #     repo="https://github.com/manhlamabc123/DefectGuard",
        #     mode='rb',
        #     config=CONFIG
        # )
        # self.model_params = dvc.api.read(
        #     'models/deepjit/deepjit.json',
        #     repo="https://github.com/manhlamabc123/DefectGuard",
        #     mode='r',
        #     config=CONFIG
        # )
        # self.dictionary = dvc.api.read(
        #     f'models/deepjit/{dictionary}_dict.pkl',
        #     repo="https://github.com/manhlamabc123/DefectGuard",
        #     mode='rb',
        #     config=CONFIG
        # )
        self.model_pt, self.model_params, self.dictionary = load_metadata(model, version, dictionary)
        self.model = None
        self.device = device
        
    def initialize(self):
        # Load dictionary
        dictionary = pickle.loads(self.dictionary)   
        dict_msg, dict_code = dictionary

        # Load parameters
        params = json.loads(self.model_params)

        # Set up param
        params["filter_sizes"] = [int(k) for k in params["filter_sizes"].split(',')]
        params["vocab_msg"], params["vocab_code"] = len(dict_msg), len(dict_code)
        params["class_num"] = 1

        # Create model and Load pretrain
        self.model = DeepJITModel(params).to(device=self.device)
        self.model.load_state_dict(torch.load(io.BytesIO(self.model_pt), map_location=self.device))

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