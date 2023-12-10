from defectguard.models.BaseWraper import BaseWraper
import pickle
from defectguard.utils.utils import download_folder, SRC_PATH

class LAPredict(BaseWraper):
    def __init__(self, dataset='platform', project='within', device="cpu"):
        self.model_name = 'lapredict'
        self.dataset = dataset
        self.project = project
        self.initialized = False
        self.model = None
        self.device = device
        download_folder(self.model_name, self.dataset)
        
    def initialize(self):
        with open(f"{SRC_PATH}/models/metadata/{self.model_name}/{self.dataset}_{self.project}", "rb") as f:
            self.model = pickle.load(f)

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