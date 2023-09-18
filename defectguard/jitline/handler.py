from defectguard.BaseHandler import BaseHandler
from .model import JITLineModel
from defectguard.utils.utils import download_folder, SRC_PATH

class JITLine(BaseHandler):
    def __init__(self, version='platform_within', device="cpu"):
        self.model_name = 'jitline'
        self.version = version
        self.initialized = False
        self.model = None
        self.device = device
        download_folder(self.model_name, self.version)
        
    def initialize(self):
        self.model = JITLineModel(load_path=f"{SRC_PATH}/models/{self.model_name}/{self.version}")

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