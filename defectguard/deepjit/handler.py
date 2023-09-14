from defectguard.BaseHandler import BaseHandler
from defectguard.download import download

class DeepJITHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        
    def initialize(self):
        # Download model's meta data
        download('deepjit')

        # Init model with meta data

        # Set initialized to True

    def preprocess(self, data):
        pass

    def inference(self, model_input):
        pass

    def postprocess(self, inference_output):
        pass

    def handle(self, data):
        pass