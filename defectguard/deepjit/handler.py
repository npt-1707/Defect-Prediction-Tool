from defectguard.BaseHandler import BaseHandler
from defectguard.download import download
import dvc.api
from defectguard.config.config import CONFIG

class DeepJITHandler(BaseHandler):
    def __init__(self):
        self.model_pt = dvc.api.read(
            'models/deepjit/deepjit.pt',
            repo="https://github.com/manhlamabc123/DefectGuard",
            mode='rb',
            config=CONFIG
        )
        self.model_params = dvc.api.read(
            'models/deepjit/deepjit.json',
            repo="https://github.com/manhlamabc123/DefectGuard",
            mode='rb',
            config=CONFIG
        )
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        
    def initialize(self):
        # Download model's meta data
        pass

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