
from hyperparams import hyperparams
from dataset import Contrastive_Dataset
from model import SuperResolution
import torch

class Inference:
    def __init__(self, model, device, load_model, model_path):
        self.model = model
        self.device = device
        self.load_model = load_model
        self.model_path = model_path
        self.load()

    def load(self):
        if self.load_model:
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            print("Model loaded successfully")
        else:
            print("No model loaded")
    
    def infer(self, img):
        assert img.shape[1:] == (3, 64, 64)
        with torch.no_grad():
            img = img.to(self.device)
            sr_img = self.model(img)
        return sr_img
