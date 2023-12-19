import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class Contrastive_Dataset(Dataset):
    # TODO: Add docstring
    def __init__(self, root_dir):
        self.root_dir_lq = root_dir
        self.transform_hq = transforms.Compose([
            transforms.Resize((256, 256)),
            # TODO: Add optional normalization
            transforms.ToTensor(),
        ])

        self.transform_lq = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.images = glob.glob(root_dir + "/*.jpg")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        image = Image.open(self.images[idx])
        # Convert to RGB
        image = image.convert('RGB')
        # Transform image
        hq_image = self.transform_hq(image)
        lq_image = self.transform_lq(image)

        return hq_image, lq_image, self.images[idx].split("/")[-1]
