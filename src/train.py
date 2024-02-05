
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from hyperparams import hyperparams
from tqdm import tqdm


class ModelTrainer:
    def __init__(self,
                 model,
                 dataset,
                 device,
                 epochs,
                 lr,
                 batch_size,
                 num_workers,
                 pin_memory=True,
                 save_model=True,
                 debug_mode=False):

        self.model = model
        self.dataset = dataset
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_loader = None
        self.test_loader = None
        self.criterion = nn.MSELoss(reduction="sum").to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.min_loss = np.inf
        self.save_model = save_model
        self.debug_mode = debug_mode
        self.train_writer = SummaryWriter("./runs/sr_256x256/train/")
        self.test_writer = SummaryWriter("./runs/sr_256x256/test/")

        # Functions to run on init
        self.create_dataloader()

    def create_dataloader(self):
        if self.debug_mode:
            self.dataset = torch.utils.data.Subset(
                self.dataset, list(range(100)))
            print(self.dataset)
        train_set, test_set = random_split(self.dataset, [0.9, 0.1])
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )
        print("DataLoaders created successfully")

    def train(self, epoch):
        self.model.train()
        for idx, batch in tqdm(enumerate(self.train_loader), disable=not self.debug_mode):
            img1, img2, label = batch
            img1, img2 = img1.to(self.device), img2.to(self.device)

            sr_image = self.model(img2)
            loss = self.criterion(sr_image, img1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Add epoch loss
        self.train_writer.add_scalar('Loss/train', loss.item(), epoch)

    def test(self, epoch):
        with torch.no_grad():
            self.model.eval()

            for idx, batch in enumerate(self.test_loader):
                img1, img2, label = batch
                img1, img2 = img1.to(self.device), img2.to(self.device)
                sr_image = self.model(img2)
                loss = self.criterion(sr_image, img1)
            self.test_writer.add_scalar('Loss/test', loss.item(), epoch)

            # Add images to tensorboard
            # Convert images to grid format
            img2_resized = F.interpolate(
                img2[:4, ...], size=sr_image.shape[2:])
            img1_grid = torchvision.utils.make_grid(img1[:4, ...], padding=5)
            img2_grid = torchvision.utils.make_grid(img2_resized, padding=5)
            sr_image_grid = torchvision.utils.make_grid(
                sr_image[:4, ...], padding=5)

            self.test_writer.add_image('Original HQ Images', img1_grid, epoch)
            self.test_writer.add_image('Original LQ Images', img2_grid, epoch)
            self.test_writer.add_image(
                'Super Resolved Images', sr_image_grid, epoch)

        return loss.item()

    def run(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            self.train(epoch=epoch)
            test_loss = self.test(epoch=epoch)

            if self.save_model:
                self.save(test_loss, epoch)

            print(f"Epoch: {epoch} | Test Loss: {test_loss}")

    def save(self, loss, epoch):
        if self.min_loss > loss:
            print(f"Saving model at epoch {epoch}")
            torch.save(self.model.state_dict(), os.path.join(
                hyperparams.save_path, "model.pth"))
            self.min_loss = loss

    def load(self, weights):
        self.model.load_state_dict(torch.load(weights))
        self.model.to(self.device)

    def generate_images(self, weights):
        self.load(weights)
        self.model.eval()

        # TODO: ADD Inference Loop to generate new images.
