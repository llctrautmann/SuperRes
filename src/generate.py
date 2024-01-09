from train import ModelTrainer
from model import SuperResolution
from dataset import Contrastive_Dataset
from hyperparams import hyperparams

def generate_images():
    trainer = ModelTrainer(model=SuperResolution(),
                           dataset=Contrastive_Dataset(hyperparams.data_dir),
                           device=hyperparams.device,
                           epochs=hyperparams.epochs,
                           lr=hyperparams.lr,
                           batch_size=hyperparams.batch_size,
                           num_workers=hyperparams.num_workers,
                           pin_memory=hyperparams.pin_memory,
                           save_model=hyperparams.save_model,
                           debug_mode=False)
    trainer.load(hyperparams.weights)
    trainer.generate_images()