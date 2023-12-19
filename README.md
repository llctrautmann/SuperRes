# Super Resolution Model

This repository contains a Python-based implementation of a Super Resolution model using PyTorch. The model is trained on a dataset of images, with the aim of enhancing the resolution of low-quality images. The project is organized into several Python scripts. Jupyter notebooks were used for development.

## Codebase Structure


- `notebooks/train.ipynb`: Contains the main training loop for the Super Resolution model. It sets up the model, optimizer, and loss function, and then trains the model over a specified number of epochs. It also includes code for logging and visualization using TensorBoard.

- `notebooks/dataset.ipynb`: Defines the Contrastive_Dataset class, which is a custom PyTorch Dataset for loading and preprocessing the image data.

- `notebooks/model.ipynb`: Defines the architecture of the Super Resolution model. It includes a DenseBlock class for the dense blocks in the model, and a SuperResolution class for the overall model.

- `src/train.py`: Contains the ModelTrainer class, which encapsulates the training loop and provides methods for training and testing the model.

- `src/dataset.py`: Contains the Contrastive_Dataset class, similar to the one in `notebooks/dataset.ipynb`.

- `src/model.py`: Contains the SuperResolution class, similar to the one in `notebooks/model.ipynb`.

- `src/main.py`: The main entry point for training the model. It creates an instance of ModelTrainer and runs the training process.

- `src/hyperparams.py`: Defines a HyperParams class that holds all the hyperparameters for the model and training process.

- `.gitignore`: Specifies which files and directories should be ignored by Git.

## Usage

To train the model, run the `main.py` script:

This project requires the following Python libraries:

- PyTorch
- torchvision
- numpy
- tqdm
- PIL
- matplotlib

## Note
The image data used for training the model is not included in the repository and should be placed in the data/cryptopunks/ directory. The images are expected to be in JPEG format and need to be compressed into a single channel image that will be converted back into RGB in the dataloading process. 
