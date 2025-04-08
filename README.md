# MNIST DCGAN PyTorch

A simple implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate handwritten digits based on the MNIST dataset.

## File Structure

* `config.py`: Contains hyperparameters and configuration settings.
* `model.py`: Defines the Generator and Discriminator network architectures (DCGAN).
* `utils.py`: Contains utility functions (e.g., weight initialization).
* `train.py`: The main script to train the GAN model.
* `Output/`: Directory where generated image samples and final model weights (`.pth`) are saved.
* `dataset/`: Directory where the MNIST dataset is automatically downloaded.

## Requirements

* Python 3.x
* PyTorch
* Torchvision

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```
    *(Replace `<your-repository-url>` and `<repository-directory-name>`)*

2.  **Install dependencies:**
    (It's recommended to use a Python virtual environment)
    ```bash
    pip install torch torchvision
    ```

## How to Run

1.  Execute the main training script from the project's root directory:
    ```bash
    python train.py
    ```

2.  **Output:**
    * The script will automatically download the MNIST dataset to the `dataset/` folder if it's not found.
    * Training progress (Epoch, Batch, Discriminator Loss, Generator Loss) will be printed to the console.
    * After each epoch, a sample grid of generated digits (e.g., `epoch_001.png`) will be saved in the `Output/` directory.
    * Upon completion, the final trained Generator (`mnist_dcgan_generator.pth`) and Discriminator (`mnist_dcgan_discriminator.pth`) model weights will be saved in the `Output/` directory.
