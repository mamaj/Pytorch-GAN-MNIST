from pathlib import Path
from re import X
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import torch.distributions as td

DATAPATH = Path.home() / '.datasets'

def display(x):
    """
    x (Tensor): B, C, W, H
    """
    x = x.detach().cpu()
    x = make_grid(x).permute(1, 2, 0)
    plt.imshow(x)
    plt.axis('off')
    

def load_mnist(binary=False):
    transforms_list = transforms.Compose([
        transforms.ToTensor(),
        (lambda x: (x > 0.5).to(x.dtype)) if binary else transforms.Normalize(0.5, 0.5)
        ])
    
    train_ds = datasets.MNIST(
        root=DATAPATH,
        train=True,
        download=True,
        transform=transforms_list
        )
    
    test_ds = datasets.MNIST(
        root=DATAPATH,
        train=False,
        download=True,
        transform=transforms_list
        )
    
    return train_ds, test_ds
        


def plot_latent_images(model, n, digit_size=28, ax=None):
    """Plots n x n digit images decoded from the latent space."""
    
    norm = td.Normal(0, 1)
    grid_x = norm.icdf(torch.linspace(0.05, 0.95, n))
    grid_y = norm.icdf(torch.linspace(0.05, 0.95, n))
    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = torch.tensor([[xi, yi]])
            x = model.generator(z)
            digit = torch.reshape(x[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(image, cmap='Greys_r')
    ax.axis('Off')
    return im