from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from celluloid import Camera
from torch.distributions.kl import kl_divergence as kl
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange

from gan import GAN, GAN_FC
from utils import display, load_mnist, plot_latent_images

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparams
latent_dim = 128
batch_size = 64
epochs = 100
gen_iters = 1
lr = 0.0002
betas=(0.5, 0.999)


mnist_train, mnist_test = load_mnist()
train_loader = DataLoader(mnist_train, shuffle=True, batch_size=batch_size, drop_last=True)
# test_loader = DataLoader(mnist_test, shuffle=False, batch_size=64)


# model = GAN(latent_dim=latent_dim)
model = GAN_FC(latent_dim=latent_dim)
model.to(DEVICE)


gen_optim = torch.optim.Adam(lr=lr, params=model.generator.parameters(), betas=betas)
disc_optim = torch.optim.Adam(lr=lr, params=model.discriminator.parameters(), betas=betas)


disc_loss_list, gen_loss_list = [], []

for _ in trange(epochs):
    
    for x_real, _ in tqdm(train_loader):
        x_real = x_real.to(DEVICE)

        x_fake = model.sample(batch_size)
        
        x = torch.cat((x_real, x_fake))
        y = torch.cat((
            torch.ones((batch_size, 1)),
            torch.zeros((batch_size, 1))
            ))
        
        disc_logit = model.discriminator(x)
        disc_loss = F.binary_cross_entropy_with_logits(disc_logit, y)
    
        disc_optim.zero_grad()
        disc_loss.backward()
        disc_optim.step()
        disc_loss_list.append(disc_loss.item())
        
        # train the generator
        for i in range(gen_iters):
            x_fake = model.sample(batch_size)
            
            disc_logit = model.discriminator(x_fake)
            # gen_loss = F.logsigmoid(-1 * disc_logit).mean() # 1-sigmoid(l) = sigmoid(-l)
            # gen_loss = - F.logsigmoid(disc_logit).mean() 
            gen_loss = F.binary_cross_entropy_with_logits(disc_logit, torch.ones((batch_size, 1)))
            
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()
            gen_loss_list.append(gen_loss.item())
        



# display samples
display(model.sample(32))

# display loss
fig, ax = plt.subplots()
ax.plot(disc_loss_list, 'r', label='disc')
ax2 = ax.twiny()
ax2.plot(gen_loss_list, 'b', label='gen')
fig.legend(loc='right')