import torch
import torch.nn as nn
import torch.distributions as td

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((-1, *self.shape))
    
    
class GAN(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.d = latent_dim
        
        Activation = nn.LeakyReLU
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            Activation(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            Activation(),
            nn.Flatten(),
            nn.Linear(64*6*6, 512),
            Activation(),
            nn.Linear(512, 1)
        )

        Activation = nn.LeakyReLU
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 512),
            Activation(),
            nn.Linear(512, 64*6*6),
            Activation(),
            Reshape(64, 6, 6),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            Activation(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, output_padding=1),
            nn.Sigmoid(),
        )

        self.prior = td.Independent(
            td.Normal(loc=torch.zeros(latent_dim),
                      scale=torch.ones(latent_dim) * 1.),
            reinterpreted_batch_ndims=1
        )
        
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.prior.base_dist.loc = self.prior.base_dist.loc.to(*args, **kwargs)
        self.prior.base_dist.scale = self.prior.base_dist.scale.to(*args, **kwargs)
        
    
    def sample(self, n=1):
        z = self.prior.sample((n, ))
        return self.generator(z)

    
    def forward(self, x):
        raise NotImplemented



class GAN_FC(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.d = latent_dim
        
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.LeakyReLU(0.2), # nn.Dropout(0.3),
            nn.Linear(256, 256), nn.LeakyReLU(0.2), # nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 784), nn.Tanh(),
            Reshape(1, 28, 28)
        )

        self.prior = td.Independent(
            td.Normal(loc=torch.zeros(latent_dim),
                      scale=torch.ones(latent_dim) * 1.),
            reinterpreted_batch_ndims=1
        )
        
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.prior.base_dist.loc = self.prior.base_dist.loc.to(*args, **kwargs)
        self.prior.base_dist.scale = self.prior.base_dist.scale.to(*args, **kwargs)
        
    
    def sample(self, n=1):
        z = self.prior.sample((n, ))
        return self.generator(z)

    
    def forward(self, x):
        raise NotImplemented