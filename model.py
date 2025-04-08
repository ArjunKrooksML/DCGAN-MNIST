import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels, features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features, features * 2),
            self._block(features * 2, features * 4),
            nn.Conv2d(features * 4, 1, kernel_size=3, stride=1, padding=0),
        )

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x).view(-1)

class Generator(nn.Module):
    def __init__(self, z_dim, channels, features):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            self._block(z_dim, features * 4),
            self._block(features * 4, features * 2),
            nn.ConvTranspose2d(features * 2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)