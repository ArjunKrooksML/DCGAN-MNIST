import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import time

# Import configurations and models
import config
from model import Discriminator, Generator
from utils import initialize_weights

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * config.CHANNELS, [0.5] * config.CHANNELS),
])

# Load MNIST dataset
dataset = datasets.MNIST(root=config.DATASET_DIR, train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Initialize models and optimizers
generator = Generator(config.Z_DIM, config.CHANNELS, config.GEN_FEATURES).to(config.DEVICE)
discriminator = Discriminator(config.CHANNELS, config.DISC_FEATURES).to(config.DEVICE)

initialize_weights(generator)
initialize_weights(discriminator)

opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE)
opt_disc = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE)

criterion = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(64, config.Z_DIM, 1, 1).to(config.DEVICE)

# Training loop
generator.train()
discriminator.train()

print("Starting Training...")
start_time = time.time()

for epoch in range(config.EPOCHS):
    epoch_start_time = time.time()
    real_label = 1.0
    fake_label = 0.0

    for batch_idx, (real_images, _) in enumerate(loader):
        real_images = real_images.to(config.DEVICE)
        batch_size = real_images.size(0)

        # Train Discriminator
        discriminator.zero_grad()
        real_labels = torch.full((batch_size,), real_label, device=config.DEVICE)
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
        fake_images = generator(noise)
        fake_labels = torch.full((batch_size,), fake_label, device=config.DEVICE)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)

        loss_disc = loss_real + loss_fake
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        generator.zero_grad()
        output_fake_for_gen = discriminator(fake_images)
        loss_gen = criterion(output_fake_for_gen, real_labels)
        loss_gen.backward()
        opt_gen.step()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] Batch {batch_idx+1}/{len(loader)} | Loss D: {loss_disc.item():.4f} | Loss G: {loss_gen.item():.4f}")

    with torch.no_grad():
        generator.eval()
        fake_samples = generator(fixed_noise)
        fake_samples_rescaled = (fake_samples * 0.5) + 0.5
        save_image(fake_samples_rescaled, f"{config.OUTPUT_DIR}/epoch_{epoch+1:03d}.png", normalize=False)
        generator.train()

    epoch_time = time.time() - epoch_start_time
    print(f"----> Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")

total_time = time.time() - start_time
print(f"Training finished in {total_time:.2f} seconds.")
print(f"Generated images saved in '{config.OUTPUT_DIR}' directory.")

# Save models
torch.save(generator.state_dict(), os.path.join(config.OUTPUT_DIR, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(config.OUTPUT_DIR, "discriminator.pth"))
print("Models saved.")