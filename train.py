import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Unet
from diffuse import forward_diffusion_sample, sample_timestep


epochs = 10000
batch_size = 12
image_size = 256
root = "raw"
lr = 2e-4

T = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transformed_dataset():
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    data = ImageFolder(root, transform=transform)
    return data


def show_tensor_image(tensor_image):
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.0),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])
    if len(tensor_image.shape) == 4:
        tensor_image = tensor_image[0, :, :, :]
    plt.imshow(transform(tensor_image))


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_image(model):
    sample = torch.randn((1, 3, image_size, image_size), device=device)
    for i in tqdm(range(0, T)[::-1]):
        t = torch.full((1, ), i, device=device, dtype=torch.long)
        sample = sample_timestep(sample, t, model)
    return sample


dataset = load_transformed_dataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def Train():
    unet = Unet(3, 32).to(device)
    optim = Adam(unet.parameters(), lr=lr)

    steps = 0
    print("Start training...")
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            data = batch[0].to(device)

            unet.zero_grad()
            t = torch.randint(0, T, (batch_size, ), device=device).long()
            loss = get_loss(unet, data, t)
            loss.backward()
            optim.step()

            print(f"[{epoch}/{epochs}][{step}/{len(dataloader)}]  loss: {loss.item()}")
            steps += 1
            if steps % 50 == 0:
                print("Sampling...")
                sample = sample_image(unet)
                save_image(sample, f"samples/{steps}.png", "PNG")
                torch.save(unet, f"models/unet-{steps}.pth")
                torch.save(optim, f"models/adam-{steps}.pth")


if __name__ == "__main__":
    Train()
