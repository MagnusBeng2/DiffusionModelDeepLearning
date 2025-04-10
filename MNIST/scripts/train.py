import os
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import argparse
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import multiprocessing

from model import UNet  # ⬅️ Import your model from model.py
from evaluate import evaluate_model  # ⬅️ FID evaluation during training

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

# === Dataset ===
class MNISTFromNPZ(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.images = torch.tensor(data['images'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# === EMA ===
def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        msd = model.state_dict()
        for key, param in ema_model.state_dict().items():
            if key in msd:
                param.copy_(decay * param + (1.0 - decay) * msd[key])

# === Loss Curve ===
def save_loss_curve(losses, timesteps):
    os.makedirs("results", exist_ok=True)
    idx = len([f for f in os.listdir("results") if f.startswith("loss_curve")]) + 1
    path = f"results/loss_curve{idx}.png"

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(f"Loss Curve - Timesteps: {timesteps}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Integer ticks only
    plt.savefig(path)
    plt.close()
    print(f"Saved loss curve to {path}")

# === Training ===
def train_ddpm(batch_size, epochs, learning_rate, timesteps, beta_schedule, enable_cfg=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))
    
    dataset = MNISTFromNPZ("data/processed/mnist_train.npz")
    num_workers = min(8, multiprocessing.cpu_count())
    print(f"Using {num_workers} DataLoader workers.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("Starting training...")
    print("Warming up DataLoader...")
    start_time = time.time()
    for i, (images, _) in enumerate(dataloader):
        print(f"First batch loaded in: {time.time() - start_time:.2f} seconds")
        break

    model = UNet().to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Beta schedule
    beta_t = torch.linspace(0.0001, 0.02, timesteps).to(device) if beta_schedule == "linear" else torch.tensor([
        math.cos((t / timesteps) * (math.pi / 2)) ** 2 for t in range(timesteps)], device=device)
    alpha_bar_t = torch.cumprod(1.0 - beta_t, dim=0)

    losses = []
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # Classifier-free guidance: randomly drop labels
            if enable_cfg and torch.rand(1).item() < 0.1:
                labels = None

            t = torch.randint(0, timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            alpha_bar = alpha_bar_t[t].view(-1, 1, 1, 1)
            noisy_images = torch.sqrt(alpha_bar) * images + torch.sqrt(1 - alpha_bar) * noise

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                predicted_noise = model(noisy_images, t, labels)
                loss = criterion(predicted_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"models/ddpm_mnist_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

            ema_path = f"models/ema_ddpm_mnist_epoch_{epoch+1}.pth"
            torch.save(ema_model.state_dict(), ema_path)
            fid_info = evaluate_model(
                model_path=ema_path,
                num_samples=10000,
                timesteps=1000,
                use_random_noise=True,
                save_results=False
            )
            print(f"[Epoch {epoch+1}] FID = {fid_info['FID']:.2f}, Inception = {fid_info['Inception Score'][0]:.2f} ± {fid_info['Inception Score'][1]:.2f}")

    save_loss_curve(losses, timesteps)
    torch.save(model.state_dict(), "models/ddpm_mnist.pth")
    torch.save(ema_model.state_dict(), f"models/ema_ddpm_mnist_epoch_{epoch}.pth")

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--enable_cfg", action="store_true", help="Enable classifier-free guidance")
    parser.add_argument("--beta_schedule", type=str, default="linear")
    args = parser.parse_args()

    train_ddpm(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        enable_cfg=args.enable_cfg
    )
