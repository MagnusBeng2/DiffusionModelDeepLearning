import os
import re
import math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import numpy as np
import scipy.linalg
import argparse
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
from tqdm import tqdm
from model import UNet
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = "models/ddpm_cifar_final.pth"
TEST_DATA_PATH = "data/processed/cifar_test.npz"
IMAGE_SAVE_DIR = "images"
RESULTS_DIR = "results"

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def cosine_schedule(timesteps):
    s = 0.008
    t = torch.linspace(0, timesteps - 1, timesteps)
    return torch.cos((t / timesteps + s) / (1 + s) * (math.pi / 2)) ** 2

def reverse_diffusion(model, device, noisy_images, timesteps, labels=None, guidance_weight=3.0):
    model.eval()
    beta_t = (1 - cosine_schedule(timesteps)).to(device)
    alpha_t = (1.0 - beta_t).to(device)
    alpha_bar_t = torch.cumprod(alpha_t, dim=0).to(device)

    x_t = noisy_images
    with torch.no_grad():
        for t in tqdm(reversed(range(timesteps)), desc="Denoising images"):
            z = torch.randn_like(x_t) if t > 0 else 0
            alpha = alpha_t[t].view(-1, 1, 1, 1)
            alpha_bar = alpha_bar_t[t].view(-1, 1, 1, 1)
            t_tensor = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)

            pred_cond = model(x_t, t_tensor, labels)
            pred_uncond = model(x_t, t_tensor, None)
            predicted_noise = pred_uncond + guidance_weight * (pred_cond - pred_uncond)

            x_t = (1 / torch.sqrt(alpha)) * (
                x_t - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
            ) + torch.sqrt(beta_t[t]) * z

            x_t = torch.clamp(x_t, -1.0, 1.0)
    return x_t

def get_next_filename():
    files = [f for f in os.listdir(IMAGE_SAVE_DIR) if f.startswith("generated_samples_debug") and f.endswith(".png")]
    numbers = [int(re.match(r"generated_samples_debug(\d+)\.png", f).group(1)) for f in files if re.match(r"generated_samples_debug(\d+)\.png", f)]
    next_number = max(numbers, default=0) + 1
    return os.path.join(IMAGE_SAVE_DIR, f"generated_samples_debug{next_number}.png")

def get_next_evaluation_filename():
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("evaluation_results") and f.endswith(".txt")]
    numbers = [int(re.match(r"evaluation_results(\d+)\.txt", f).group(1)) for f in files if re.match(r"evaluation_results(\d+)\.txt", f)]
    next_number = max(numbers, default=0) + 1
    return os.path.join(RESULTS_DIR, f"evaluation_results{next_number}.txt")

def load_real_images_from_npz(num_samples, device):
    data = np.load(TEST_DATA_PATH)
    images = torch.tensor(data['images'], dtype=torch.float32)
    if len(images) < num_samples:
        raise ValueError(f"Requested {num_samples} samples, but only {len(images)} available.")
    indices = torch.randperm(len(images))[:num_samples]
    return images[indices].to(device)

def evaluate_model(model_path, num_samples, timesteps, use_random_noise=True, save_results=True, batch_size=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU index:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(device))

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    real_images = load_real_images_from_npz(num_samples, device)

    if use_random_noise:
        print("Generating images from pure noise...")
    else:
        print("Using slightly noisy real images...")

    generated_images = []
    for i in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - i)
        if use_random_noise:
            noisy = torch.randn(bs, 3, 32, 32, device=device)
        else:
            noisy = real_images[i:i+bs]

        labels = torch.randint(0, 10, (bs,), device=device)
        samples = reverse_diffusion(model, device, noisy, timesteps, labels=labels, guidance_weight=3.0)
        generated_images.append(samples.cpu())

    generated_images = torch.cat(generated_images, dim=0)

    filename = get_next_filename()
    vutils.save_image(generated_images[:16], filename, normalize=True)
    print(f"Saved generated images to '{filename}'")

    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception.fc = nn.Identity()
    inception.eval().to(device)

    def get_features(images, model, batch_size=256):
        features = []
        model.eval()
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch = (batch + 1) / 2
            batch = batch.repeat(1, 3, 1, 1)

            with torch.no_grad():
                batch = batch.to(device)
                batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
                feats = model(batch).detach().cpu().numpy()

            features.append(feats)
            torch.cuda.empty_cache()

        return np.vstack(features)

    real_features = get_features(real_images, inception)
    gen_features = get_features(generated_images, inception)

    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(0), np.cov(gen_features, rowvar=False)

    sigma_real += np.eye(sigma_real.shape[0]) * 1e-6
    sigma_gen += np.eye(sigma_gen.shape[0]) * 1e-6

    cov_sqrt = scipy.linalg.sqrtm(sigma_real @ sigma_gen)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = np.real_if_close(cov_sqrt, tol=1000)

    fid_score = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)

    def run_inception_in_batches(images, model, device, batch_size=128):
        preds = []
        model.eval()
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size]
            batch = (batch + 1) / 2
            batch = batch.repeat(1, 3, 1, 1)

            with torch.no_grad():
                batch = batch.to(device)
                batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
                logits = model(batch).softmax(dim=-1).cpu().numpy()

            preds.append(logits)
            torch.cuda.empty_cache()

        return np.concatenate(preds, axis=0)

    preds = run_inception_in_batches(generated_images, inception, device)

    splits = 10
    scores = []
    for split in np.array_split(preds, splits):
        kl = split * (np.log(split + 1e-10) - np.log(np.mean(split, axis=0) + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    inception_score = np.mean(scores)
    inception_std = np.std(scores)

    print(f"FID: {fid_score:.2f}")
    print(f"Inception Score: {inception_score:.2f} ± {inception_std:.2f}")

    if save_results:
        eval_file = get_next_evaluation_filename()
        with open(eval_file, "w") as f:
            f.write(f"FID: {fid_score:.2f}\n")
            f.write(f"Inception Score: {inception_score:.2f} ± {inception_std:.2f}\n")
        print(f"Saved evaluation results to '{eval_file}'")

    return {
        "FID": fid_score,
        "Inception Score": (inception_score, inception_std)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPM model.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--use_random_noise", action="store_true")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        num_samples=args.num_samples,
        timesteps=args.timesteps,
        use_random_noise=args.use_random_noise
    )
