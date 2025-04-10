import os
import numpy as np
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
BATCH_SIZE = 512  # doesn't matter too much here

def normalize_and_reshape(images):
    # Scale [0, 255] -> [-1, 1]
    images = (images.astype(np.float32) / 127.5) - 1.0
    return images.transpose(0, 3, 1, 2)  # NHWC -> NCHW

def save_npz(images, labels, split_name):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_DIR, f"cifar_{split_name}.npz")
    np.savez_compressed(path, images=images, labels=labels)
    print(f"Saved {split_name} set to {path}")

def process_cifar(subset_ratio=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
    ])

    train_dataset = datasets.CIFAR10(root=RAW_DATA_DIR, train=True, transform=transform, download=False)
    test_dataset = datasets.CIFAR10(root=RAW_DATA_DIR, train=False, transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    train_images = train_images.mul(255).byte().numpy().transpose(0, 2, 3, 1)  # convert to uint8 NHWC
    test_images = test_images.mul(255).byte().numpy().transpose(0, 2, 3, 1)

    if subset_ratio < 1.0:
        n_train = int(len(train_images) * subset_ratio)
        n_test = int(len(test_images) * subset_ratio)
        idx_train = np.random.choice(len(train_images), n_train, replace=False)
        idx_test = np.random.choice(len(test_images), n_test, replace=False)
        train_images, train_labels = train_images[idx_train], train_labels[idx_train]
        test_images, test_labels = test_images[idx_test], test_labels[idx_test]

    train_images = normalize_and_reshape(train_images)
    test_images = normalize_and_reshape(test_images)

    save_npz(train_images, train_labels.numpy(), "train")
    save_npz(test_images, test_labels.numpy(), "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=1.0, help="Subset ratio (e.g., 0.1 for 10%)")
    args = parser.parse_args()
    process_cifar(args.subset)
