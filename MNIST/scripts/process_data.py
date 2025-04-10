import os
import struct
import numpy as np
import argparse

# Define file paths
RAW_DATA_DIR = "data/raw/MNIST/raw"
PROCESSED_DATA_DIR = "data/processed/"

TRAIN_IMAGES_FILE = os.path.join(RAW_DATA_DIR, "train-images-idx3-ubyte")
TRAIN_LABELS_FILE = os.path.join(RAW_DATA_DIR, "train-labels-idx1-ubyte")
TEST_IMAGES_FILE = os.path.join(RAW_DATA_DIR, "t10k-images-idx3-ubyte")
TEST_LABELS_FILE = os.path.join(RAW_DATA_DIR, "t10k-labels-idx1-ubyte")

def load_idx_images(file_path):
    with open(file_path, "rb") as f:
        magic, num_images = struct.unpack(">II", f.read(8))
        if magic != 2051:
            raise ValueError("Invalid magic number for images")
        rows, cols = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return data

def load_idx_labels(file_path):
    with open(file_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number for labels")
        return np.frombuffer(f.read(), dtype=np.uint8)

def normalize_and_reshape(images):
    # Convert to float32 and scale to [-1, 1]
    images = (images.astype(np.float32) / 127.5) - 1.0
    # Reshape to PyTorch format: (N, 1, 28, 28)
    return images[:, np.newaxis, :, :]

def save_npz(images, labels, split_name):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_DIR, f"mnist_{split_name}.npz")
    np.savez_compressed(path, images=images, labels=labels)
    print(f"Saved {split_name} set to {path}")

def process_mnist(subset_ratio=1.0):
    # Load data
    train_images = load_idx_images(TRAIN_IMAGES_FILE)
    train_labels = load_idx_labels(TRAIN_LABELS_FILE)
    test_images = load_idx_images(TEST_IMAGES_FILE)
    test_labels = load_idx_labels(TEST_LABELS_FILE)

    # Subset if needed
    if subset_ratio < 1.0:
        n_train = int(len(train_images) * subset_ratio)
        n_test = int(len(test_images) * subset_ratio)
        idx_train = np.random.choice(len(train_images), n_train, replace=False)
        idx_test = np.random.choice(len(test_images), n_test, replace=False)
        train_images, train_labels = train_images[idx_train], train_labels[idx_train]
        test_images, test_labels = test_images[idx_test], test_labels[idx_test]

    # Normalize and reshape
    train_images = normalize_and_reshape(train_images)
    test_images = normalize_and_reshape(test_images)

    # Save
    save_npz(train_images, train_labels, "train")
    save_npz(test_images, test_labels, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=1.0, help="Subset ratio (e.g., 0.1 for 10%)")
    args = parser.parse_args()
    process_mnist(args.subset)
