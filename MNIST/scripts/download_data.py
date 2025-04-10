import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define data directories
DATA_DIR = "data/raw/"

def download_mnist():
    """Download MNIST dataset if it is not already present."""
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        print(f"MNIST dataset already exists in `{DATA_DIR}`, skipping download.")
        return

    print("Downloading MNIST dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
    datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

    print(f"MNIST dataset downloaded and stored in `{DATA_DIR}`.")

# Run only if executed directly
if __name__ == "__main__":
    download_mnist()
