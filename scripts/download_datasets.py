#!/usr/bin/env python3
"""
Script to download datasets used in the generative models project.
"""

import os
import argparse
import requests
import zipfile
import io
from tqdm import tqdm
import torchvision.datasets as datasets
from torchvision import transforms

def download_file(url, destination):
    """Download a file from url to destination with progress bar"""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
        
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download incomplete")

def download_and_extract_zip(url, extract_to):
    """Download a zip file and extract its contents"""
    print(f"Downloading from {url}...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to download: {response.status_code}")
        return
    
    print("Extracting zip file...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Files extracted to {extract_to}")

def download_mnist(data_dir):
    """Download MNIST dataset"""
    print("Downloading MNIST dataset...")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
        print(f"MNIST dataset downloaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    except Exception as e:
        print(f"Error downloading MNIST: {e}")

def download_cifar10(data_dir):
    """Download CIFAR-10 dataset"""
    print("Downloading CIFAR-10 dataset...")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
        print(f"CIFAR-10 dataset downloaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}")

def download_tiny_shakespeare(data_dir):
    """Download Tiny Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = os.path.join(data_dir, "tiny_shakespeare.txt")
    
    print("Downloading Tiny Shakespeare dataset...")
    os.makedirs(data_dir, exist_ok=True)
    
    download_file(url, filepath)
    print(f"Tiny Shakespeare dataset downloaded to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for generative models")
    parser.add_argument("--dataset", type=str, choices=["all", "mnist", "cifar10", "tiny_shakespeare"], 
                      default="all", help="Dataset to download")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save datasets")
    args = parser.parse_args()
    
    if args.dataset in ["all", "mnist"]:
        download_mnist(os.path.join(args.data_dir, "mnist"))
        
    if args.dataset in ["all", "cifar10"]:
        download_cifar10(os.path.join(args.data_dir, "cifar10"))
        
    if args.dataset in ["all", "tiny_shakespeare"]:
        download_tiny_shakespeare(os.path.join(args.data_dir, "text"))
    
    print("Downloads complete!")

if __name__ == "__main__":
    main()