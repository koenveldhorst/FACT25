"""
Dataset classes for ImageNet and its variants


Expects the following file structure:

    data
    ↪ ImageNet
        ↪ Wordnet ID #0
            ↪ Image #0
            ↪ ...

    ↪ ImageNetC
        ↪ Distortion #0
            ↪ Severity level #0
                ↪ Wordnet ID #0
                    ↪ Image #0

                    ↪ ...

                ↪ ...
            ↪ ...
        ↪ ...
"""

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dataset
from PIL import Image
from typing import List, Sequence, Dict

IMAGENET_C_DIR = "imagenet_c"
IMAGENET_DIR = "imagenet"

class ImageNetIndexer:
    """
    Container of dictionaries for useful imagenet mappings between index, wordnet id and name.
    """

    def __init__(self, path: str):
        # columns are "n", "name" and "id" (i.e., wordnet id)
        df = pd.read_csv(path, sep="|")

        self.n_to_id = dict(zip(df["n"], df["id"]))
        self.n_to_name = dict(zip(df["n"], df["name"]))

        self.id_to_n = dict(zip(df["id"], df["n"]))
        self.id_to_name = dict(zip(df["id"], df["name"]))

class ImageNetC(Dataset):
    """
    Consists of 10% of each of 5 severity levels + 50% from imagenet.
    """

    def __init__(
            self, root: str, distortion_name: str, severity_levels: Sequence[int],
            indexer: ImageNetIndexer, inetc_perc=0.1, inet_perc=0.5
        ):

        self.samples = []
        self.transform = get_imagenet_transform()
        
        # iterate over imagenet c severity levels
        for lvl in severity_levels:
            dir = os.path.join(root, IMAGENET_C_DIR, distortion_name, str(lvl))

            # iterate over classes
            for cls in os.listdir(dir):
                cls_dir = os.path.join(dir, cls)

                if not os.path.isdir(cls_dir):
                    continue
                
                img_cap = int(len(os.listdir(cls_dir)) * inetc_perc)

                # iterate over images
                for i, path in enumerate(os.listdir(cls_dir), 1):
                    if i > img_cap:
                        break

                    # add sample
                    self.samples.append((
                        os.path.join(cls_dir, path),
                        indexer.id_to_n[cls]
                    ))
        
        dir = os.path.join(root, IMAGENET_DIR)

        # iterate over classes
        for cls in os.listdir(dir):
            cls_dir = os.path.join(dir, cls)

            if not os.path.isdir(cls_dir):
                continue
            
            img_cap = int(len(os.listdir(cls_dir)) * inet_perc)

            # iterate over images
            for i, path in enumerate(os.listdir(cls_dir), 1):
                if i > img_cap:
                    break
                self.samples.append((os.path.join(cls_dir, path), cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = self.transform(Image.open(path).convert('RGB'))
        return img, label
    
class ImageNet(Dataset):
    def __init__(self, root: str, indexer: ImageNetIndexer):
        self.samples = []
        self.transform = get_imagenet_transform()
        self.indexer = indexer

        dir = os.path.join(root, IMAGENET_DIR)

        # iterate over classes
        for cls in os.listdir(dir):
            cls_dir = os.path.join(dir, cls)

            if not os.path.isdir(cls_dir):
                continue
            
            # iterate over images
            for path in os.listdir(cls_dir):
                # add sample
                self.samples.append((
                    os.path.join(cls_dir, path),
                    indexer.id_to_n[cls]
                ))

        self.caption_dir = os.path.join(root, "caption")
        
    def get_caption_dir(self):
        return self.caption_dir
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = self.transform(Image.open(path).convert('RGB'))
        return img, label

def get_imagenet_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    