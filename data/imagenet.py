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
from typing import List, Sequence, Dict, Set

IMAGENET_C_DIR = "imagenet_c"
IMAGENET_DIR = "imagenet"

# number of corruption levels for imagenet-c
N_CORRUPTION_SEVERITY_LVLS = 5

class Indexer:
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
            self, root: str, distortion_name: str,
            indexer: Indexer, transform, inetc_perc=0.1, inet_perc=0.5, load_img=False,
            classes: Set[str] = None
        ):
        """
        # Input
        * `inetc_perc`: percentage of each of the 5 corruption levels of imagenet-c to load
        * `inet_perc`: percentage of imagenet to load
        * `load_img`: If `True`, images will be automatically loaded when `__getitem__` is called
        * `classes`: Wordnet IDs to load. If `None` all classes are loaded
        """

        assert os.path.exists(root), f"'{root}' does not exist yet. Please generate the dataset first."

        self.samples = []
        self.transform = transform
        self.load_img = load_img
        self.caption_dir = os.path.join(root, f"imagenet-c_caption_{distortion_name}")

        # iterate over imagenet c severity levels
        for lvl in range(1, N_CORRUPTION_SEVERITY_LVLS+1):
            dir = os.path.join(root, IMAGENET_C_DIR, distortion_name, str(lvl))

            # iterate over classes
            for cls in os.listdir(dir):
                if classes is not None and cls not in classes:
                    continue

                cls_dir = os.path.join(dir, cls)
                # ignore files
                if not os.path.isdir(cls_dir):
                    continue
                
                dirs = os.listdir(cls_dir)

                # take a different section from each level to avoid repeated images
                idx_start = int(len(dirs) * inetc_perc) * (lvl-1)
                idx_end = int(len(dirs) * inetc_perc) * lvl
                print(idx_end - idx_start, idx_start, idx_end)

                # iterate over images
                for path in dirs[idx_start:idx_end]:                    
                    # add sample
                    self.samples.append((
                        os.path.join(cls_dir, path),
                        indexer.id_to_n[cls]
                    ))
        
        dir = os.path.join(root, IMAGENET_DIR)

        # iterate over classes
        for cls in os.listdir(dir):
            if classes is not None and cls not in classes:
                continue
        
            cls_dir = os.path.join(dir, cls)
            # ignore files
            if not os.path.isdir(cls_dir):
                continue
            
            dirs = os.listdir(cls_dir)

            # take a different section from corrupted images to avoid repeated images
            idx_start = int(len(dirs) * inetc_perc) * N_CORRUPTION_SEVERITY_LVLS
            idx_end = idx_start + int(len(dirs) * inet_perc)
            print(idx_end - idx_start, idx_start, idx_end)

            # iterate over images
            for path in dirs[idx_start:idx_end]:
                self.samples.append((os.path.join(cls_dir, path), cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert('RGB')) if self.load_img else None

        return {
            "path": path,
            "label": label,
            "img": img
        }

    
class ImageNet(Dataset):
    def __init__(self, root: str, indexer: Indexer, transform, load_img=False, classes: Set[str] = None):
        """
        # Args
        * `load_img`: If `True`, images will be automatically loaded when `__getitem__` is called
        * `classes`: Wordnet IDs to load. If `None` all classes are loaded
        """

        assert os.path.exists(root), f"'{root}' does not exist yet. Please generate the dataset first."

        self.samples = []
        self.transform = transform
        self.load_img = load_img
        self.caption_dir = os.path.join(root, "imagenet_caption")

        dir = os.path.join(root, IMAGENET_DIR)

        # iterate over classes
        for cls in os.listdir(dir):
            if classes is not None and cls not in classes:
                continue
            
            cls_dir = os.path.join(dir, cls)
            # ignore files
            if not os.path.isdir(cls_dir):
                continue
            
            # iterate over images
            for path in os.listdir(cls_dir):
                # add sample
                self.samples.append((
                    os.path.join(cls_dir, path),
                    indexer.id_to_n[cls]
                ))
        
    def get_caption_dir(self):
        return self.caption_dir
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert('RGB')) if self.load_img else None

        return {
            "path": path,
            "label": label,
            "img": img
        }
    