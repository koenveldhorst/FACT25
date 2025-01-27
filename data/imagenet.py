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
import torchvision.models as models
import torchvision.datasets as dataset
from PIL import Image
from tqdm import tqdm
from typing import List, Sequence, Dict, Set

IMAGENET_C_DIR = "imagenet_c"
IMAGENET_DIR = "imagenet"

# number of corruption levels for imagenet-c
N_CORRUPTION_SEVERITY_LVLS = 5

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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
            indexer: Indexer, transform, inetc_perc=0.1, inet_perc=0.5,
            classes: Set[str] = None
        ):
        """
        # Input
        * `inetc_perc`: percentage of each of the 5 corruption levels of imagenet-c to load
        * `inet_perc`: percentage of imagenet to load
        * `classes`: Wordnet IDs to load. If `None` all classes are loaded
        """

        assert os.path.exists(root), f"'{root}' does not exist yet. Please generate the dataset first."

        self.paths = []
        targets = []
        self.transform = transform
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

                # iterate over images
                for path in dirs[idx_start:idx_end]:                    
                    # add sample
                    self.paths.append(os.path.join(cls_dir, path))
                    targets.append(indexer.id_to_n[cls])
        
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

            # iterate over images
            for path in dirs[idx_start:idx_end]:
                # add sample
                self.paths.append(os.path.join(cls_dir, path))
                targets.append(indexer.id_to_n[cls])

        self.targets = torch.tensor(targets).long()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.transform(Image.open(path).convert('RGB'))

        return {
            "path": path,
            "label": self.targets[idx],
            "img": img
        }

    
class ImageNet(Dataset):
    def __init__(self, root: str, indexer: Indexer, transform, classes: Set[str] = None):
        """
        # Args
        * `classes`: Wordnet IDs to load. If `None` all classes are loaded
        """

        assert os.path.exists(root), f"'{root}' does not exist yet. Please generate the dataset first."

        self.paths = []
        targets = []
        self.transform = transform
        self.caption_dir = os.path.join(root, "imagenet_caption")
        self.classes = []

        dir = os.path.join(root, IMAGENET_DIR)

        # iterate over classes
        for cls in os.listdir(dir):
            if classes is not None and cls not in classes:
                continue
            self.classes.append(cls)
            
            cls_dir = os.path.join(dir, cls)
            # ignore files
            if not os.path.isdir(cls_dir):
                continue
            
            # iterate over images
            for path in os.listdir(cls_dir):
                # add sample
                self.paths.append(os.path.join(cls_dir, path))
                targets.append(indexer.id_to_n[cls])
        
        self.targets = torch.tensor(targets).long()
        
    def get_caption_dir(self):
        return self.caption_dir
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.transform(Image.open(path).convert('RGB'))

        return {
            "path": path,
            "label": self.targets[idx],
            "img": img
        }
    

def get_class_accuracies(
    data_root: str,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    idx = Indexer(os.path.join(data_root, "label_mapping.csv"))
    dataset = ImageNet(data_root, idx, transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, drop_last=False)

    model = models.resnet50(weights="IMAGENET1K_V1").to(device)

    correct = torch.zeros(len(dataset.classes))
    total = torch.zeros(len(dataset.classes))
    for batch in tqdm(loader):
        images = batch["img"].to(device)
        targets = batch["label"].to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i in range(len(preds)):
            pred, target = preds[i].item(), targets[i].item()
            correct[target] += pred == target
            total[target] += 1

    return { "correct": correct, "total": total, "acc": correct / total }
