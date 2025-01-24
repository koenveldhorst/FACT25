import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch
from torch import nn
import clip

# for loading dataset
from data.celeba import CelebA, get_transform_celeba
from data.waterbirds import Waterbirds, get_transform_cub
from data.imagenet import ImageNetIndexer, ImageNet, ImageNetC

# for various functions
from function.extract_caption import extract_caption ## default-> cuda:0/ clip:ViT-B/32
from function.extract_keyword import extract_keyword
from function.calculate_similarity import calc_similarity
from function.print_similarity import print_similarity

from tqdm import tqdm
import os
import time
import pandas as pd

import argparse
from typing import Dict

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--dataset", type = str, default = 'waterbird', help="dataset") #celeba, waterbird
    parser.add_argument("--model", type=str, default='best_model_CUB_erm.pth') #best_model_CelebA_erm.pth, best_model_CelebA_dro.pth, best_model_CUB_erm.pth, best_model_CUB_dro.pth
    parser.add_argument("--extract_caption", default = True) # TODO
    parser.add_argument("--save_result", default = True)
    args = parser.parse_args()
    return args

def load_dataset():
    # TODO: split in dataset and b2t function

    # TODO: device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    # load dataset
    args = parse_args()

    match args.dataset:
        case 'imagenet' | 'imagenet-r' | 'imagenet-c':
            imagenet_idx = ImageNetIndexer("data/imagenet_variants/label_mapping.csv")
            n_to_name = imagenet_idx.n_to_name
        case _:
            imagenet_idx = None

    match args.dataset:
        case 'waterbird':
            n_to_name = { 0: "landbird", 1: "waterbird" }

            caption_dir = 'data/cub/caption/'
            val_dataset = Waterbirds(
                data_dir='data/cub/data/waterbird_complete95_forest2water2',
                split='val', transform=get_transform_cub()
            )
        case 'celeba':
            n_to_name = { 0: "not blond", 1: "blond" }
            
            caption_dir = 'data/celebA/caption/'
            val_dataset = CelebA(
                data_dir='data/celebA/data/', split='val', transform=get_transform_celeba()
            )
        case 'imagenet':
            val_dataset = ImageNet("data/imagenet_variants", imagenet_idx, load_img=True)
            caption_dir = val_dataset.caption_dir
        case "imagenet-c":
            val_dataset = ImageNetC("data/imagenet_variants", "snow", imagenet_idx, load_img=True)
            caption_dir = val_dataset.caption_dir

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)

def b2t(
    dataloader: torch.utils.data.DataLoader, n_to_name: Dict,
    erm_model: nn.Module,
    dataset_name: str, model_name: str, extract_caption: bool,
    caption_dir: str, result_path="result/", model_dir="model/", diff_dir="diff/",
    device="cpu"      
):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)

    # extract caption
    if extract_caption:
        print("Start extracting captions..")
        if not os.path.exists(caption_dir):
            os.makedirs(caption_dir)

        for batch in tqdm(dataloader.dataset):
            img_path = batch["path"]

            caption = extract_caption(img_path)
            caption_path = os.path.join(
                caption_dir,
                str(batch["label"]) + "_" + os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            )

            with open(caption_path, 'w') as f:
                f.write(caption)

        print(f"Captions of {len(dataloader.dataset)} images extracted")

    # correctify dataset
    result_path = os.path.join(result_dir, dataset_name + "_" +  os.path.splitext(model_name)[0] + ".csv")

    # TODO: should this not run every time new captions are extracted also?
    if not os.path.exists(result_path):
        # TODO: fix
        match dataset_name:
            case 'imagenet' | 'imagenet-r' | 'imagenet-c':
                model = models.resnet50(weights="IMAGENET1K_V1")
            case _:
                model = torch.load(os.path.join(model_dir, model_name))

        model = model.to(device)
        model.eval()
        start_time = time.time()
        print("Pretrained model \"{}\" loaded".format(model_name))

        result = {
            "image": [],
            "pred": [],
            "actual": [],
            "group": [],
            "spurious": [],                
            "correct": [],
            "caption": [],
        }

        with torch.no_grad():
            running_corrects = 0
            for batch in tqdm(dataloader):
                images = batch["img"].to(device)
                targets = batch["label"].to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                for i in range(len(preds)):
                    img_path = batch["path"][i]
                    pred = preds[i]
                    actual = targets[i]

                    caption_path = os.path.join(
                        caption_dir,
                        str(actual.item()) + "_" + os.path.splitext(os.path.basename(img_path))[0] + ".txt"
                    )
                    with open(caption_path, "r") as f:
                        caption = f.readline()
                    
                    if "group_label" in batch:
                        group = batch.get("group_label")[i]
                        result['group'].append(group.item())
                    if "spurious_label" in batch:
                        spurious = batch.get("spurious_label")[i]
                        result['spurious'].append(spurious.item())

                    result['image'].append(img_path)
                    result['pred'].append(pred.item())
                    result['actual'].append(actual.item())
                    
                    result['caption'].append(caption)
                    if pred == actual:
                        result['correct'].append(1)
                        running_corrects += 1
                    else:
                        result['correct'].append(0)

            print("# of correct examples : ", running_corrects)
            print("# of wrong examples : ", len(dataloader.dataset) - running_corrects)
            print("# of all examples : ", len(dataloader.dataset))
            print("Accuracy : {:.2f}%".format(running_corrects/len(dataloader.dataset)*100))

        df = pd.DataFrame(result)
        df.to_csv(result_path)
        print("Classified result stored")
    else:
        df = pd.read_csv(result_path)
        print("Classified result \"{}\" loaded".format(result_path))

    # extract keyword
    df_wrong = df[df['correct'] == 0]
    df_correct = df[df['correct'] == 1]
    
    for label, name in n_to_name.items(): 
        df_class = df[df['actual'] == label] 
        df_wrong_class = df_wrong[df_wrong['actual'] == label]
        df_correct_class = df_correct[df_correct['actual'] == label]
        caption_wrong_class = ' '.join(df_wrong_class['caption'].tolist())
        keywords_class = extract_keyword(caption_wrong_class)

        # calculate similarity
        print("Start calculating scores..")
        # TODO: remove first arg from func?
        similarity_wrong_class = calc_similarity("", df_wrong_class['image'], keywords_class)
        similarity_correct_class = calc_similarity("", df_correct_class['image'], keywords_class)

        dist_class = similarity_wrong_class - similarity_correct_class
        
        print("Result for class :", name)
        diff_0 = print_similarity(keywords_class, keywords_class, dist_class, dist_class, df_class)
        print("*"*60)

        if args.save_result:
            diff_path = os.path.join(
                diff_dir,
                dataset_name + "_" +  os.path.splitext(model_name)[0] + "_" +  str(name) + ".csv"
            )
            diff_0.to_csv(diff_path)

if __name__ == "__main__":
    load_dataset()
    b2t()