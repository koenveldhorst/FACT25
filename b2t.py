# TODO: update readme with updated dataset loading

import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch import nn

# for loading dataset
from data import celeba, waterbirds, imagenet

# for various functions
from function.extract_caption import extract_caption ## default-> cuda:0/ clip:ViT-B/32
from function.extract_keyword import extract_keyword
from function.calculate_similarity import calc_similarity
from function.print_similarity import print_similarity

from tqdm import tqdm
import clip
import os
import pandas as pd
from collections import defaultdict

import argparse
from typing import Dict, Any

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--dataset", type = str, default = 'waterbird', help="dataset") #celeba, waterbird
    parser.add_argument("--model", type=str, default='best_model_CUB_erm.pth') #best_model_CelebA_erm.pth, best_model_CelebA_dro.pth, best_model_CUB_erm.pth, best_model_CUB_dro.pth
    parser.add_argument("--extract_caption", action="store_true", default=False)
    parser.add_argument("--save_result", default = True)
    args = parser.parse_args()
    return args

def load_dataset(
    dataset_name: str
):
    """
    Loads one of the following datasets: 'imagenet', 'imagenet-r', 'imagenet-c',
    'waterbird', or 'celeba'.
    """
    match dataset_name:
        case 'imagenet' | 'imagenet-r' | 'imagenet-c':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imagenet.MEAN, imagenet.STD)
            ])

            imagenet_idx = imagenet.Indexer("data/imagenet_variants/label_mapping.csv")
        case _:
            transform = classes = n_to_name = imagenet_idx = None

    match dataset_name:
        case 'waterbird':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(waterbirds.MEAN, waterbirds.STD)
            ])                
            n_to_name = { 0: "landbird", 1: "waterbird" }

            caption_dir = 'data/cub/caption/'
            dataset = waterbirds.Waterbirds(
                root='data/cub/data/waterbird_complete95_forest2water2',
                split='valid',
                transform=transform
            )
        case 'celeba':
            transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(celeba.MEAN, celeba.STD),
            ])
            n_to_name = { 0: "not blond", 1: "blond" }
            
            caption_dir = 'data/celeba/caption/'
            dataset = celeba.CelebA(
                root='data/celeba/',
                split='valid',
                transform=transform
            )
        case 'imagenet':
            dataset = imagenet.ImageNet(
                "data/imagenet_variants_curated", imagenet_idx, transform,
                max_per_class=50, classes=["n02219486"]
            )
            caption_dir = dataset.caption_dir

            n_to_name = { imagenet_idx.id_to_n[id]: imagenet_idx.id_to_name[id] for id in dataset.classes }
        case "imagenet-c":
            classes = ["n02690373", "n02009912"] # airliner, American egret

            dataset = imagenet.ImageNetC(
                "data/imagenet_variants", "weather/snow", imagenet_idx, transform,
                classes=classes
            )
            caption_dir = dataset.caption_dir

            n_to_name = { imagenet_idx.id_to_n[id]: imagenet_idx.id_to_name[id] for id in dataset.classes }
        case "imagenet-r":
            classes = ["n02769748", "n01484850"] # backpack, white shark

            dataset = imagenet.ImageNet(
                "data/imagenet_variants", imagenet_idx, transform,
                classes=classes, folder=imagenet.IMAGENET_R_DIR
            )
            caption_dir = dataset.caption_dir

    loader = DataLoader(dataset, batch_size=256, num_workers=4, drop_last=False)

    return loader, n_to_name, caption_dir

def b2t(
    dataloader: torch.utils.data.DataLoader,
    n_to_name: Dict[int, str],
    model: nn.Module,
    caption_dir: str,
    overwrite_captions: bool,
    result_file: str,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, pd.DataFrame]:
    """
    Performs bias keyword discovery on image data.

    Captions are stored as "[class index]_[file name].txt in the specified `caption_dir`.

    # Input
    * `dataloader`: Torch dataloader that returns paths to images and respective class labels
    * `n_to_name`: Mapping from label number to string name
    * `model`: Torch model used for classification
    * `caption_dir`: Folder where generated captions are written to
    * `extract_caption`: Whether to overwrite captions if caption folder already exists
    * `result_file`: Classification result output path
    * `device`: Torch device

    # Returns
    Returns the mappings between class names and discovered bias keywords
    """

    # * generate captions *
    caption_dir_exists = os.path.exists(caption_dir)

    if not caption_dir_exists or overwrite_captions:
        print("Start extracting captions..")        
        if not caption_dir_exists:
            os.makedirs(caption_dir)

        for batch in tqdm(dataloader.dataset):
            img_path = batch["path"]

            caption = extract_caption(img_path)
            caption_path = os.path.join(
                caption_dir,
                f"{str(batch["label"].item())}_{os.path.splitext(os.path.basename(img_path))[0]}.txt"
            )

            with open(caption_path, 'w') as f:
                f.write(caption)

        print(f"Captions of {len(dataloader.dataset)} images extracted")

    # * classify samples *
    if overwrite_captions or not os.path.isfile(result_file):
        result_dir, _ = os.path.split(result_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        model = model.to(device)
        model.eval()

        result = defaultdict(lambda: [])

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
                        f"{str(actual.item())}_{os.path.splitext(os.path.basename(img_path))[0]}.txt"
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

        df_result = pd.DataFrame(result)
        df_result.to_csv(result_file)
        print("Classified result stored")
    else:
        df_result = pd.read_csv(result_file)
        print("Classified result \"{}\" loaded".format(result_file))

    # * extract keywords *
    # false positives and negatives
    df_incorrect = df_result[df_result['correct'] == 0]
    # true positives and negatives
    df_correct = df_result[df_result['correct'] == 1]
    
    class_keywords = {}
    model, preprocess = clip.load('ViT-B/32', device)
    for label, name in n_to_name.items():         
        # false negatives & true positives
        df_incorrect_class = df_incorrect[df_incorrect['actual'] == label]
        df_correct_class = df_correct[df_correct['actual'] == label]

        # concatenate captions of false negative images and extract captions
        caption_wrong_class = ' '.join(df_incorrect_class['caption'].tolist())
        keywords_class = extract_keyword(caption_wrong_class)

        # calculate similarity
        print("Start calculating scores..")
        similarity_wrong_class = calc_similarity(df_incorrect_class['image'], keywords_class, model, preprocess, device)
        similarity_correct_class = calc_similarity(df_correct_class['image'], keywords_class, model, preprocess, device)

        dist_class = similarity_wrong_class - similarity_correct_class
        
        print("Result for class :", name)
        df_class = df_result[df_result['actual'] == label] 
        class_keywords[name] = print_similarity(keywords_class, dist_class, df_class)
        print("*"*60)

    return class_keywords

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    model_name = os.path.splitext(args.model)[0]

    match model_name:
        # check for torchvision model
        case "imagenet-resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1")
        case "imagenet-ViT":
            model = models.vit_b_16(weights="IMAGENET1K_V1")
        # otherwise load from model folder
        case _:
            model = torch.load(os.path.join("model/", args.model), map_location=device)

    # load data
    loader, n_to_name, caption_dir = load_dataset(args.dataset)

    # run B2T
    class_keywords = b2t(
        loader, n_to_name, model, caption_dir,
        overwrite_captions=args.extract_caption,
        result_file=f"result/{args.dataset}_{model_name}.csv",
        device=device
    )

    # store keywords
    for class_name, df_keywords in class_keywords.items():
        df_keywords.to_csv(
            os.path.join("diff", f"{args.dataset}_{model_name}_{class_name}.csv")
        )
