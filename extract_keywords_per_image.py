import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torchvision.models as models
from torch.utils.data import DataLoader
import torch
import clip

# for loading dataset
from data.celeba import CelebA, get_transform_celeba
from data.waterbirds import Waterbirds, get_transform_cub

# for various functions
from function.extract_keyword import extract_keyword
from function.calculate_similarity import calc_similarity
from keyword_function import keyword_per_img

from mapping_labels import map_folder_to_imagenet

from tqdm import tqdm
import os
import time
import pandas as pd


import argparse

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--dataset", type = str, default = 'waterbird', help="dataset") #celeba, waterbird
    parser.add_argument("--model", type=str, default='best_model_CUB_erm.pth') #best_model_CelebA_erm.pth, best_model_CelebA_dro.pth, best_model_CUB_erm.pth, best_model_CUB_dro.pth
    # parser.add_argument("--extract_caption", default = True)
    # parser.add_argument("--save_result", default = True)
    args = parser.parse_args()
    return args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset
args = parse_args()

if args.dataset == 'waterbird':
    preprocess = get_transform_cub()
    class_names = ['landbird', 'waterbird']
    # group_names = ['landbird_land', 'landbird_water', 'waterbird_land', 'waterbird_water']
    image_dir = 'data/cub/data/waterbird_complete95_forest2water2/'
    caption_dir = 'data/cub/caption/'
    val_dataset = Waterbirds(data_dir='data/cub/data/waterbird_complete95_forest2water2', split='val', transform=preprocess)
elif args.dataset == 'celeba':
    preprocess = get_transform_celeba()
    class_names = ['not blond', 'blond']
    # group_names = ['not blond_female', 'not blond_male', 'blond_female', 'blond_male']
    image_dir = 'data/celebA/data/image_align_celeba'
    caption_dir = 'data/celebA/caption/'
    val_dataset = CelebA(data_dir='data/celebA/data/', split='val', transform=preprocess)
elif args.dataset == 'imagenet-r':
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    reverse_mapping = {v: k for k, v in map_folder_to_imagenet.items()}
    # tested on subset of data, to test on real data: use 'data/imagenetR/data/imagenet-r/'
    image_dir = ''
    caption_dir = 'data/imagenetR/caption/'
    val_dataset = dataset.ImageFolder(root="data/imagenetR/data/imagenet-r-test", transform=preprocess, 
                                      target_transform=lambda label: reverse_mapping[val_dataset.classes[label]])
    class_names = [reverse_mapping[label] for label in val_dataset.classes]
elif args.dataset == 'imagenet':
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    reverse_mapping = {v: k for k, v in map_folder_to_imagenet.items()}
    # tested on subset of data, to test on real data: use 'data/imagenetR/data/imagenet-r/'
    image_dir = ''
    caption_dir = 'data/imagenet/caption/'
    val_dataset = dataset.ImageFolder(root="data/imagenet/data/imagenet-test", transform=preprocess, 
                                      target_transform=lambda label: reverse_mapping[val_dataset.classes[label]])
    class_names = [reverse_mapping[label] for label in val_dataset.classes]

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)


result_dir = 'result/'
result_path = result_dir + args.dataset +"_" +  args.model.split(".")[0] + ".csv"
df = pd.read_csv(result_path)
print("Classified result \"{}\" loaded".format(result_path))

keyword_dir = 'keyword_extraction_csv/'
if not os.path.exists(keyword_dir):
    os.makedirs(keyword_dir)

# extract keyword
if args.dataset == 'imagenet-r' or args.dataset == 'imagenet':
    df_wrong = df[df['correct'] == 0]
    df_correct = df[df['correct'] == 1]
    for labels in class_names: 
        df_class = df[df['actual'] == labels] 
        df_wrong_class = df_wrong[df_wrong['actual'] == labels]
        df_correct_class = df_correct[df_correct['actual'] == labels]
        caption_wrong_class = ' '.join(df_wrong_class['caption'].tolist())
        keywords_class = extract_keyword(caption_wrong_class)

        # calculate similarity
        print("Start calculating scores..")
        similarity, similarity_matrix, list_images = calc_similarity(image_dir, df_wrong_class['image'], keywords_class, extract_sim_matrix=True)
        keywords_df = keyword_per_img(similarity_matrix, list_images, keywords_class)
        keyword_path = keyword_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  str(labels) + ".csv"
        keywords_df.to_csv(keyword_path)

else:
    df_wrong = df[df['correct'] == 0]
    df_correct = df[df['correct'] == 1]
    df_class_0 = df[df['actual'] == 0] # not blond, landbird
    df_class_1 = df[df['actual'] == 1] # blond, waterbird
    df_wrong_class_0 = df_wrong[df_wrong['actual'] == 0]
    df_wrong_class_1 = df_wrong[df_wrong['actual'] == 1]
    df_correct_class_0 = df_correct[df_correct['actual'] == 0]
    df_correct_class_1 = df_correct[df_correct['actual'] == 1]

    caption_wrong_class_0 = ' '.join(df_wrong_class_0['caption'].tolist())
    caption_wrong_class_1 = ' '.join(df_wrong_class_1['caption'].tolist())

    keywords_class_0 = extract_keyword(caption_wrong_class_0)
    keywords_class_1 = extract_keyword(caption_wrong_class_1)

    # calculate similarity
    print("Start calculating scores..")
    similarity_wrong_class_0, similarity_matrix_class_0, list_images_class_0 = calc_similarity(image_dir, df_wrong_class_0['image'], keywords_class_0, extract_sim_matrix=True)
    similarity_wrong_class_1, similarity_matrix_class_1, list_images_class_1 = calc_similarity(image_dir, df_wrong_class_1['image'], keywords_class_1, extract_sim_matrix=True)

    keywords_df_class_0 = keyword_per_img(similarity_matrix_class_0, list_images_class_0, class_names[0])
    keywords_df_class_1 = keyword_per_img(similarity_matrix_class_1, list_images_class_1, class_names[1])
 
    keyword_path_class_0 = keyword_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[0] + ".csv"
    keyword_path_class_1 = keyword_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[1] + ".csv"
    keywords_df_class_0.to_csv(keyword_path_class_0)
    keywords_df_class_1.to_csv(keyword_path_class_1)