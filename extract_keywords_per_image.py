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
    args = parser.parse_args()
    return args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset
args = parse_args()

if args.dataset == 'waterbird':
    class_names = ['landbird', 'waterbird']
    image_dir = ''
elif args.dataset == 'celeba':
    class_names = ['not blond', 'blond']
    image_dir = ''
elif args.dataset == 'imagenet-r':
    reverse_mapping = {v: k for k, v in map_folder_to_imagenet.items()}
    image_dir = ''
    class_names = [reverse_mapping[label] for label in ["n01484850", "n02769748"]]
elif args.dataset == 'imagenet-c':
    reverse_mapping = {v: k for k, v in map_folder_to_imagenet.items()}
    image_dir = ''
    class_names = [reverse_mapping[label] for label in ["n02690373", "n02009912"]]
elif args.dataset == 'imagenet':
    reverse_mapping = {v: k for k, v in map_folder_to_imagenet.items()}
    image_dir = ''
    class_names = [reverse_mapping[label] for label in ["n03642806", "n04317175", "n02219486", "n03535780", "n02071294", "n03781244"]]
    # class_names = [reverse_mapping[label] for label in ["n01440764", "n01443537", "n01484850", "n02219486"]]

result_dir = 'result/'
result_path = result_dir + args.dataset +"_" +  args.model.split(".")[0] + ".csv"
df = pd.read_csv(result_path)
print("Classified result \"{}\" loaded".format(result_path))

keyword_dir = 'keyword_extraction_csv/'
if not os.path.exists(keyword_dir):
    os.makedirs(keyword_dir)

# extract keyword
if args.dataset == 'imagenet-r' or args.dataset == 'imagenet' or args.dataset == 'imagenet-c':
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
        similarity_matrix = calc_similarity(image_dir, df_wrong_class['image'], keywords_class, extract_sim_matrix=True)
        keywords_df = keyword_per_img(similarity_matrix, df_wrong_class['image'].tolist(), df_wrong_class['pred'].tolist(), 
                                      df_wrong_class['actual'].tolist(), df_wrong_class['caption'].tolist(), keywords_class)
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
    similarity_matrix_class_0 = calc_similarity(image_dir, df_wrong_class_0['image'], keywords_class_0, extract_sim_matrix=True)
    similarity_matrix_class_1 = calc_similarity(image_dir, df_wrong_class_1['image'], keywords_class_1, extract_sim_matrix=True)

    keywords_df_class_0 = keyword_per_img(similarity_matrix_class_0, df_wrong_class_0['image'].tolist(), df_wrong_class_0['pred'].tolist(), 
                                      df_wrong_class_0['actual'].tolist(), df_wrong_class_0['caption'].tolist(), keywords_class_0)
    keywords_df_class_1 = keyword_per_img(similarity_matrix_class_1, df_wrong_class_1['image'].tolist(), df_wrong_class_1['pred'].tolist(), 
                                      df_wrong_class_1['actual'].tolist(), df_wrong_class_1['caption'].tolist(), keywords_class_1)
 
    keyword_path_class_0 = keyword_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[0] + ".csv"
    keyword_path_class_1 = keyword_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[1] + ".csv"
    keywords_df_class_0.to_csv(keyword_path_class_0)
    keywords_df_class_1.to_csv(keyword_path_class_1)