import os
import clip
import torch
from pathlib import Path
import skimage.io as io
from PIL import Image
from tqdm import tqdm
import pandas as pd  

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def calc_similarity(
        images, keywords, model, preprocess, extract_sim_matrix=False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
    # return nothing if there are no keywords
    if len(keywords) == 0:
        return torch.empty(0)

    # load the model
    images = [Image.fromarray(io.imread(image)) for image in images]

    similarity_list = []
    image_list_chunked = list_chunk(images, 2000)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in keywords]).to(device)
    for image_list in tqdm(image_list_chunked):
        # prepare the inputs
        image_inputs = torch.cat([preprocess(pil_image).unsqueeze(0) for pil_image in image_list]).to(device) # (1909, 3, 224, 224)

        # calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_inputs)

        # pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T) # (1909, 20)
        similarity_list.append(similarity)

    if extract_sim_matrix: 
        similarity_matrix = torch.cat(similarity_list)
        return similarity_matrix
    else: 
        similarity = torch.cat(similarity_list).mean(dim=0)
        return similarity