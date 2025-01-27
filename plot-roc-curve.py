from sklearn.metrics import roc_curve, auc
from data.waterbirds import get_transform_cub
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import pandas as pd
import clip
import torch

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def similarity_func(image_dir, images, keywords):
    """
    Calculate similarity scores for all images and all keywords.

    Parameters:
        image_dir (str): Directory containing images.
        images (list): List of image file paths.
        keywords (list): List of keywords to compute similarity.

    Returns:
        similarity_scores (torch.Tensor): Tensor of similarity scores, shape (num_images, num_keywords).
    """
    # Load the model and process images and keywords
    image_paths = [image_dir + image for image in images]
    images = [Image.fromarray(io.imread(image)) for image in image_paths]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    image_inputs = torch.cat([preprocess(pil_image).unsqueeze(0) for pil_image in images]).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {keyword}") for keyword in keywords]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity (batch of images against keywords)
    scores = (image_features @ text_features.T).cpu().numpy()
    return scores

def classification_score(image_dir, images):
    """
    Calculate classification scores for all images.

    Parameters:
        image_dir (str): Directory containing images.
        images (list): List of image file paths.

    Returns:
        probabilities (list): List of classification probabilities.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = "model/"

    # Load the model
    model = torch.load(model_dir + "best_model_Waterbirds_erm.pth", map_location=device)
    model = model.to(device)
    model.eval()

    # Preprocess images
    preprocess = get_transform_cub()
    image_paths = [image_dir + image for image in images]

    # Perform inference
    probabilities = []
    with torch.no_grad():
        for image in tqdm(image_paths, desc="Classifying images"):
            images = Image.open(image).convert("RGB")
            images = preprocess(images).unsqueeze(0).to(device)
            outputs = model(images)
            probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    return probabilities


def assign_to_subgroups(similarity_scores, keywords, thresholding_method="mean"):
    """
    Assign images to subgroups based on similarity scores and a thresholding method.

    Parameters:
        similarity_scores (np.ndarray): Array of similarity scores for all images and keywords.
        keywords (list): List of keywords for subgroup analysis.
        thresholding_method (str): subgroup method. Options: "mean", "median", "percentile".

    Returns:
        subgroup_labels (dict): Keys: keyword. Values: 1 = high similarity, 0 = low similarity.
    """
    subgroup_labels = {}
    valid_methods = ["mean", "median", "percentile"]
    for idx, keyword in enumerate(keywords):
        keyword_similarity_scores = similarity_scores[:, idx]

        if thresholding_method == "mean":
            threshold = np.mean(keyword_similarity_scores)
        elif thresholding_method == "median":
            threshold = np.median(keyword_similarity_scores)
        elif thresholding_method == "percentile":
            threshold = np.percentile(keyword_similarity_scores, 75)
        else:
            raise ValueError(f"Invalid thresholding_method. Choose from {valid_methods}")

        binary_labels = (keyword_similarity_scores > threshold).astype(int)
        subgroup_labels[keyword] = (keyword_similarity_scores, binary_labels)

    return subgroup_labels


def compute_roc_and_auc(predicted_scores, true_labels):
    """
    Compute ROC curve and AUC for a given set of predicted scores and true labels.

    Parameters:
        predicted_scores (np.ndarray): Array of model's predicted probabilities or scores.
        true_labels (np.ndarray): Array of binary labels (1 for positive class, 0 for negative class).

    Returns:
        fpr (np.ndarray): False positive rates.
        tpr (np.ndarray): True positive rates.
        roc_auc (float): Area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curves(subgroup_data, keywords, clip_scores, csv_file_path):
    """
    Generate and plot ROC curves for each subgroup (keyword).

    Parameters:
        subgroup_data (dict): Dictionary where keys are keywords and values are tuples of (predicted_scores, true_labels).
        keywords (list): List of keywords to plot ROC curves for.
    """
    plt.figure(figsize=(6, 5))

    for keyword in keywords:
        if keyword not in subgroup_data:
            print(f"Keyword '{keyword}' not found in data. Skipping.")
            continue

        predicted_scores, true_labels = subgroup_data[keyword]
        fpr, tpr, roc_auc = compute_roc_and_auc(predicted_scores, true_labels)

        plt.plot(fpr, tpr, label=f"{keyword} ({clip_scores[keyword]:.2f}) = {roc_auc:.2f}", linewidth=2)

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line for random guessing
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curves for Subgroups", fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    plt.xlim([0, 1])
    plt.ylim([0, 1])


    fig_path = "plots/" + csv_file_path.split(".")[0].split("/")[1].split("_")[0] + "_roc_plot.png"
    plt.savefig(f"{fig_path}", dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    image_dir = "data/cub/data/waterbird_complete95_forest2water2/"
    results_csv = "result/waterbird_best_model_Waterbirds_erm.csv"
    b2t_csv = "diff/waterbird_best_model_Waterbirds_erm_waterbird.csv"
    keywords = ["bamboo", "forest", "woods", "species", "bird"]

    # Load in the csv files
    results = pd.read_csv(results_csv)
    b2t_scores = pd.read_csv(b2t_csv)

    # Clip scores per keyword + image subset
    clip_scores = b2t_scores.loc[b2t_scores["Keyword"].isin(keywords), "Score"].values
    clip_scores = {keyword: score for keyword, score in zip(keywords, clip_scores)}
    images = results.loc[results["actual"] == 1, "image"].to_list() # Only waterbird images

    # Compute scores and determine subgroups
    similarity_scores = similarity_func(image_dir, images, keywords)
    classifier_scores = classification_score(image_dir, images)
    subgroup_labels = assign_to_subgroups(similarity_scores, keywords, thresholding_method="mean")

    # Prepare data for plotting
    subgroup_data = {keyword: (classifier_scores, subgroup_labels[keyword][1]) for keyword in keywords}

    plot_roc_curves(subgroup_data, keywords, clip_scores, results_csv)