import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from data.waterbirds import get_transform_cub
from PIL import Image
from tqdm import tqdm
import skimage.io as io
import pandas as pd
import clip
import torch
import re

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

def plot_roc_curves(subgroup_data, keywords, csv_file_path):
    """
    Generate and plot ROC curves for each subgroup (keyword).

    Parameters:
        subgroup_data (dict): Dictionary where keys are keywords and values are tuples of (predicted_scores, true_labels).
        keywords (list): List of keywords to plot ROC curves for.
    """
    plt.figure(figsize=(8, 6))

    for keyword in keywords:
        if keyword not in subgroup_data:
            print(f"Keyword '{keyword}' not found in data. Skipping.")
            continue

        predicted_scores, true_labels = subgroup_data[keyword]
        fpr, tpr, roc_auc = compute_roc_and_auc(predicted_scores, true_labels)

        plt.plot(fpr, tpr, label=f"{keyword} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line for random guessing
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves for Subgroups", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    fig_path = "plots/" + csv_file_path.split(".")[0].split("/")[1].split("_")[0] + "_roc_plot.png"
    plt.savefig(f"{fig_path}", dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")


def assign_to_subgroups(similarity_scores, keywords, thresholding_method="top3"):
    """
    Assign images to subgroups based on similarity scores and a thresholding method.

    Parameters:
        similarity_scores (np.ndarray): Array of similarity scores for all images and keywords.
        keywords (list): List of keywords for subgroup analysis.
        thresholding_method (str): Method to threshold images into subgroups. Options: "top3", "mean", "median".

    Returns:
        subgroup_indices (dict): Dictionary where keys are keywords and values are indices of images in that subgroup.
    """
    num_images, num_keywords = similarity_scores.shape
    subgroup_indices = {keyword: [] for keyword in keywords}

    for idx, keyword in enumerate(keywords):
        keyword_similarity_scores = similarity_scores[:, idx]

        if thresholding_method == "top1":
            # Assign an image to the subgroup if the score is the highest for that image
            top1_indices = np.argmax(similarity_scores, axis=1)
            in_top1 = top1_indices == idx
            subgroup_indices[keyword] = np.where(in_top1)[0]
        elif thresholding_method == "top3":
            # Assign an image to the subgroup if the score is among the top-3 for that image
            top3_indices = np.argsort(similarity_scores, axis=1)[:, -3:]
            in_top3 = np.any(top3_indices == idx, axis=1)
            subgroup_indices[keyword] = np.where(in_top3)[0]
        elif thresholding_method == "mean":
            # Assign an image to the subgroup if its score is above the mean
            mean_threshold = np.mean(keyword_similarity_scores)
            above_mean = keyword_similarity_scores > mean_threshold
            subgroup_indices[keyword] = np.where(above_mean)[0]
        elif thresholding_method == "median":
            # Assign an image to the subgroup if its score is above the median
            median_threshold = np.median(keyword_similarity_scores)
            above_median = keyword_similarity_scores > median_threshold
            subgroup_indices[keyword] = np.where(above_median)[0]
        elif thresholding_method == "custom":
            # Assign an image to the subgroup if its score is in the top 25%
            custom_threshold = np.percentile(keyword_similarity_scores, 60)
            above_custom = keyword_similarity_scores > custom_threshold
            subgroup_indices[keyword] = np.where(above_custom)[0]
        else:
            raise ValueError("Invalid thresholding_method. Choose from 'top1', 'top3', 'mean', 'median', 'custom'.")

    return subgroup_indices

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
    # Load the model
    image_paths = [image_dir + image for image in images]
    images = [Image.fromarray(io.imread(image)) for image in image_paths]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    image_inputs = torch.cat([preprocess(pil_image).unsqueeze(0) for pil_image in images]).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in keywords]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity (batch of images against keywords)
    scores = (image_features @ text_features.T) # Shape: (num_images, num_keywords)
    return scores

def classification_score(image_dir, images):
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

def calculate_and_plot_auroc(image_dir, results_csv, keywords, thresholding_method="mean"):
    """
    Integrate AUROC calculation and plotting with the existing setup.

    Parameters:
        image_dir (str): Directory containing images.
        results_csv (str): Path to the CSV file with classification results.
        keywords (list): List of keywords for subgroup analysis.
    """
    # Load classification results
    results = pd.read_csv(results_csv)

    # Prepare image list where actual label is 1 (waterbird)
    image_list = results.loc[results["actual"] == 1, "image"].to_list()

    # Calculate similarity scores for all images and all keywords
    similarity_scores = similarity_func(image_dir, image_list, keywords).cpu().numpy()

    # Assign images to subgroups based on similarity scores
    subgroup_indices = assign_to_subgroups(similarity_scores, keywords, thresholding_method)

    # Prepare data for ROC calculation
    subgroup_data = {}
    for keyword, indices in subgroup_indices.items():
        # Ensure the indices map to actual labels
        image_paths = [image_list[i] for i in indices]
        predicted_scores = classification_score(image_dir, image_paths)
        true_labels = results.iloc[indices]["actual"].to_numpy()

        # Store predicted scores and true labels
        print(f"Keyword '{keyword}': {len(predicted_scores)} scores and {len(true_labels)} labels.")
        subgroup_data[keyword] = (predicted_scores, true_labels)

    # Plot ROC curves
    plot_roc_curves(subgroup_data, keywords, results_csv)

# Example usage
if __name__ == "__main__":
    image_dir = "data/cub/data/waterbird_complete95_forest2water2/"
    results_csv = "result/waterbird_best_model_Waterbirds_erm.csv"
    keywords = ["bamboo", "forest", "woods", "species", "bird"]
    calculate_and_plot_auroc(image_dir, results_csv, keywords, thresholding_method="mean")
