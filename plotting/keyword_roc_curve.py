from scipy.stats import linregress
from sklearn.metrics import roc_curve, auc
from data import waterbirds, celeba, imagenet
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.models as models
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


def similarity_func(image_paths, keywords):
    """
    Calculate similarity scores for all images and all keywords.

    Parameters:
        image_paths (list): List of image file paths.
        keywords (list): List of keywords to compute similarity.

    Returns:
        similarity_scores (torch.Tensor): Tensor of similarity scores, shape (num_images, num_keywords).
    """
    # Load the model and process images and keywords
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


def classification_score(image_paths, preprocess, class_n, model):
    """
    Calculate classification scores for all images.

    Parameters:
        image_paths (list): List of image file paths.

    Returns:
        probabilities (list): List of classification probabilities.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = torch.load("model/" + model, map_location=device)
    model = model.to(device)
    model.eval()

    # Perform inference
    probabilities = []
    with torch.no_grad():
        for image in tqdm(image_paths, desc="Classifying images"):
            images = Image.open(image).convert("RGB")
            images = preprocess(images).unsqueeze(0).to(device)
            outputs = model(images)
            probabilities.extend(torch.softmax(outputs, dim=1)[:, class_n].cpu().numpy())

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


def plot_roc_curves(subgroup_data, keywords, clip_scores, fig_path):
    """
    Generate and plot ROC curves for each subgroup (keyword).

    Parameters:
        subgroup_data (dict): Dictionary where keys are keywords and values are tuples of (predicted_scores, true_labels).
        keywords (list): List of keywords to plot ROC curves for.
        clip_scores (dict): Dictionary where keys are keywords and values are CLIP scores.
        fig_path (str): Path to save the figure.
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

    fig_path = fig_path + "_roc_plot.pdf"
    plt.savefig(f"{fig_path}", dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")


def plot_correlation(subgroup_data, keywords, clip_scores, fig_path):
    """
    Generate and plot correlation between AUC and CLIP scores.

    Parameters:
        subgroup_data (dict): Dictionary where keys are keywords and values are tuples of (predicted_scores, true_labels).
        keywords (list): List of keywords to plot correlation for.
        clip_scores (dict): Dictionary where keys are keywords and values are CLIP scores.
        fig_path (str): Path to save the figure.
    """
    plt.figure(figsize=(6, 5))

    auc_scores = []
    for keyword in keywords:
        if keyword not in subgroup_data:
            print(f"Keyword '{keyword}' not found in data. Skipping.")
            continue

        predicted_scores, true_labels = subgroup_data[keyword]
        _, _, roc_auc = compute_roc_and_auc(predicted_scores, true_labels)
        auc_scores.append((clip_scores[keyword], roc_auc))

    auc_scores = np.array(auc_scores)
    clip_scores_array = auc_scores[:, 0]
    auc_values = auc_scores[:, 1]

    slope, intercept, _, _, _ = linregress(clip_scores_array, auc_values)
    x_range = np.linspace(np.min(clip_scores_array), np.max(clip_scores_array), 100)
    y_range = slope * x_range + intercept

    y_pred = slope * clip_scores_array + intercept
    residuals = auc_values - y_pred
    mse = np.mean(residuals ** 2)
    ci = 1.96 * np.sqrt(
        mse * (1 / len(clip_scores_array)
        + (x_range - np.mean(clip_scores_array))**2
        / np.sum((clip_scores_array - np.mean(clip_scores_array))**2))
    )

    plt.scatter(clip_scores_array, auc_values, s=50, color="teal", alpha=0.8)
    plt.plot(x_range, y_range, color="gray", linestyle="--", linewidth=2)
    plt.xlabel("CLIP Score", fontsize=14)
    plt.ylabel("AUC Score", fontsize=14)
    plt.title("Correlation between AUC and CLIP Score", fontsize=14)

    plt.fill_between(x_range, y_range - ci, y_range + ci, color="gray", alpha=0.2)

    plt.text(
        x=np.max(clip_scores_array) - 1.15,
        y=np.max(auc_values) - 0.01,
        s=f"coef = {slope:.3f}",
        fontsize=12,
        color="black",
    )

    plt.tight_layout()

    fig_path = fig_path + "_AUC_clip_plot.pdf"
    plt.savefig(f"{fig_path}", dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")


def sample_roc_keywords(keywords, clip_scores):
    """
    Sample a subset of keywords for ROC curve plotting.

    Parameters:
        keywords (list): List of keywords to sample from.
        num_keywords (int): Number of keywords to sample.

    Returns:
        roc_keywords (list): List of sampled keywords.
    """
    np.random.seed(42)
    keywords = np.array(keywords)
    high_scores = np.random.choice(keywords[:5], 2, replace=False)
    low_scores = np.random.choice(keywords[-5:], 2, replace=False)
    mid_scores = np.random.choice(keywords[5:-5], 1, replace=False)

    roc_keywords = np.concatenate((high_scores, mid_scores, low_scores))
    roc_keywords = sorted(roc_keywords, key=lambda x: clip_scores[x], reverse=True)

    return roc_keywords


def load_specifications(dataset_name, class_label):
    match dataset_name:
        case 'imagenet' | 'imagenet-r' | 'imagenet-c':
            imagenet_idx = imagenet.Indexer("data/imagenet_variants/label_mapping.csv")
            class_ids = ["n02071294", "n02219486", "n03535780", "n03642806", "n03781244", "n04317175"]

            class_n = next(
                (imagenet_idx.name_to_n[imagenet_idx.id_to_name[class_id]]
                for class_id in class_ids
                if class_label in imagenet_idx.id_to_name[class_id].split(",")),
                None
            )

            if class_n is None:
                raise ValueError(f"Class '{class_label}' not found in valid ImageNet classes.")

            model = "imagenet-resnet50"
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imagenet.MEAN, imagenet.STD)
            ])

        case 'waterbird':
            class_names = ['landbird', 'waterbird']
            class_n = class_names.index(class_label)

            model = "best_model_Waterbirds_erm.pth"
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(waterbirds.MEAN, waterbirds.STD)
            ])

        case 'celeba':
            class_names = ['not blond', 'blond']
            class_n = class_names.index(class_label)

            model = "best_model_CelebA_erm.pth"
            transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(celeba.MEAN, celeba.STD),
            ])

    return transform, class_n, model


def plot_roc_figure(args):
    print(f"Analyzing dataset: {args.dataset}, class: {args.class_label}")
    transform, class_n, model = load_specifications(args.dataset, args.class_label)
    results_csv = f"result/{args.dataset}_{model.split('.')[0]}.csv"
    b2t_csv = f"diff/{args.dataset}_{model.split('.')[0]}_{args.class_label}.csv"
    fig_path = "plotting/plots/" + args.dataset + "_" + args.class_label

    # Load in the csv files
    results = pd.read_csv(results_csv)
    b2t_scores = pd.read_csv(b2t_csv)

    # Clip scores per keyword + image subset
    keywords = b2t_scores.groupby("Keyword")["Score"].mean()
    keywords = keywords.sort_values(ascending=False).index.tolist()
    clip_scores = b2t_scores.loc[b2t_scores["Keyword"].isin(keywords), "Score"].values
    clip_scores = {keyword: score for keyword, score in zip(keywords, clip_scores)}
    images = results.loc[results["actual"] == class_n, "image"].to_list()
    print(f"Completed loading {len(images)} images")

    # TODO: temp
    images = ["data/cub/data/waterbird_complete95_forest2water2/" + image for image in images]

    # Compute scores and determine subgroups
    similarity_scores = similarity_func(images, keywords)
    classifier_scores = classification_score(images, transform, class_n, model)
    subgroup_labels = assign_to_subgroups(similarity_scores, keywords, thresholding_method="mean")

    # Prepare data for plotting
    subgroup_data = {keyword: (classifier_scores, subgroup_labels[keyword][1]) for keyword in keywords}

    roc_keywords = sample_roc_keywords(keywords, clip_scores)
    plot_roc_curves(subgroup_data, roc_keywords, clip_scores, fig_path)
    plot_correlation(subgroup_data, keywords, clip_scores, fig_path)