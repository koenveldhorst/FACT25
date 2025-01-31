from plotting.keyword_roc_curve import plot_roc_figure
from plotting.keyword_clipscore import plot_clip_figure
import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROC curves for subgroups.")
    parser.add_argument("--dataset", type=str, default="waterbird", help="Dataset to use for analysis.")
    parser.add_argument("--class_label", type=str, default="waterbird", help="Class to analyze.")
    args = parser.parse_args()
    return args


def get_model_name(dataset_name):
    match dataset_name:
        case 'imagenet' | 'imagenet-r' | 'imagenet-c':
            model = "imagenet-resnet50"

        case 'waterbird':
            model = "best_model_Waterbirds_erm.pth"

        case 'celeba':
            model = "best_model_CelebA_erm.pth"

    return model


def sample_subset_keywords(keywords, clip_scores):
    """
    Sample a subset of keywords for ROC curve plotting.

    Parameters:
        keywords (list): List of keywords to sample from.
        num_keywords (int): Number of keywords to sample.

    Returns:
        subset_keywords (list): List of sampled keywords.
    """
    keywords = np.array(keywords)
    high_scores = keywords[:2]
    low_scores = keywords[-2:]
    mid_scores = [keywords[len(keywords) // 2]]

    subset_keywords = np.concatenate((high_scores, mid_scores, low_scores))
    subset_keywords = sorted(subset_keywords, key=lambda x: clip_scores[x], reverse=True)

    return subset_keywords


def load_keywords(args):
    model = get_model_name(args.dataset)
    b2t_csv = f"diff/{args.dataset}_{model.split('.')[0]}_{args.class_label}.csv"
    b2t_scores = pd.read_csv(b2t_csv)

    keywords = b2t_scores.groupby("Keyword")["Score"].mean()
    keywords = keywords.sort_values(ascending=False).index.tolist()
    clip_scores = b2t_scores.loc[b2t_scores["Keyword"].isin(keywords), "Score"].values
    clip_scores = {keyword: score for keyword, score in zip(keywords, clip_scores)}

    subset_keywords = sample_subset_keywords(keywords, clip_scores)
    return keywords, subset_keywords, clip_scores


if __name__ == "__main__":
    args = parse_args()
    keywords, subset_keywords, clip_scores = load_keywords(args)

    plot_clip_figure(args, subset_keywords, clip_scores)
    plot_roc_figure(args, keywords, subset_keywords, clip_scores)
