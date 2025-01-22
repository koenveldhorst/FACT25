import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from function.calculate_similarity import calc_similarity
import pandas as pd

def compute_roc_and_auc(similarity_scores, correct_labels):
    """
    Compute ROC curve and AUC for a given set of similarity scores and true labels.

    Parameters:
        similarity_scores (np.ndarray): Array of similarity scores.
        correct_labels (np.ndarray): Array of binary labels (1 for correct, 0 for incorrect).

    Returns:
        fpr (np.ndarray): False positive rates.
        tpr (np.ndarray): True positive rates.
        roc_auc (float): Area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(correct_labels, similarity_scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_roc_curves(similarity_data, keywords, csv_file_path):
    """
    Generate and plot ROC curves for each subgroup (keyword).

    Parameters:
        similarity_data (dict): Dictionary where keys are keywords and values are tuples of (similarity_scores, correct_labels).
        keywords (list): List of keywords to plot ROC curves for.
    """
    plt.figure(figsize=(10, 8))

    for keyword in keywords:
        if keyword not in similarity_data:
            print(f"Keyword '{keyword}' not found in data. Skipping.")
            continue

        similarity_scores, correct_labels = similarity_data[keyword]
        fpr, tpr, roc_auc = compute_roc_and_auc(similarity_scores, correct_labels)

        plt.plot(fpr, tpr, label=f"{keyword} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line for random guessing
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves for Subgroups", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    fig_path = "plots/" + csv_file_path.split(".")[0].split("_")[0] + "_roc_plot.png"
    plt.savefig(f"{fig_path}", dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")


# Integration with existing setup
def calculate_and_plot_auroc(image_dir, results_csv, keywords):
    """
    Integrate AUROC calculation and plotting with the existing setup.

    Parameters:
        image_dir (str): Directory containing images.
        results_csv (str): Path to the CSV file with classification results.
        keywords (list): List of keywords for subgroup analysis.
        calc_similarity_func (function): Function to calculate similarity scores.
    """
    # Load classification results
    results = pd.read_csv(results_csv)

    # Prepare data for similarity calculation
    similarity_data = {}

    for keyword in keywords:
        # Filter images containing the current keyword
        filtered_images = results[results["caption"].str.contains(keyword, case=False, na=False)]

        if filtered_images.empty:
            print(f"No images found for keyword '{keyword}'. Skipping.")
            continue

        # Calculate similarity scores
        similarity_scores = calc_similarity(image_dir, filtered_images["image"].tolist(), [keyword])

        # Store similarity scores and correct labels
        similarity_data[keyword] = (similarity_scores.cpu().numpy(), filtered_images["correct"].to_numpy())

    # Plot ROC curves
    plot_roc_curves(similarity_data, keywords, results_csv)

# Example usage
# image_dir = "path/to/image/directory"
# results_csv = "path/to/results.csv"
# keywords = ["bamboo", "forest", "water", "species", "bird"]
# calculate_and_plot_auroc(image_dir, results_csv, keywords)
if __name__ == "__main__":
    image_dir = "data/cub/data/waterbird_complete95_forest2water2/"
    results_csv = "result/waterbird_best_model_Waterbirds_erm.csv"
    keywords = ["bamboo", "forest", "water", "species", "bird"]
    calculate_and_plot_auroc(image_dir, results_csv, keywords)