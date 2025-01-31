import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_clip_figure(args, keywords, clip_scores):
    """
    Function to create a CLIP score bar plot for specified keywords.

    Parameters:
        csv_file_path (str): Path to the CSV file containing the data.
        keywords_to_display (list): List of keywords to display in the plot.

    Returns:
        None
    """
    keywords_to_display = [keywords + " -" for keywords in keywords]
    scores = [clip_scores[keyword] for keyword in keywords]

    # Plot the horizontal bar chart
    plt.figure(figsize=(9, 7))
    bars = plt.barh(
        keywords_to_display,
        scores,
        color="teal",
        alpha=0.8,
        height=0.5  # Adjust bar width
    )

    # Add CLIP score values to the right of the bars
    for bar in bars:
        text_position = bar.get_width() + 0.05 if bar.get_width() > 0 else 0.05
        plt.text(
            text_position,  # Slightly to the right of the bar
            bar.get_y() + bar.get_height() / 2,  # Centered vertically
            f"{bar.get_width():.2f}",  # Format score with 2 decimal places
            va="center",
            fontsize=24,
        )

    plt.yticks(fontsize=24)

    # Customize plot appearance
    plt.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)  # Vertical zero-line
    plt.gca().spines["left"].set_visible(False)  # Remove Y-axis line
    plt.gca().spines["top"].set_visible(False)  # Remove the top border
    plt.gca().spines["right"].set_visible(False)  # Remove the right border
    plt.gca().yaxis.set_ticks_position("none")  # Remove horizontal dotted lines
    plt.gca().xaxis.set_tick_params(width=1.2)  # Adjust X-axis ticks
    plt.gca().xaxis.grid(True, linestyle="--", linewidth=1.0, alpha=0.6)  # Adjust vertical dotted lines
    plt.grid(visible=False, axis="y")  # Ensure no gridlines on Y-axis

    # Adjusting X-axis spacing
    plt.xticks(fontsize=20)
    plt.xlabel("CLIP score", fontsize=26)
    plt.title("CLIP Scores for Selected Keywords", fontsize=28, pad=20)
    plt.gca().invert_yaxis()  # Invert Y-axis for top-to-bottom sorting
    plt.tight_layout()

    fig_dir = f"plotting/plots/{args.dataset}"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = f"{fig_dir}/{args.class_label}_clip_plot.png"
    plt.savefig(f"{fig_path}", dpi=600, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")
