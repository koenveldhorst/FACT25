import pandas as pd
import matplotlib.pyplot as plt

def create_clip_plot(csv_file_path, keywords_to_display):
    """
    Function to create a CLIP score bar plot for specified keywords.

    Parameters:
        csv_file_path (str): Path to the CSV file containing the data.
        keywords_to_display (list): List of keywords to display in the plot.

    Returns:
        None
    """
    try:
        # Load the data from the CSV file
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file at {csv_file_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {csv_file_path} is empty or corrupted.")
        return

    # Check if 'Keyword' exists in the dataframe
    if 'Keyword' not in data.columns:
        print("Error: The column 'Keyword' does not exist in the provided data.")
        return

    # Filter the data to include only the specified keywords
    filtered_data = data[data['Keyword'].isin(keywords_to_display)]

    # Modify the keywords to include a trailing " -"
    filtered_data = filtered_data.copy()
    filtered_data["Keyword"] = filtered_data["Keyword"] + " -"

    # Sort the filtered data by CLIP score for better visualization
    filtered_data = filtered_data.sort_values(by="Score", ascending=False)

    # Plot the horizontal bar chart
    plt.figure(figsize=(6, 4))
    bars = plt.barh(
        filtered_data["Keyword"],
        filtered_data["Score"],
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
            fontsize=10,
        )

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
    plt.xticks(fontsize=10)
    plt.xlabel("CLIP score", fontsize=12)
    plt.title("CLIP Scores for Selected Keywords", fontsize=14)
    plt.gca().invert_yaxis()  # Invert Y-axis for top-to-bottom sorting
    plt.tight_layout()

    fig_path = "plots/" + csv_file_path.split(".")[0].split("_")[-1] + "_clip_plot.pdf"
    plt.savefig(f"{fig_path}", dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")


# Example usage:
# csv_file = "path/to/your/csvfile.csv"
# keywords = ["bamboo", "forest", "woods", "species", "bird"]
# create_clip_plot(csv_file, keywords)
if __name__ == "__main__":
    csv_file = "diff/waterbird_best_model_Waterbirds_erm_waterbird.csv"
    keywords_waterbird = ["bamboo", "forest", "woods", "species", "bird"]
    create_clip_plot(csv_file, keywords_waterbird)
