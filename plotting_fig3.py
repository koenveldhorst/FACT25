from plotting.keyword_roc_curve import plot_roc_figure
from plotting.keyword_clipscore import plot_clip_figure
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROC curves for subgroups.")
    parser.add_argument("--dataset", type=str, default="waterbird", help="Dataset to use for analysis.")
    parser.add_argument("--class_label", type=str, default="waterbird", help="Class to analyze.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    plot_roc_figure(args)
    # plot_clip_figure(args)
