import argparse
from zeroshot import clip_zeroshot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='celeba', choices=['celeba', 'waterbirds'])
    parser.add_argument('--data_dir', default='/data')
    parser.add_argument('--use_keywords', action="store_true")

    args = parser.parse_args()
    clip_zeroshot.main(args)