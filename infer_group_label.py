import argparse
from b2t_debias import infer_group_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='celeba', choices=['celeba', 'waterbirds'])
    parser.add_argument('--data_dir', default='/data')
    parser.add_argument('--save_path', default='./b2t_debias/pseudo_bias/celeba.pt')

    args = parser.parse_args()
    infer_group_label.main(args)
