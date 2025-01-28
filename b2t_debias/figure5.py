import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import clip

from sklearn.metrics import classification_report
from tqdm import tqdm

from data.celeba import CelebA
from data.waterbirds import Waterbirds

import celeba_templates
import waterbirds_templates
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def main(args):
    model, preprocess = clip.load('RN50', jit=False)  # RN50, RN101, RN50x4, ViT-B/32

    crop = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    transform = transforms.Compose([crop, preprocess])
    # print('hello')

    # load dataset
    if args.dataset == 'waterbirds':
        data_dir = os.path.join(args.data_dir, 'waterbird_complete95_forest2water2')
        val_dataset = Waterbirds(data_dir=data_dir, split='val', transform=transform)
        templates = waterbirds_templates.templates
        class_templates = waterbirds_templates.class_templates
        class_keywords_all = waterbirds_templates.class_keywords_all
    elif args.dataset == 'celeba':
        data_dir = os.path.join(args.data_dir, 'celeba')
        val_dataset = CelebA(data_dir=data_dir, split='val', transform=transform)
        templates = celeba_templates.templates
        class_templates = celeba_templates.class_templates
        class_keywords_all = celeba_templates.class_keywords_all
    else:
        raise NotImplementedError

    # print('hello')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)
    temperature = 0.02  # redundant parameter
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # get average CLIP embedding from multiple template prompts
    with torch.no_grad():
        zeroshot_weights = []
        for class_keywords in class_keywords_all:
            print('hi')
            texts = [template.format(class_template.format(class_keyword)) for template in templates for class_template in class_templates for class_keyword in class_keywords]
            texts = clip.tokenize(texts)

            class_embeddings = model.encode_text(texts.to(device))
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    # run CLIP zero-shot classifier
    landbird_pred = []
    landbird_actual = []
    waterbird_pred = []
    waterbird_actual = []
    celeba_pred = []
    celeba_actual = []

    print('hii')
    with torch.no_grad():
        for (image, (target, target_g, target_s), _) in tqdm(val_dataloader):
            image = image.cuda()
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ zeroshot_weights / temperature

            # get classifier predictions
            probs = logits.softmax(dim=-1).cpu()
            conf, pred = torch.max(probs, dim=1)

            if args.dataset == 'waterbirds':
                # minor group if
                # (target, target_s) == (0, 1): landbird on water background
                # (target, target_s) == (1, 0): waterbird on land background
                is_minor_pred = (((target == 0) & (pred == 1)) | ((target == 1) & (pred == 0))).long()
                is_minor = (((target == 0) & (target_s == 1)) | ((target == 1) & (target_s == 0))).long()
                
                landbird_minor_pred = (1-target) * probs[:, 1]
                waterbird_minor_pred = target * probs[:, 0]
                landbird_minor = (((target == 0) & (target_s == 1))).long()
                waterbird_minor = (((target == 1) & (target_s == 0))).long()

                landbird_pred.append(landbird_minor_pred)
                landbird_actual.append(landbird_minor)
                waterbird_pred.append(waterbird_minor_pred)
                waterbird_actual.append(waterbird_minor)
            if args.dataset == 'celeba':
                # minor group if
                # (target, target_s) == (1, 1): blond man
                is_minor_pred = ((target == 1) & (pred == 1)).long()
                is_minor = ((target == 1) & (target_s == 1)).long()

                celeba_minor_pred = target * probs[:, 1]

                celeba_pred.append(celeba_minor_pred)
                celeba_actual.append(is_minor)

    if args.dataset == 'waterbirds':
        landbird_pred_tensor = (torch.cat(landbird_pred, dim=0)).flatten()
        landbird_true_tensor = (torch.cat(landbird_actual, dim=0)).flatten()
        waterbird_pred_tensor = (torch.cat(waterbird_pred, dim=0)).flatten()
        waterbird_true_tensor = (torch.cat(waterbird_actual, dim=0)).flatten()

        fpr, tpr, thresholds = roc_curve(landbird_true_tensor.numpy(), landbird_pred_tensor.numpy())
        fpr1, tpr1, thresholds1 = roc_curve(waterbird_true_tensor.numpy(), waterbird_pred_tensor.numpy())

        # Plot ROC curve for landbirds
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Landbirds')
        plt.legend(loc='lower right')
        plt.savefig('roc_curves_landbirds.png')

        # Plot ROC curve for Waterbirds
        plt.figure()
        plt.plot(fpr1, tpr1, color='blue', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Waterbirds')
        plt.legend(loc='lower right')
        plt.savefig('roc_curves_waterbirds.png')

    elif args.dataset == 'celeba':
        celeba_pred_tensor = (torch.cat(celeba_pred, dim=0)).flatten()
        celeba_true_tensor = (torch.cat(celeba_actual, dim=0)).flatten()

        fpr, tpr, thresholds = roc_curve(celeba_true_tensor.numpy(), celeba_pred_tensor.numpy())

        # Plot ROC curve for landbirds
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for CelebA blond')
        plt.legend(loc='lower right')
        plt.savefig('roc_curves_celeba.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='celeba', choices=['celeba', 'waterbirds'])
    parser.add_argument('--data_dir', default='/data')
    parser.add_argument('--save_path', default='./pseudo_bias/celeba.pt')

    args = parser.parse_args()
    main(args)
