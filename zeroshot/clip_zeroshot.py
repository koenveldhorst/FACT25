import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import clip
from tqdm import tqdm

from data import waterbirds, celeba

from zeroshot import celeba_templates
from zeroshot import waterbirds_templates

def main(args):
    model, preprocess = clip.load('RN50', jit=False)  # RN50, RN101, RN50x4, ViT-B/32

    crop = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    transform = transforms.Compose([crop, preprocess])

    # load dataset
    if args.dataset == 'waterbirds':
        data_dir = args.data_dir
        train_dataset = waterbirds.Waterbirds(root=data_dir, split='train', transform=transform)
        templates = waterbirds_templates.templates
        class_templates_all = waterbirds_templates.class_names_all
        class_keywords_all = waterbirds_templates.pos_class_keywords_all
    elif args.dataset == 'celeba':
        data_dir = args.data_dir
        train_dataset = celeba.CelebA(root=data_dir, split='train', transform=transform)
        templates = celeba_templates.templates
        class_templates_all = celeba_templates.class_names_all
        class_keywords_all = celeba_templates.class_keywords_all
    else:
        raise NotImplementedError

    train_dataloader = DataLoader(train_dataset, batch_size=256, num_workers=4, drop_last=False)
    temperature = 0.02  # redundant parameter

    # get average CLIP embedding from multiple template prompts
    with torch.no_grad():
        zeroshot_weights = []
        for class_templates, class_keywords in zip(class_templates_all, class_keywords_all):
            if args.use_keywords:
                texts = [
                    template.format(f"{class_template} in the {class_keyword}")
                        for template in templates
                            for class_template in class_templates
                                for class_keyword in class_keywords
                ]
            else:
                texts = [
                    template.format(class_template)
                        for template in templates
                            for class_template in class_templates
                ]
            
            texts = clip.tokenize(texts).cuda()

            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    # run CLIP zero-shot classifier
    total_correct, max_correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            image = batch["img"].cuda()
            target = batch["label"]

            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ zeroshot_weights / temperature

            # get classifier predictions
            probs = logits.softmax(dim=-1).cpu()
            _, pred = torch.max(probs, dim=1)

            total_correct += torch.sum((pred == target).long())
            max_correct()

    print(f"Accuracy: {total_correct / max_correct:.2f} ({total_correct} / {max_correct})")
