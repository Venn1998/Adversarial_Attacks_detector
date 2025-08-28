from torchvision.models.resnet import BasicBlock, Bottleneck
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import torch.nn.functional as F
from torchvision.utils import save_image
import re
from typing import Literal, Dict, Any

DATASET_PATH = "../data"
imagenet_path = os.path.join(DATASET_PATH, "TinyImageNet/")
with open(os.path.join(imagenet_path, "label_list.json"), "r") as f:
    label_names = json.load(f)
NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])


def show_prediction(img, label, pred, K=5, adv_img=None, noise=None):

    if isinstance(img, torch.Tensor):
        # Tensor image to numpy
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * NORM_STD[None, None]) + NORM_MEAN[None, None]
        img = np.clip(img, a_min=0.0, a_max=1.0)
        label = label.item()

    # Plot on the left the image with the true label as title.
    # On the right, have a horizontal bar plot with the top k predictions including probabilities
    if noise is None or adv_img is None:
        fig, ax = plt.subplots(1, 2, figsize=(
            10, 2), gridspec_kw={'width_ratios': [1, 1]})
    else:
        fig, ax = plt.subplots(1, 5, figsize=(12, 2), gridspec_kw={
                               'width_ratios': [1, 1, 1, 1, 2]})

    ax[0].imshow(img)
    ax[0].set_title(label_names[label])
    ax[0].axis('off')

    if adv_img is not None and noise is not None:
        # Visualize adversarial images
        adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
        adv_img = (adv_img * NORM_STD[None, None]) + NORM_MEAN[None, None]
        adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
        ax[1].imshow(adv_img)
        ax[1].set_title('Adversarial')
        ax[1].axis('off')
        # Visualize noise
        noise = noise.cpu().permute(1, 2, 0).numpy()
        noise = noise * 0.5 + 0.5  # Scale between 0 to 1
        ax[2].imshow(noise)
        ax[2].set_title('Noise')
        ax[2].axis('off')
        # buffer
        ax[3].axis('off')

    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    ax[-1].barh(np.arange(K), topk_vals*100.0, align='center',
                color=["C0" if topk_idx[i] != label else "C2" for i in range(K)])
    ax[-1].set_yticks(np.arange(K))
    ax[-1].set_yticklabels([label_names[c] for c in topk_idx])
    ax[-1].invert_yaxis()
    ax[-1].set_xlabel('Confidence')
    ax[-1].set_title('Predictions')

    plt.show()
    plt.close()


def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):
    device = imgs.device
    # Determine prediction of the model
    inp_imgs = imgs.clone().requires_grad_()
    preds = model(inp_imgs.to(device))
    preds = F.log_softmax(preds, dim=-1)
    # Calculate loss by NLL
    loss = -torch.gather(preds, 1, labels.to(device).unsqueeze(dim=-1))
    loss.sum().backward()
    # Update image to adversarial example as written above
    noise_grad = torch.sign(inp_imgs.grad.to(imgs.device))
    fake_imgs = imgs + epsilon * noise_grad
    fake_imgs.detach_()
    return fake_imgs, noise_grad


def save_adversarial_images(adv_examples, labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = adv_examples[0].device

    # Create tensors for mean and std, ensuring they are on the same device as the images
    mean = torch.tensor(NORM_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(NORM_STD, device=device).view(3, 1, 1)

    for i, (adv_img, label) in enumerate(zip(adv_examples, labels)):
        # Create a subdirectory for the class label
        class_dir = os.path.join(output_dir, str(label.item()))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # 1. Denormalize the image
        denorm_img = adv_img * std + mean

        # 2. Clip the image values to be in the valid [0, 1] range
        clipped_img = torch.clamp(denorm_img, 0, 1)

        # 3. Save the corrected, displayable image
        img_path = os.path.join(class_dir, f'adv_image_{i:04d}.png')
        save_image(clipped_img, img_path)


def eval_model(dataset_loader, model, img_func=None):
    device = model.parameters().__next__().device
    tp, tp_5, counter = 0., 0., 0.
    for imgs, labels in tqdm(dataset_loader, desc="Validating..."):
        imgs = imgs.to(device)
        labels = labels.to(device)
        if img_func is not None:
            imgs = img_func(imgs, labels)
        with torch.no_grad():
            preds = model(imgs)
        tp += (preds.argmax(dim=-1) == labels).sum()
        tp_5 += (preds.topk(5, dim=-1)[1] ==
                 labels[..., None]).any(dim=-1).sum()
        counter += preds.shape[0]
    acc = tp/counter
    top5 = tp_5/counter
    print(f"Top-1 error: {(100.0 * (1 - acc)):4.2f}%")
    print(f"Top-5 error: {(100.0 * (1 - top5)):4.2f}%")
    return acc, top5


def get_resnet_blocks(model):
    layers_to_probe = {}
    for name, module in model.named_modules():
        if isinstance(module, (BasicBlock, Bottleneck)):
            layers_to_probe[name] = module
    return layers_to_probe
