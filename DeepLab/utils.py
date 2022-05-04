import os

import numpy as np
import torch
import torchvision
from PIL import Image
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CarvanaDataset
from model import DeepLab
from settings import DEVICE
import albumentations as A


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        path= r"D:/datasets/people_segmentation/segmentation/train.txt",
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        path= r"D:/datasets/people_segmentation/segmentation/val.txt",
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    print("=> Checking accuracy")
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    loop = tqdm(loader)
    with torch.no_grad():
        for x, y in loop:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            del x
            preds = (preds > 0.5).float()
            if y.shape != preds.shape:
                y = torchvision.transforms.functional.resize(y, size=preds.shape[2:])
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )
            del y
            del preds

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()
    return dice_score


def save_predictions_as_imgs(loader, model, folder, device):
    print("=> Saving predictions as imgs")

    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    loop = tqdm(loader)

    for idx, (x, y) in enumerate(loop):
        if idx % 100 == 0:
            x = x.to(device=device)
            with torch.no_grad():
                preds = model(x)
                del x
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            del preds
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
            del y
    model.train()

def get_masked_image(path, threshhold=0.6):
    model = DeepLab().cuda()
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    with Image.open(path) as im:
        x = np.array(im.convert("RGB"))

    transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    augmentations = transform(image=x)
    x = augmentations["image"]
    x = x.float().unsqueeze(0).to(device=DEVICE)

    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > threshhold).float()

    torchvision.utils.save_image(
        x, f"before.png"
    )

    preds = preds[0]
    x = x[0]

    for i in range(3):
        x[i] *= preds[0]

    torchvision.utils.save_image(
        x, f"back.png"
    )
