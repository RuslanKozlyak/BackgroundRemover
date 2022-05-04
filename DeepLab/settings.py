import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import get_loaders

LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
PERSON_IMG_DIR = r"D:/datasets/people_segmentation/images"
PERSON_MASK_DIR = r"D:/datasets/people_segmentation/masks"

TRAIN_IMG_FILE = r"D:/datasets/people_segmentation/segmentation/train.txt"
VAL_IMG_FILE = r"D:/datasets/people_segmentation/segmentation/val.txt"

train_transform = A.Compose(
    [
        A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH,p=1.0),
        A.HorizontalFlip(p=0.3),
        A.ChannelShuffle(p=0.3),
        A.CoarseDropout(p=0.3, min_holes=3, max_holes=10, max_height=32, max_width=32),
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

train_loader, val_loader = get_loaders(
    PERSON_IMG_DIR,
    PERSON_MASK_DIR,
    PERSON_IMG_DIR,
    PERSON_MASK_DIR,
    BATCH_SIZE,
    train_transform,
    val_transforms,
    NUM_WORKERS,
    PIN_MEMORY,
)