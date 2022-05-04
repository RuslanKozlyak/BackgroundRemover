import os
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
import albumentations as A


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, path, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        with open(path) as f:
            lines = f.read().splitlines()
        self.images = lines

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.images[index] + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            try:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            except:
                transforms = A.Compose(
                    [
                        A.Resize(height=512, width=512,p=1),
                        A.Normalize(
                            mean=[0.0, 0.0, 0.0],
                            std=[1.0, 1.0, 1.0],
                            max_pixel_value=255.0,
                        ),
                        ToTensorV2(),
                    ],
                )
                augmentations = transforms(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

        return image, mask