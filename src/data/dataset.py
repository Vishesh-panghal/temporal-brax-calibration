"""PyTorch Dataset implementation for BRAX chest radiographs."""

import os
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Allow large medical radiographs without DecompressionBombError
import torch
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_TARGETS = ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema']


def get_default_transforms(image_size: int = 512, is_training: bool = True) -> Callable:
    """Standard image transforms for chest radiographs with configurable resolution."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class BraxDataset(Dataset):
    """Dataset for BRAX chest radiography studies with strict integrity checks."""

    def __init__(
        self,
        manifest: Union[str, pd.DataFrame],
        split: Optional[str] = None,
        temporal_bin: Optional[str] = None,
        image_root: str = "",
        image_size: int = 512,
        targets: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        uncertainty_policy: str = "u_zero",
        allow_synthetic: bool = False,
    ):
        super().__init__()
        if isinstance(manifest, str):
            self.df = pd.read_csv(manifest)
        else:
            self.df = manifest.copy()

        if split is not None:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        if temporal_bin is not None:
            self.df = self.df[self.df['temporal_bin'] == temporal_bin].reset_index(drop=True)

        self.image_root = image_root
        self.image_size = image_size
        self.targets = targets or DEFAULT_TARGETS
        self.transform = transform or get_default_transforms(image_size=image_size, is_training=(split == 'train'))
        self.uncertainty_policy = uncertainty_policy
        self.allow_synthetic = allow_synthetic

        # Prepare label matrix
        label_df = self.df[self.targets].copy()
        if self.uncertainty_policy == "u_zero":
            label_df = label_df.fillna(0.0)

        self.labels = label_df.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        row = self.df.iloc[idx]
        img_rel_path = str(row.get('PngPath', ''))

        # Construct full image path
        if self.image_root:
            img_path = os.path.join(self.image_root, img_rel_path)
        else:
            img_path = img_rel_path

        # Strict integrity check: research runs must load verified physical images
        if os.path.exists(img_path) and os.path.isfile(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                raise IOError(f"Corrupted image file at {img_path}: {str(e)}")
        else:
            if self.allow_synthetic:
                # Strictly permitted ONLY for offline unit tests/dry-runs
                image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
            else:
                raise FileNotFoundError(
                    f"Required radiograph not found at '{img_path}'. "
                    "Ensure image_root is set correctly in config.yaml or that images are transferred. "
                    "Synthetic placeholders are strictly prohibited in research runs."
                )

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        meta = {
            'patient_id': str(row.get('PatientID', '')),
            'study_date': str(row.get('StudyDate', '')),
            'split': str(row.get('split', '')),
            'temporal_bin': str(row.get('temporal_bin', '')),
            'view_position': str(row.get('ViewPosition', '')),
            'png_path': img_rel_path,
        }

        return image_tensor, label_tensor, meta
