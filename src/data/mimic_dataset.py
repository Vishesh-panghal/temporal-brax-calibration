"""PyTorch Dataset implementation for MIMIC-CXR-JPG chest radiographs."""

import os
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_TARGETS = ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema']


def get_mimic_eval_transforms(image_size: int = 512) -> Callable:
    """Standard evaluation image transforms matching BRAX 512x512 pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class MimicCxrDataset(Dataset):
    """
    Dataset for MIMIC-CXR-JPG chest radiographs.
    Loads frontal test images and provides standardized tensor representations.
    """

    def __init__(
        self,
        manifest: Union[str, pd.DataFrame],
        image_root: str = "data/mimic/images",
        image_size: int = 512,
        targets: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        view_position: Optional[str] = None,
        allow_synthetic: bool = False,
    ):
        super().__init__()
        if isinstance(manifest, str):
            self.df = pd.read_csv(manifest)
        else:
            self.df = manifest.copy()

        if view_position is not None:
            self.df = self.df[self.df["ViewPosition"] == view_position].reset_index(drop=True)

        self.image_root = image_root
        self.image_size = image_size
        self.targets = targets or DEFAULT_TARGETS
        self.transform = transform or get_mimic_eval_transforms(image_size=image_size)
        self.allow_synthetic = allow_synthetic

        # Prepare label matrix
        label_df = self.df[self.targets].fillna(0.0)
        self.labels = label_df.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        row = self.df.iloc[idx]
        rel_path = str(row.get("relative_path", row.get("PngPath", "")))

        if self.image_root:
            img_path = os.path.join(self.image_root, rel_path)
        else:
            img_path = rel_path

        if os.path.exists(img_path) and os.path.isfile(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                raise IOError(f"Corrupted MIMIC image file at {img_path}: {str(e)}")
        else:
            if self.allow_synthetic:
                # Allowed only for offline dry-run testing
                image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
            else:
                raise FileNotFoundError(
                    f"Required radiograph not found at '{img_path}'. "
                    "Ensure images are downloaded to image_root via scripts/download_mimic_images.py."
                )

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        meta = {
            "patient_id": str(row.get("subject_id", "")),
            "study_id": str(row.get("study_id", "")),
            "dicom_id": str(row.get("dicom_id", "")),
            "view_position": str(row.get("ViewPosition", "")),
            "relative_path": rel_path,
        }

        return image_tensor, label_tensor, meta
