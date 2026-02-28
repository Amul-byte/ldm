"""Dataset utilities with synthetic fallback for all training stages."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_model.util import DEFAULT_JOINTS, DEFAULT_NUM_CLASSES, DEFAULT_WINDOW, assert_shape


class SyntheticGaitDataset(Dataset):
    """Synthetic dataset producing skeleton, IMU streams, and labels."""

    def __init__(
        self,
        length: int = 128,
        window: int = DEFAULT_WINDOW,
        joints: int = DEFAULT_JOINTS,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.length = length
        self.window = window
        self.joints = joints
        self.num_classes = num_classes

    def __len__(self) -> int:
        """Return dataset length."""
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one synthetic sample dictionary."""
        del idx
        skeleton = torch.randn(self.window, self.joints, 3)
        hip = torch.randn(self.window, 6)
        wrist = torch.randn(self.window, 6)
        label = torch.randint(low=0, high=self.num_classes, size=(1,)).squeeze(0)
        assert_shape(skeleton, [self.window, self.joints, 3], "SyntheticGaitDataset.skeleton")
        assert_shape(hip, [self.window, 6], "SyntheticGaitDataset.hip")
        assert_shape(wrist, [self.window, 6], "SyntheticGaitDataset.wrist")
        return {"skeleton": skeleton, "hip": hip, "wrist": wrist, "label": label}


class TorchFileGaitDataset(Dataset):
    """Dataset reading `.pt`/`.pth` files containing tensor dictionaries."""

    def __init__(self, path: str, window: int = DEFAULT_WINDOW, joints: int = DEFAULT_JOINTS) -> None:
        super().__init__()
        self.window = window
        self.joints = joints
        payload = torch.load(path, map_location="cpu")
        self.skeleton = payload["skeleton"].float()
        self.hip = payload["hip"].float()
        self.wrist = payload["wrist"].float()
        self.label = payload["label"].long()

        assert_shape(self.skeleton, [None, window, joints, 3], "TorchFileGaitDataset.skeleton")
        assert_shape(self.hip, [self.skeleton.shape[0], window, 6], "TorchFileGaitDataset.hip")
        assert_shape(self.wrist, [self.skeleton.shape[0], window, 6], "TorchFileGaitDataset.wrist")
        assert_shape(self.label, [self.skeleton.shape[0]], "TorchFileGaitDataset.label")

    def __len__(self) -> int:
        """Return number of samples."""
        return self.skeleton.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one sample dictionary."""
        return {
            "skeleton": self.skeleton[idx],
            "hip": self.hip[idx],
            "wrist": self.wrist[idx],
            "label": self.label[idx],
        }


def create_dataloader(
    dataset_path: Optional[str],
    batch_size: int,
    shuffle: bool = True,
    synthetic_length: int = 128,
    window: int = DEFAULT_WINDOW,
    joints: int = DEFAULT_JOINTS,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> DataLoader:
    """Create dataloader from a dataset path or synthetic fallback mode."""
    if dataset_path:
        dataset = TorchFileGaitDataset(path=dataset_path, window=window, joints=joints)
    else:
        dataset = SyntheticGaitDataset(
            length=synthetic_length,
            window=window,
            joints=joints,
            num_classes=num_classes,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
