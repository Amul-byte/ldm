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
        a_stream = torch.randn(self.window, 3)
        omega_stream = torch.randn(self.window, 3)
        label = torch.randint(low=0, high=self.num_classes, size=(1,)).squeeze(0)
        assert_shape(skeleton, [self.window, self.joints, 3], "SyntheticGaitDataset.skeleton")
        assert_shape(a_stream, [self.window, 3], "SyntheticGaitDataset.A")
        assert_shape(omega_stream, [self.window, 3], "SyntheticGaitDataset.Omega")
        return {
            "skeleton": skeleton,
            "A": a_stream,
            "Omega": omega_stream,
            "label": label,
            "fps": torch.tensor(30, dtype=torch.long),
        }


class TorchFileGaitDataset(Dataset):
    """Dataset reading `.pt`/`.pth` files containing tensor dictionaries."""

    def __init__(self, path: str, window: int = DEFAULT_WINDOW, joints: int = DEFAULT_JOINTS) -> None:
        super().__init__()
        self.window = window
        self.joints = joints
        payload = torch.load(path, map_location="cpu")
        self.skeleton = payload["skeleton"].float()
        self.A = payload["A"].float()
        self.Omega = payload["Omega"].float()
        self.label = payload["label"].long()
        self.fps = int(payload["fps"])
        sensor_identity = payload["sensor_identity"]
        assert sensor_identity["A"] == "right_hip", "A stream must correspond to right_hip"
        assert sensor_identity["Omega"] == "left_wrist", "Omega stream must correspond to left_wrist"
        assert self.fps == 30, f"SmartFallMM constraint violated: expected 30 FPS, got {self.fps}"

        assert_shape(self.skeleton, [None, window, joints, 3], "TorchFileGaitDataset.skeleton")
        assert_shape(self.A, [self.skeleton.shape[0], window, 3], "TorchFileGaitDataset.A")
        assert_shape(self.Omega, [self.skeleton.shape[0], window, 3], "TorchFileGaitDataset.Omega")
        assert_shape(self.label, [self.skeleton.shape[0]], "TorchFileGaitDataset.label")

    def __len__(self) -> int:
        """Return number of samples."""
        return self.skeleton.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one sample dictionary."""
        return {
            "skeleton": self.skeleton[idx],
            "A": self.A[idx],
            "Omega": self.Omega[idx],
            "label": self.label[idx],
            "fps": torch.tensor(self.fps, dtype=torch.long),
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
