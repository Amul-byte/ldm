"""Dataset utilities with torch-file and paired CSV-folder modes."""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_NUM_CLASSES,
    DEFAULT_WINDOW,
    assert_shape,
    get_joint_labels,
    validate_joint_labels,
)

ACTIVITY_RE = re.compile(r"A(\d{2})", re.IGNORECASE)


def _parse_label_14(fname: str, num_classes: int = DEFAULT_NUM_CLASSES) -> int:
    """Parse activity code Axx from filename into zero-based label index."""
    match = ACTIVITY_RE.search(fname)
    if match is None:
        return 0
    activity = int(match.group(1))
    return max(0, min(num_classes - 1, activity - 1))


def _fill_nan_with_column_mean(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values with per-column means (or 0 if a full column is NaN)."""
    out = arr.astype(np.float32, copy=True)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D array, got {out.shape}")
    nan_mask = np.isnan(out)
    if nan_mask.any():
        col_mean = np.nanmean(out, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        out[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return out


def read_csv_files(folder: Optional[str]) -> Dict[str, pd.DataFrame]:
    """Read all CSV files from a folder into a filename->DataFrame map."""
    result: Dict[str, pd.DataFrame] = {}
    if folder is None or not os.path.isdir(folder):
        return result
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(folder, fname)
        try:
            result[fname] = pd.read_csv(path, header=None)
        except Exception:
            print(f"[dataset] Skipped empty/unreadable file: {fname}")
            continue
    return result


def _skeleton_frame_to_joints(frame_block: np.ndarray, joints: int = DEFAULT_JOINTS) -> np.ndarray:
    """Convert [T,96] or [T,97] skeleton rows into [T, J, 3]."""
    if frame_block.ndim != 2:
        raise ValueError(f"Skeleton input must be 2D, got {frame_block.shape}")
    if frame_block.shape[1] == joints * 3 + 1:
        frame_block = frame_block[:, 1:]
    if frame_block.shape[1] != joints * 3:
        raise ValueError(f"Expected {joints * 3} skeleton columns, got {frame_block.shape[1]}")
    return frame_block.reshape(frame_block.shape[0], joints, 3).astype(np.float32)


def _extract_sensor_accel3(df: pd.DataFrame) -> np.ndarray:
    """Extract tri-axial accelerometer columns from sensor CSV."""
    # SmartFallMM CSV schema: [timestamp_ms, iso_datetime, elapsed_s, acc_x, acc_y, acc_z].
    if df.shape[1] >= 6:
        accel_df = df.iloc[:, 3:6].apply(pd.to_numeric, errors="coerce")
        return _fill_nan_with_column_mean(accel_df.values.astype(np.float32))

    # Strict fallback: accept files containing only 3 accel columns.
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)]
    if numeric_df.shape[1] == 3:
        return _fill_nan_with_column_mean(numeric_df.values.astype(np.float32))
    raise ValueError(
        f"Sensor CSV schema mismatch: expected >=6 columns with accel at [3:6] or exactly 3 accel columns, got {df.shape[1]}"
    )


def _windowed(arr: np.ndarray, window: int, stride: int) -> Sequence[np.ndarray]:
    """Create sliding windows from a [T,...] array."""
    if arr.shape[0] < window:
        return []
    return [arr[s : s + window] for s in range(0, arr.shape[0] - window + 1, stride)]


class TorchFileGaitDataset(Dataset):
    """Dataset reading `.pt`/`.pth` files containing tensor dictionaries."""

    def __init__(self, path: str, window: int = DEFAULT_WINDOW, joints: int = DEFAULT_JOINTS) -> None:
        super().__init__()
        self.window = window
        self.joints = joints
        payload = torch.load(path, map_location="cpu")
        self.skeleton = payload["skeleton"].float()
        # Prefer explicit accel naming and keep legacy fallback support.
        self.A_hip = payload["A_hip"].float() if "A_hip" in payload else payload["A"].float()
        self.A_wrist = payload["A_wrist"].float() if "A_wrist" in payload else payload["Omega"].float()
        self.label = payload["label"].long()
        self.fps = int(payload["fps"])
        sensor_identity = payload.get("sensor_identity", {})
        joint_labels = payload["joint_labels"]
        validate_joint_labels(joint_labels)
        if sensor_identity:
            hip_id = sensor_identity.get("A_hip", sensor_identity.get("A", ""))
            wrist_id = sensor_identity.get("A_wrist", sensor_identity.get("Omega", ""))
            assert hip_id in {"meta_hip", "right_hip"}, "Hip accel stream must correspond to meta_hip"
            assert wrist_id in {"weta_wrist", "meta_wrist", "left_wrist"}, "Wrist accel stream must correspond to weta_wrist"
        assert self.fps == 30, f"SmartFallMM constraint violated: expected 30 FPS, got {self.fps}"

        assert_shape(self.skeleton, [None, window, joints, 3], "TorchFileGaitDataset.skeleton")
        assert_shape(self.A_hip, [self.skeleton.shape[0], window, 3], "TorchFileGaitDataset.A_hip")
        assert_shape(self.A_wrist, [self.skeleton.shape[0], window, 3], "TorchFileGaitDataset.A_wrist")
        assert_shape(self.label, [self.skeleton.shape[0]], "TorchFileGaitDataset.label")

    def __len__(self) -> int:
        """Return number of samples."""
        return self.skeleton.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one sample dictionary."""
        return {
            "skeleton": self.skeleton[idx],
            "A_hip": self.A_hip[idx],
            "A_wrist": self.A_wrist[idx],
            "label": self.label[idx],
            "fps": torch.tensor(self.fps, dtype=torch.long),
            "joint_labels": list(get_joint_labels()),
        }


class CSVPairedGaitDataset(Dataset):
    """Dataset built from three CSV folders: skeleton, hip accel, and wrist accel."""

    def __init__(
        self,
        skeleton_folder: str,
        hip_folder: str,
        wrist_folder: str,
        window: int = DEFAULT_WINDOW,
        joints: int = DEFAULT_JOINTS,
        stride: int = 30,
        num_classes: int = DEFAULT_NUM_CLASSES,
        normalize_sensors: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.window = window
        self.joints = joints
        self.fps = 30

        skeleton_map = read_csv_files(skeleton_folder)
        hip_map = read_csv_files(hip_folder)
        wrist_map = read_csv_files(wrist_folder)
        common = sorted(set(skeleton_map).intersection(hip_map).intersection(wrist_map))
        if not common:
            raise ValueError("No common CSV filenames across skeleton/hip/wrist folders")

        x_windows = []
        hip_windows = []
        wrist_windows = []
        labels = []
        skipped_parse = 0
        skipped_short = 0

        for fname in common:
            try:
                skel = _skeleton_frame_to_joints(_fill_nan_with_column_mean(skeleton_map[fname].values), joints=joints)
                hip = _extract_sensor_accel3(hip_map[fname])
                wrist = _extract_sensor_accel3(wrist_map[fname])
            except Exception:
                print(f"[dataset] Skipped invalid paired sample: {fname}")
                skipped_parse += 1
                continue

            t = min(skel.shape[0], hip.shape[0], wrist.shape[0])
            skel = skel[:t]
            hip = hip[:t]
            wrist = wrist[:t]

            skel_w = _windowed(skel, window, stride)
            hip_w = _windowed(hip, window, stride)
            wrist_w = _windowed(wrist, window, stride)
            n = min(len(skel_w), len(hip_w), len(wrist_w))
            if n == 0:
                skipped_short += 1
            for i in range(n):
                x_windows.append(skel_w[i])
                hip_windows.append(hip_w[i])
                wrist_windows.append(wrist_w[i])
                labels.append(_parse_label_14(fname, num_classes=num_classes))

        if len(x_windows) == 0:
            raise ValueError(
                "No valid paired windows found in CSV folders "
                f"(common_files={len(common)}, parse_failed={skipped_parse}, too_short_for_window={skipped_short}, "
                f"window={window}, stride={stride})"
            )

        self.skeleton = torch.tensor(np.stack(x_windows), dtype=torch.float32)
        self.A_hip = torch.tensor(np.stack(hip_windows), dtype=torch.float32)
        self.A_wrist = torch.tensor(np.stack(wrist_windows), dtype=torch.float32)
        self.label = torch.tensor(labels, dtype=torch.long)

        if normalize_sensors:
            hip_mean = self.A_hip.mean(dim=(0, 1), keepdim=True)
            hip_std = self.A_hip.std(dim=(0, 1), keepdim=True).clamp_min(eps)
            wrist_mean = self.A_wrist.mean(dim=(0, 1), keepdim=True)
            wrist_std = self.A_wrist.std(dim=(0, 1), keepdim=True).clamp_min(eps)
            self.A_hip = (self.A_hip - hip_mean) / hip_std
            self.A_wrist = (self.A_wrist - wrist_mean) / wrist_std

        assert_shape(self.skeleton, [None, window, joints, 3], "CSVPairedGaitDataset.skeleton")
        assert_shape(self.A_hip, [self.skeleton.shape[0], window, 3], "CSVPairedGaitDataset.A_hip")
        assert_shape(self.A_wrist, [self.skeleton.shape[0], window, 3], "CSVPairedGaitDataset.A_wrist")
        assert_shape(self.label, [self.skeleton.shape[0]], "CSVPairedGaitDataset.label")

    def __len__(self) -> int:
        """Return number of samples."""
        return self.skeleton.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one sample dictionary."""
        return {
            "skeleton": self.skeleton[idx],
            "A_hip": self.A_hip[idx],
            "A_wrist": self.A_wrist[idx],
            "label": self.label[idx],
            "fps": torch.tensor(self.fps, dtype=torch.long),
            "joint_labels": list(get_joint_labels()),
        }


def create_dataset(
    dataset_path: Optional[str],
    window: int = DEFAULT_WINDOW,
    joints: int = DEFAULT_JOINTS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    skeleton_folder: Optional[str] = None,
    hip_folder: Optional[str] = None,
    wrist_folder: Optional[str] = None,
    stride: int = 30,
    normalize_sensors: bool = True,
) -> Dataset:
    """Create dataset object from torch-file or paired CSV folders."""
    if dataset_path:
        return TorchFileGaitDataset(path=dataset_path, window=window, joints=joints)
    if skeleton_folder and hip_folder and wrist_folder:
        return CSVPairedGaitDataset(
            skeleton_folder=skeleton_folder,
            hip_folder=hip_folder,
            wrist_folder=wrist_folder,
            window=window,
            joints=joints,
            stride=stride,
            num_classes=num_classes,
            normalize_sensors=normalize_sensors,
        )
    raise ValueError(
        "Strict proposal mode requires either --dataset_path or all CSV folders: "
        "--skeleton_folder, --hip_folder, --wrist_folder."
    )


def create_dataloader(
    dataset_path: Optional[str],
    batch_size: int,
    shuffle: bool = True,
    window: int = DEFAULT_WINDOW,
    joints: int = DEFAULT_JOINTS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    skeleton_folder: Optional[str] = None,
    hip_folder: Optional[str] = None,
    wrist_folder: Optional[str] = None,
    stride: int = 30,
    normalize_sensors: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    sampler: Optional[Sampler] = None,
    dataset: Optional[Dataset] = None,
    drop_last: bool = True,
) -> DataLoader:
    """Create dataloader from torch-file or paired CSV folders."""
    if dataset is None:
        dataset = create_dataset(
            dataset_path=dataset_path,
            window=window,
            joints=joints,
            num_classes=num_classes,
            skeleton_folder=skeleton_folder,
            hip_folder=hip_folder,
            wrist_folder=wrist_folder,
            stride=stride,
            normalize_sensors=normalize_sensors,
        )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "sampler": sampler,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)
