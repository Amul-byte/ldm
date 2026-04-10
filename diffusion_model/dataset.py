"""Dataset utilities with torch-file and paired CSV-folder modes."""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from diffusion_model.gait_metrics import (
    DEFAULT_GAIT_METRICS_DIM,
    GAIT_CACHE_VERSION,
    compute_gait_metrics_numpy,
    load_gait_metrics_csv,
    save_gait_metrics_csv,
)
from diffusion_model.util import (
    DEFAULT_FPS,
    DEFAULT_JOINTS,
    DEFAULT_NUM_CLASSES,
    SOURCE_JOINTS_32,
    DEFAULT_WINDOW,
    assert_shape,
    get_joint_labels,
    project_skeleton_to_canonical_numpy,
    project_skeleton_to_canonical_torch,
    require_canonical_joint_count,
    validate_joint_labels,
)

ACTIVITY_RE = re.compile(r"A(\d{2})", re.IGNORECASE)
SUBJECT_RE = re.compile(r"S(\d+)", re.IGNORECASE)


def _parse_label_14(fname: str, num_classes: int = DEFAULT_NUM_CLASSES) -> int:
    """Parse activity code Axx from filename into zero-based label index."""
    match = ACTIVITY_RE.search(fname)
    if match is None:
        return 0
    activity = int(match.group(1))
    return max(0, min(num_classes - 1, activity - 1))


def _parse_subject_id(name: str) -> int:
    """Parse subject code Sxx from filename-like strings."""
    match = SUBJECT_RE.search(str(name))
    if match is None:
        raise ValueError(f"Could not parse subject id from: {name}")
    return int(match.group(1))


def parse_subject_list(raw: str) -> list[int]:
    """Parse a comma-separated subject list like '28,29,30'."""
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    return [int(item) for item in items]


def extract_subject_ids(dataset: Dataset) -> Optional[list[int]]:
    """Return per-sample subject ids when available on a dataset or subset."""
    if isinstance(dataset, Subset):
        base_subject_ids = extract_subject_ids(dataset.dataset)
        if base_subject_ids is None:
            return None
        return [base_subject_ids[idx] for idx in dataset.indices]
    subject_ids = getattr(dataset, "subject_ids", None)
    if subject_ids is None:
        return None
    if isinstance(subject_ids, torch.Tensor):
        return [int(x) for x in subject_ids.tolist()]
    return [int(x) for x in subject_ids]


def split_train_val_dataset(
    dataset: Dataset,
    val_split: float,
    seed: int,
    train_subjects: Optional[list[int]] = None,
    logger=None,
) -> tuple[Dataset, Optional[Dataset]]:
    """Split dataset into train/val subsets, using subject-wise partitioning when requested."""
    subject_ids = extract_subject_ids(dataset)
    if train_subjects:
        if subject_ids is None:
            raise ValueError("Subject-wise split requested, but dataset does not expose per-sample subject_ids.")
        train_subject_set = {int(sid) for sid in train_subjects}
        all_subjects = sorted(set(subject_ids))
        train_idx = [idx for idx, sid in enumerate(subject_ids) if sid in train_subject_set]
        val_idx = [idx for idx, sid in enumerate(subject_ids) if sid not in train_subject_set]
        missing_subjects = sorted(train_subject_set.difference(all_subjects))
        if missing_subjects and logger is not None:
            logger.warning("Requested train subjects not present in dataset: %s", missing_subjects)
        if not train_idx:
            raise ValueError("Subject-wise split produced an empty training set.")
        if not val_idx:
            raise ValueError("Subject-wise split produced an empty validation set.")
        if logger is not None:
            logger.info("Subject-wise split enabled.")
            logger.info("Train subjects: %s", sorted(set(subject_ids[idx] for idx in train_idx)))
            logger.info("Val subjects: %s", sorted(set(subject_ids[idx] for idx in val_idx)))
            logger.info("Train samples=%s Val samples=%s", len(train_idx), len(val_idx))
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    if val_split <= 0.0:
        return dataset, None
    total = len(dataset)
    if total < 2:
        raise ValueError("Validation split requires at least 2 samples.")
    n_val = max(1, int(round(total * val_split)))
    n_val = min(n_val, total - 1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=g).tolist()
    train_idx = perm[:-n_val]
    val_idx = perm[-n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


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
    """Convert raw skeleton CSV rows into canonical [T, 16, 3] joint tensors."""
    if frame_block.ndim != 2:
        raise ValueError(f"Skeleton input must be 2D, got {frame_block.shape}")
    if frame_block.shape[1] in {SOURCE_JOINTS_32 * 3 + 1, DEFAULT_JOINTS * 3 + 1}:
        frame_block = frame_block[:, 1:]
    if frame_block.shape[1] == SOURCE_JOINTS_32 * 3:
        pose = frame_block.reshape(frame_block.shape[0], SOURCE_JOINTS_32, 3).astype(np.float32) / 1000.0
    elif frame_block.shape[1] == DEFAULT_JOINTS * 3:
        pose = frame_block.reshape(frame_block.shape[0], DEFAULT_JOINTS, 3).astype(np.float32) / 1000.0
    else:
        raise ValueError(
            f"Expected {SOURCE_JOINTS_32 * 3}/{SOURCE_JOINTS_32 * 3 + 1} or "
            f"{DEFAULT_JOINTS * 3}/{DEFAULT_JOINTS * 3 + 1} skeleton columns, got {frame_block.shape[1]}"
        )
    pose = project_skeleton_to_canonical_numpy(pose)
    if joints != pose.shape[1]:
        raise ValueError(f"Canonical skeleton has {pose.shape[1]} joints, but dataset requested {joints}")
    return pose


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


def _default_gait_cache_dir(dataset_path: Optional[str], skeleton_folder: Optional[str]) -> str:
    if dataset_path:
        root = os.path.dirname(os.path.abspath(dataset_path)) or "."
        stem = os.path.splitext(os.path.basename(dataset_path))[0]
        return os.path.join(root, f"{stem}_gait_cache_{GAIT_CACHE_VERSION}")
    if skeleton_folder:
        return os.path.join(os.path.abspath(skeleton_folder), f"_gait_cache_{GAIT_CACHE_VERSION}")
    return os.path.abspath(f"gait_cache_{GAIT_CACHE_VERSION}")


def _cached_or_compute_gait_metrics(
    skeleton: np.ndarray,
    cache_path: Optional[str],
    fps: float = DEFAULT_FPS,
    disable_cache: bool = False,
) -> np.ndarray:
    if cache_path and not disable_cache and os.path.isfile(cache_path):
        try:
            return load_gait_metrics_csv(cache_path)
        except Exception:
            pass
    vector, named = compute_gait_metrics_numpy(skeleton, fps=fps)
    if cache_path and not disable_cache:
        save_gait_metrics_csv(cache_path, named)
    return vector.astype(np.float32)


class TorchFileGaitDataset(Dataset):
    """Dataset reading `.pt`/`.pth` files containing tensor dictionaries."""

    def __init__(
        self,
        path: str,
        window: int = DEFAULT_WINDOW,
        joints: int = DEFAULT_JOINTS,
        gait_cache_dir: Optional[str] = None,
        disable_gait_cache: bool = False,
    ) -> None:
        super().__init__()
        require_canonical_joint_count(joints, "TorchFileGaitDataset")
        self.window = window
        self.joints = joints
        self.gait_cache_dir = gait_cache_dir or _default_gait_cache_dir(dataset_path=path, skeleton_folder=None)
        self.disable_gait_cache = disable_gait_cache
        payload = torch.load(path, map_location="cpu")
        self.skeleton = project_skeleton_to_canonical_torch(payload["skeleton"].float() / 1000.0)
        # Prefer explicit accel naming and keep legacy fallback support.
        self.A_hip = payload["A_hip"].float() if "A_hip" in payload else payload["A"].float()
        self.A_wrist = payload["A_wrist"].float() if "A_wrist" in payload else payload["Omega"].float()
        gait_metrics = payload.get("gait_metrics", None)
        if gait_metrics is None:
            metrics = []
            os.makedirs(self.gait_cache_dir, exist_ok=True)
            for idx in range(self.skeleton.shape[0]):
                cache_path = os.path.join(self.gait_cache_dir, f"{os.path.splitext(os.path.basename(path))[0]}_{idx:06d}.csv")
                vector = _cached_or_compute_gait_metrics(
                    self.skeleton[idx].cpu().numpy(),
                    cache_path=cache_path,
                    fps=float(payload.get("fps", DEFAULT_FPS)),
                    disable_cache=disable_gait_cache,
                )
                metrics.append(vector)
            self.gait_metrics = torch.tensor(np.stack(metrics), dtype=torch.float32)
        else:
            gait_metrics = torch.as_tensor(gait_metrics, dtype=torch.float32)
            if gait_metrics.ndim == 1:
                gait_metrics = gait_metrics.unsqueeze(0).expand(self.skeleton.shape[0], -1).contiguous()
            elif gait_metrics.ndim != 2:
                raise ValueError(f"Expected gait_metrics to have shape [N,G] or [G], got {tuple(gait_metrics.shape)}")
            if gait_metrics.shape[0] != self.skeleton.shape[0]:
                raise ValueError(
                    "gait_metrics sample count must match skeleton sample count: "
                    f"{gait_metrics.shape[0]} vs {self.skeleton.shape[0]}"
                )
            if gait_metrics.shape[1] != DEFAULT_GAIT_METRICS_DIM:
                raise ValueError(
                    f"Expected gait_metrics dim {DEFAULT_GAIT_METRICS_DIM}, got {gait_metrics.shape[1]}"
                )
            self.gait_metrics = gait_metrics
        if "label" in payload:
            self.label = payload["label"].long()
        else:
            raise ValueError("Missing required 'label' in dataset payload.")
        subject_ids = payload.get("subject_ids", None)
        if subject_ids is None:
            for candidate_key in ("filenames", "file_names", "sample_names", "sample_ids"):
                names = payload.get(candidate_key, None)
                if names is None:
                    continue
                try:
                    subject_ids = [_parse_subject_id(name) for name in names]
                    break
                except Exception:
                    subject_ids = None
        if subject_ids is None:
            self.subject_ids = None
        else:
            self.subject_ids = torch.as_tensor(subject_ids, dtype=torch.long)
            if self.subject_ids.ndim != 1 or self.subject_ids.shape[0] != self.skeleton.shape[0]:
                raise ValueError(
                    "subject_ids must have shape [N] matching skeleton sample count: "
                    f"{tuple(self.subject_ids.shape)} vs {self.skeleton.shape[0]}"
                )
        self.fps = int(payload["fps"])
        sensor_identity = payload.get("sensor_identity", {})
        joint_labels = payload["joint_labels"]
        validate_joint_labels(joint_labels, allow_legacy_source=True)
        if sensor_identity:
            hip_id = sensor_identity.get("A_hip", sensor_identity.get("A", ""))
            wrist_id = sensor_identity.get("A_wrist", sensor_identity.get("Omega", ""))
            assert hip_id in {"meta_hip", "right_hip"}, "Hip accel stream must correspond to meta_hip"
            assert wrist_id in {"weta_wrist", "meta_wrist", "left_wrist"}, "Wrist accel stream must correspond to weta_wrist"
        assert self.fps == 30, f"SmartFallMM constraint violated: expected 30 FPS, got {self.fps}"
        self._fps_tensor = torch.tensor(self.fps, dtype=torch.long)
        self._joint_labels = list(get_joint_labels())

        assert_shape(self.skeleton, [None, window, joints, 3], "TorchFileGaitDataset.skeleton")
        assert_shape(self.A_hip, [self.skeleton.shape[0], window, 3], "TorchFileGaitDataset.A_hip")
        assert_shape(self.A_wrist, [self.skeleton.shape[0], window, 3], "TorchFileGaitDataset.A_wrist")
        assert_shape(self.gait_metrics, [self.skeleton.shape[0], None], "TorchFileGaitDataset.gait_metrics")
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
            "gait_metrics": self.gait_metrics[idx],
            "label": self.label[idx],
            "fps": self._fps_tensor,
            "joint_labels": self._joint_labels,
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
        gait_cache_dir: Optional[str] = None,
        disable_gait_cache: bool = False,
    ) -> None:
        super().__init__()
        require_canonical_joint_count(joints, "CSVPairedGaitDataset")
        self.window = window
        self.joints = joints
        self.fps = 30
        self.gait_cache_dir = gait_cache_dir or _default_gait_cache_dir(dataset_path=None, skeleton_folder=skeleton_folder)
        self.disable_gait_cache = disable_gait_cache
        os.makedirs(self.gait_cache_dir, exist_ok=True)

        skeleton_map = read_csv_files(skeleton_folder)
        hip_map = read_csv_files(hip_folder)
        wrist_map = read_csv_files(wrist_folder)
        common = sorted(set(skeleton_map).intersection(hip_map).intersection(wrist_map))
        if not common:
            raise ValueError("No common CSV filenames across skeleton/hip/wrist folders")

        x_windows = []
        hip_windows = []
        wrist_windows = []
        gait_metric_windows = []
        labels = []
        subject_ids = []
        skipped_parse = 0
        skipped_short = 0

        for fname in common:
            try:
                skel = _skeleton_frame_to_joints(_fill_nan_with_column_mean(skeleton_map[fname].values), joints=joints)
                hip = _extract_sensor_accel3(hip_map[fname])
                wrist = _extract_sensor_accel3(wrist_map[fname])
                cache_path = os.path.join(self.gait_cache_dir, fname)
                gait_metrics = _cached_or_compute_gait_metrics(
                    skel,
                    cache_path=cache_path,
                    fps=float(self.fps),
                    disable_cache=disable_gait_cache,
                )
                if gait_metrics.shape[0] != DEFAULT_GAIT_METRICS_DIM:
                    raise ValueError(
                        f"Expected auto-computed gait_metrics dim {DEFAULT_GAIT_METRICS_DIM}, got {gait_metrics.shape[0]}"
                    )
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
                gait_metric_windows.append(gait_metrics)
                labels.append(_parse_label_14(fname, num_classes=num_classes))
                subject_ids.append(_parse_subject_id(fname))

        if len(x_windows) == 0:
            raise ValueError(
                "No valid paired windows found in CSV folders "
                f"(common_files={len(common)}, parse_failed={skipped_parse}, too_short_for_window={skipped_short}, "
                f"window={window}, stride={stride})"
            )

        self.skeleton = torch.tensor(np.stack(x_windows), dtype=torch.float32)
        self.A_hip = torch.tensor(np.stack(hip_windows), dtype=torch.float32)
        self.A_wrist = torch.tensor(np.stack(wrist_windows), dtype=torch.float32)
        self.gait_metrics = torch.tensor(np.stack(gait_metric_windows), dtype=torch.float32)
        self.label = torch.tensor(labels, dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids, dtype=torch.long)

        assert_shape(self.skeleton, [None, window, joints, 3], "CSVPairedGaitDataset.skeleton")
        assert_shape(self.A_hip, [self.skeleton.shape[0], window, 3], "CSVPairedGaitDataset.A_hip")
        assert_shape(self.A_wrist, [self.skeleton.shape[0], window, 3], "CSVPairedGaitDataset.A_wrist")
        assert_shape(self.gait_metrics, [self.skeleton.shape[0], None], "CSVPairedGaitDataset.gait_metrics")
        assert_shape(self.label, [self.skeleton.shape[0]], "CSVPairedGaitDataset.label")
        assert_shape(self.subject_ids, [self.skeleton.shape[0]], "CSVPairedGaitDataset.subject_ids")
        self._fps_tensor = torch.tensor(self.fps, dtype=torch.long)
        self._joint_labels = list(get_joint_labels())

    def __len__(self) -> int:
        """Return number of samples."""
        return self.skeleton.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one sample dictionary."""
        return {
            "skeleton": self.skeleton[idx],
            "A_hip": self.A_hip[idx],
            "A_wrist": self.A_wrist[idx],
            "gait_metrics": self.gait_metrics[idx],
            "label": self.label[idx],
            "fps": self._fps_tensor,
            "joint_labels": self._joint_labels,
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
    gait_cache_dir: Optional[str] = None,
    disable_gait_cache: bool = False,
) -> Dataset:
    """Create dataset object from torch-file or paired CSV folders."""
    if dataset_path:
        return TorchFileGaitDataset(
            path=dataset_path,
            window=window,
            joints=joints,
            gait_cache_dir=gait_cache_dir,
            disable_gait_cache=disable_gait_cache,
        )
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
            gait_cache_dir=gait_cache_dir,
            disable_gait_cache=disable_gait_cache,
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
    gait_cache_dir: Optional[str] = None,
    disable_gait_cache: bool = False,
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
            gait_cache_dir=gait_cache_dir,
            disable_gait_cache=disable_gait_cache,
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
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)
