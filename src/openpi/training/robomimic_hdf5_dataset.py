from __future__ import annotations

from collections.abc import Sequence
import bisect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    import h5py
    import openpi.training.config as _config


def _import_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "robomimic HDF5 data loading requires `h5py`. Install it with `uv pip install h5py`."
        ) from exc
    return h5py


def _demo_sort_key(name: str) -> tuple[int, int | str]:
    # Sort demo_0, demo_1, ... numerically where possible.
    prefix = "demo_"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, name)


def _to_hwc_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")
    if image.shape[-1] == 3:
        return image
    if image.shape[0] == 3:
        return np.transpose(image, (1, 2, 0))
    raise ValueError(f"Expected image in HWC or CHW format, got shape={image.shape}")


class RobomimicSingleFileDataset(torch.utils.data.Dataset):
    """Loads one robomimic HDF5 file and exposes timestep-level samples."""

    def __init__(
        self,
        *,
        path: str,
        low_dim_keys: Sequence[str],
        rgb_keys: Sequence[str],
        action_key: str,
        action_horizon: int,
        demo_limit: int | None = None,
        lang: str | None = None,
    ):
        self._path = str(Path(path).expanduser())
        self._low_dim_keys = tuple(low_dim_keys)
        self._rgb_keys = tuple(rgb_keys)
        self._action_key = action_key
        self._action_horizon = action_horizon
        self._lang = lang

        if not self._low_dim_keys:
            raise ValueError("robomimic_low_dim_keys must not be empty.")
        if not self._rgb_keys:
            raise ValueError("robomimic_rgb_keys must not be empty.")
        if self._action_horizon <= 0:
            raise ValueError(f"action_horizon must be > 0, got {self._action_horizon}")

        self._h5_file: h5py.File | None = None
        self._demos: list[str] = []
        self._demo_lengths: np.ndarray
        self._cumulative_lengths: np.ndarray
        self._total_len = 0

        self._index_file(demo_limit=demo_limit)

    def _index_file(self, *, demo_limit: int | None) -> None:
        h5py = _import_h5py()
        with h5py.File(self._path, "r") as f:
            if "data" not in f:
                raise ValueError(f"Invalid robomimic dataset: missing `data` group in {self._path}")

            demos = sorted(f["data"].keys(), key=_demo_sort_key)
            if demo_limit is not None:
                demos = demos[:demo_limit]
            if not demos:
                raise ValueError(f"No demos found in {self._path}")

            retained_demos = []
            demo_lengths = []
            for demo in demos:
                demo_group = f["data"][demo]
                if self._action_key not in demo_group:
                    raise KeyError(f"Missing action key `{self._action_key}` under data/{demo} in {self._path}")
                if "obs" not in demo_group:
                    raise KeyError(f"Missing `obs` group under data/{demo} in {self._path}")

                # Prefer attrs["num_samples"] if present, otherwise infer from action length.
                action_len = int(demo_group[self._action_key].shape[0])
                demo_len = int(demo_group.attrs.get("num_samples", action_len))
                demo_len = min(demo_len, action_len)
                if demo_len <= 0:
                    continue

                obs_group = demo_group["obs"]
                for key in self._low_dim_keys:
                    if key not in obs_group:
                        raise KeyError(f"Missing low-dim key `obs/{key}` under data/{demo} in {self._path}")
                if self._rgb_keys[0] not in obs_group:
                    raise KeyError(
                        f"Missing primary rgb key `obs/{self._rgb_keys[0]}` under data/{demo} in {self._path}"
                    )

                retained_demos.append(demo)
                demo_lengths.append(demo_len)

            if not demo_lengths:
                raise ValueError(f"No non-empty demos found in {self._path}")

            self._demos = retained_demos
            self._demo_lengths = np.asarray(demo_lengths, dtype=np.int64)
            self._cumulative_lengths = np.cumsum(self._demo_lengths)
            self._total_len = int(self._cumulative_lengths[-1])

    @property
    def _file(self) -> "h5py.File":
        if self._h5_file is None:
            h5py = _import_h5py()
            self._h5_file = h5py.File(self._path, "r", swmr=True, libver="latest")
        return self._h5_file

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        # h5py handles are not picklable; workers should reopen lazily.
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def __len__(self) -> int:
        return self._total_len

    def _locate_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._total_len
        if index < 0 or index >= self._total_len:
            raise IndexError(f"Index {index} out of range for dataset of size {self._total_len}")

        demo_idx = bisect.bisect_right(self._cumulative_lengths, index)
        demo_start = 0 if demo_idx == 0 else int(self._cumulative_lengths[demo_idx - 1])
        index_in_demo = index - demo_start
        return demo_idx, index_in_demo

    def _read_image(self, obs_group: Any, key: str, t: int) -> np.ndarray:
        image = np.asarray(obs_group[key][t])
        return _to_hwc_image(image)

    def __getitem__(self, index: int) -> dict[str, Any]:
        demo_idx, t = self._locate_index(int(index))

        demo_name = self._demos[demo_idx]
        demo_group = self._file["data"][demo_name]
        obs_group = demo_group["obs"]

        state_parts = [np.asarray(obs_group[key][t], dtype=np.float32).reshape(-1) for key in self._low_dim_keys]
        state = np.concatenate(state_parts, axis=-1)

        base_image = self._read_image(obs_group, self._rgb_keys[0], t)
        if len(self._rgb_keys) > 1 and self._rgb_keys[1] in obs_group:
            left_wrist_image = self._read_image(obs_group, self._rgb_keys[1], t)
            left_wrist_mask = np.True_
        else:
            left_wrist_image = np.zeros_like(base_image)
            left_wrist_mask = np.False_
        right_wrist_image = np.zeros_like(base_image)

        demo_len = int(self._demo_lengths[demo_idx])

        # h5py advanced indexing requires strictly increasing indices. Instead of
        # materializing clipped indices (which contain repeats near trajectory end),
        # read a contiguous slice and pad by repeating the final action.
        action_end = min(t + self._action_horizon, demo_len)
        actions = np.asarray(demo_group[self._action_key][t:action_end], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, None]

        if actions.shape[0] < self._action_horizon:
            pad_count = self._action_horizon - actions.shape[0]
            final_action = actions[-1:]
            actions = np.concatenate([actions, np.repeat(final_action, pad_count, axis=0)], axis=0)

        sample: dict[str, Any] = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_wrist_mask,
                "right_wrist_0_rgb": np.False_,
            },
            "actions": actions,
        }
        if self._lang is not None:
            sample["prompt"] = np.asarray(self._lang)
        return sample


class RobomimicMetaDataset(torch.utils.data.Dataset):
    """Combines multiple robomimic files and exposes size-normalized sample weights."""

    def __init__(
        self,
        datasets: Sequence[RobomimicSingleFileDataset],
        dataset_weights: Sequence[float],
        *,
        normalize_weights_by_ds_size: bool = True,
    ):
        if not datasets:
            raise ValueError("At least one robomimic source is required.")
        if len(datasets) != len(dataset_weights):
            raise ValueError("datasets and dataset_weights must have the same length.")

        self._datasets = list(datasets)
        lengths = np.asarray([len(ds) for ds in self._datasets], dtype=np.int64)
        if np.any(lengths <= 0):
            raise ValueError(f"All robomimic datasets must be non-empty. Lengths={lengths.tolist()}")

        weights = np.asarray(dataset_weights, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError(f"Dataset weights must be non-negative. Got {weights.tolist()}")

        self._offsets = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), lengths)))
        self._total_len = int(self._offsets[-1])

        if normalize_weights_by_ds_size:
            per_sample_dataset_weights = weights / lengths
        else:
            per_sample_dataset_weights = weights

        sample_weights = np.empty(self._total_len, dtype=np.float64)
        for i, (start, end) in enumerate(zip(self._offsets[:-1], self._offsets[1:], strict=True)):
            sample_weights[start:end] = per_sample_dataset_weights[i]
        self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self._total_len
        if index < 0 or index >= self._total_len:
            raise IndexError(f"Index {index} out of range for dataset of size {self._total_len}")

        dataset_idx = bisect.bisect_right(self._offsets, index) - 1
        index_in_dataset = index - int(self._offsets[dataset_idx])
        return self._datasets[dataset_idx][index_in_dataset]


def create_robomimic_dataset(
    data_config: "_config.DataConfig",
    *,
    action_horizon: int,
) -> RobomimicMetaDataset:
    if not data_config.robomimic_sources:
        raise ValueError("robomimic_sources is empty.")

    datasets = [
        RobomimicSingleFileDataset(
            path=source.path,
            low_dim_keys=data_config.robomimic_low_dim_keys,
            rgb_keys=data_config.robomimic_rgb_keys,
            action_key=data_config.robomimic_action_key,
            action_horizon=action_horizon,
            demo_limit=source.demo_limit,
            lang=source.lang,
        )
        for source in data_config.robomimic_sources
    ]
    return RobomimicMetaDataset(
        datasets,
        dataset_weights=[source.weight for source in data_config.robomimic_sources],
        normalize_weights_by_ds_size=data_config.robomimic_normalize_weights_by_ds_size,
    )
