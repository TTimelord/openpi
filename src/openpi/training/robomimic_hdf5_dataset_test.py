from pathlib import Path

import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import robomimic_hdf5_dataset as _robomimic_ds
import openpi.transforms as _transforms


def _create_single_demo_file(path: Path, *, length: int, action_dim: int, add_wrist: bool = False) -> None:
    h5py = pytest.importorskip("h5py")

    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = length

        obs = demo.create_group("obs")
        obs.create_dataset(
            "robot0_eef_pos",
            data=np.arange(length * 3, dtype=np.float32).reshape(length, 3) + 0.1,
        )
        obs.create_dataset(
            "robot0_eef_quat",
            data=np.arange(length * 4, dtype=np.float32).reshape(length, 4) + 10.0,
        )
        obs.create_dataset(
            "robot0_gripper_qpos",
            data=np.arange(length, dtype=np.float32).reshape(length, 1) + 20.0,
        )

        base_images = np.zeros((length, 8, 8, 3), dtype=np.uint8)
        for t in range(length):
            base_images[t] = np.uint8(20 + t)
        obs.create_dataset("agentview_image", data=base_images)

        if add_wrist:
            wrist_images = np.zeros((length, 8, 8, 3), dtype=np.uint8)
            for t in range(length):
                wrist_images[t] = np.uint8(50 + t)
            obs.create_dataset("wrist_image", data=wrist_images)

        actions = np.arange(length * action_dim, dtype=np.float32).reshape(length, action_dim)
        demo.create_dataset("action_abs", data=actions)


def test_robomimic_single_file_sample_construction(tmp_path: Path):
    file_path = tmp_path / "single.hdf5"
    _create_single_demo_file(file_path, length=3, action_dim=2, add_wrist=False)

    dataset = _robomimic_ds.RobomimicSingleFileDataset(
        path=str(file_path),
        low_dim_keys=("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        rgb_keys=("agentview_image", "wrist_image"),
        action_key="action_abs",
        action_horizon=4,
        lang="put the mug in the bin",
    )

    sample = dataset[2]

    np.testing.assert_allclose(
        sample["state"],
        np.array([6.1, 7.1, 8.1, 18.0, 19.0, 20.0, 21.0, 22.0], dtype=np.float32),
    )

    assert sample["image"]["base_0_rgb"].shape == (8, 8, 3)
    assert np.all(sample["image"]["base_0_rgb"] == np.uint8(22))
    assert np.all(sample["image"]["left_wrist_0_rgb"] == 0)
    assert np.all(sample["image"]["right_wrist_0_rgb"] == 0)

    assert sample["image_mask"]["base_0_rgb"] == np.True_
    assert sample["image_mask"]["left_wrist_0_rgb"] == np.False_
    assert sample["image_mask"]["right_wrist_0_rgb"] == np.False_

    # Last-step action chunk repeats the final action.
    expected_actions = np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0], [4.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(sample["actions"], expected_actions)

    assert sample["prompt"].item() == "put the mug in the bin"


def test_robomimic_multi_file_weights_size_normalized(tmp_path: Path):
    ds1_path = tmp_path / "ds1.hdf5"
    ds2_path = tmp_path / "ds2.hdf5"
    _create_single_demo_file(ds1_path, length=2, action_dim=2)
    _create_single_demo_file(ds2_path, length=5, action_dim=2)

    ds1 = _robomimic_ds.RobomimicSingleFileDataset(
        path=str(ds1_path),
        low_dim_keys=("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        rgb_keys=("agentview_image",),
        action_key="action_abs",
        action_horizon=3,
    )
    ds2 = _robomimic_ds.RobomimicSingleFileDataset(
        path=str(ds2_path),
        low_dim_keys=("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        rgb_keys=("agentview_image",),
        action_key="action_abs",
        action_horizon=3,
    )

    meta = _robomimic_ds.RobomimicMetaDataset(
        datasets=[ds1, ds2],
        dataset_weights=[0.9, 0.1],
        normalize_weights_by_ds_size=True,
    )

    weights = meta.sample_weights.numpy()
    np.testing.assert_allclose(weights[: len(ds1)].sum(), 0.9)
    np.testing.assert_allclose(weights[len(ds1) :].sum(), 0.1)


def test_robomimic_loader_integration(tmp_path: Path):
    file_path = tmp_path / "train.hdf5"
    _create_single_demo_file(file_path, length=8, action_dim=7, add_wrist=True)

    train_config = _config.TrainConfig(
        name="robomimic_hdf5_test",
        exp_name="unit",
        batch_size=2,
        num_workers=0,
        model=pi0_config.Pi0Config(action_dim=7, action_horizon=4, max_token_len=16),
        data=_config.RobomimicHDF5DataConfig(
            robomimic_sources=(
                _config.RobomimicSource(
                    path=str(file_path),
                    key="train",
                    weight=1.0,
                    # lang="put the mug in the bin",
                ),
            ),
            robomimic_low_dim_keys=("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
            robomimic_rgb_keys=("agentview_image", "wrist_image"),
            robomimic_action_key="action_abs",
            robomimic_normalize_weights_by_ds_size=True,
            model_transforms=lambda model: _transforms.Group(
                inputs=[_transforms.PadStatesAndActions(model.action_dim)],
            ),
        ),
    )

    loader = _data_loader.create_data_loader(
        train_config,
        shuffle=True,
        num_batches=2,
        skip_norm_stats=True,
    )

    # Weighted sampler should be selected when sample_weights are present and shuffle=True.
    torch_loader = loader._data_loader.torch_loader  # noqa: SLF001
    assert isinstance(torch_loader.sampler, torch.utils.data.WeightedRandomSampler)

    batches = list(loader)
    assert len(batches) == 2

    for obs, actions in batches:
        assert actions.shape == (2, 4, 7)
        assert obs.state.shape == (2, 8)
        assert set(obs.images) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
        assert set(obs.image_masks) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}

        print(actions)


if __name__ == "__main__":
    test_robomimic_loader_integration(tmp_path=Path("/tmp"))