import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

import openpi.models.model as _model
from openpi.models import pi0_config
import openpi.shared.normalize as normalize
from openpi.training import config as C
from openpi.training import data_loader as DL
from openpi.training import weight_loaders
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute normalization stats for robomimic HDF5 in openpi format.")

    parser.add_argument("--dataset-json", type=Path, required=True, help="Path to dataset JSON file.")
    parser.add_argument(
        "--dataset-split-key",
        type=str,
        default="train",
        help="Top-level key in JSON that contains training data config (default: train).",
    )

    parser.add_argument(
        "--config-name",
        type=str,
        default="robomimic_hdf5_local",
        help="Config name used to build output asset path: <assets_base_dir>/<config_name>/<repo_id>",
    )
    parser.add_argument("--assets-base-dir", type=str, default="/storage/ice1/6/7/lma326/openpi-cache")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--action-horizon", type=int, default=20)
    parser.add_argument("--action-dim", type=int, default=32)

    parser.add_argument(
        "--low-dim-key",
        action="append",
        default=[],
        help="Robomimic low-dim obs key. Repeat flag for multiple keys.",
    )
    parser.add_argument(
        "--rgb-key",
        action="append",
        default=[],
        help="Robomimic RGB obs key. Repeat flag for multiple keys.",
    )
    parser.add_argument("--action-key", type=str, default="actions_abs")
    parser.add_argument("--repo-id", type=str, default="robomimic_hdf5")
    parser.add_argument("--no-normalize-weights-by-ds-size", action="store_true")

    return parser.parse_args()


def _load_sources(dataset_json_path: Path, split_key: str) -> tuple[C.RobomimicSource, ...]:
    payload = json.loads(dataset_json_path.read_text())
    if split_key not in payload:
        raise KeyError(f"Top-level key '{split_key}' not found in {dataset_json_path}.")

    split_cfg = payload[split_key]
    if isinstance(split_cfg, dict):
        data_list = split_cfg.get("data")
    elif isinstance(split_cfg, list):
        data_list = split_cfg
    else:
        raise ValueError(
            f"Expected '{split_key}' value to be a dict with a 'data' list or a list directly, got: {type(split_cfg)}"
        )

    if not isinstance(data_list, list) or not data_list:
        raise ValueError(f"Expected '{split_key}.data' to be a non-empty list in {dataset_json_path}.")

    sources: list[C.RobomimicSource] = []
    for idx, item in enumerate(data_list):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {idx} under '{split_key}.data' must be an object.")

        if "path" not in item:
            raise KeyError(f"Entry {idx} under '{split_key}.data' is missing required field 'path'.")

        key = item.get("key", f"source_{idx}")
        weight = float(item.get("weight", 1.0))

        sources.append(
            C.RobomimicSource(
                path=str(item["path"]),
                key=str(key),
                demo_limit=item.get("demo_limit"),
                weight=weight,
                lang=item.get("lang"),
                eval=bool(item.get("eval", False)),
            )
        )

    return tuple(sources)


def _create_torch_dataloader(
    data_config: C.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[DL.TorchDataLoader, int]:
    dataset = DL.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = DL.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    if num_batches <= 0:
        raise ValueError(
            f"No batches to process: dataset_size={len(dataset)}, batch_size={batch_size}, max_frames={max_frames}"
        )

    data_loader = DL.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main() -> None:
    args = _parse_args()

    low_dim_keys = tuple(args.low_dim_key) or (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    rgb_keys = tuple(args.rgb_key) or ("agentview_image",)

    sources = _load_sources(args.dataset_json, args.dataset_split_key)

    train_config = C.TrainConfig(
        name=args.config_name,
        exp_name="compute_norm_stats",
        assets_base_dir=args.assets_base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=args.action_horizon,
            action_dim=args.action_dim,
            discrete_state_input=False,
        ),
        # Not used for stats computation, but required by TrainConfig.
        weight_loader=weight_loaders.NoOpWeightLoader(),
        data=C.RobomimicHDF5DataConfig(
            repo_id=args.repo_id,
            robomimic_sources=sources,
            robomimic_low_dim_keys=low_dim_keys,
            robomimic_rgb_keys=rgb_keys,
            robomimic_action_key=args.action_key,
            robomimic_normalize_weights_by_ds_size=not args.no_normalize_weights_by_ds_size,
            base_config=C.DataConfig(),
        ),
        wandb_enabled=False,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    data_loader, num_batches = _create_torch_dataloader(
        data_config,
        train_config.model.action_horizon,
        train_config.batch_size,
        train_config.model,
        train_config.num_workers,
        args.max_frames,
    )

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        stats["state"].update(np.asarray(batch["state"]))
        stats["actions"].update(np.asarray(batch["actions"]))

    norm_stats = {k: v.get_statistics() for k, v in stats.items()}
    output_path = train_config.assets_dirs / data_config.repo_id
    normalize.save(output_path, norm_stats)
    print(f"Saved norm stats to: {output_path}")


if __name__ == "__main__":
    main()
