import argparse
import json
from pathlib import Path

import train
from openpi.models import pi0_config
from openpi.training import config as C
from openpi.training import weight_loaders


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train openpi on robomimic HDF5 using a JSON dataset spec.")

    parser.add_argument("--dataset-json", type=Path, required=True, help="Path to dataset JSON file.")
    parser.add_argument(
        "--dataset-split-key",
        type=str,
        default="train",
        help="Top-level key in JSON that contains training data config (default: train).",
    )

    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name (checkpoint run name).")
    parser.add_argument("--assets-base-dir", type=str, default="/storage/ice1/6/7/lma326/openpi-cache")
    parser.add_argument("--checkpoint-base-dir", type=str, default="/storage/cedar/cedar0/cedarp-dxu345-0/lma326/openpi-checkpoints")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--wandb-enabled", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project-name", type=str, default="openpi")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-train-steps", type=int, default=30_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument(
        "--keep-period",
        type=int,
        default=5000,
        help="Keep checkpoint steps where step %% keep_period == 0. Set to 0 to disable retention.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--action-horizon", type=int, default=20)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument(
        "--weight-loader-path",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_base/params",
        help="Checkpoint param path for initialization.",
    )

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


def main() -> None:
    args = _parse_args()

    low_dim_keys = tuple(args.low_dim_key) or (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    rgb_keys = tuple(args.rgb_key) or ("agentview_image",)

    sources = _load_sources(args.dataset_json, args.dataset_split_key)

    if "pi05_base" in args.weight_loader_path and args.action_dim != 32:
        raise ValueError(
            "pi05_base weights expect action_dim=32. "
            f"Got --action-dim={args.action_dim}. "
            "Use --action-dim 32 for pi0.5 finetuning from pi05_base, "
            "or use a checkpoint trained with your chosen action_dim."
        )

    cfg = C.TrainConfig(
        name="robomimic_hdf5_local",
        exp_name=args.exp_name,
        project_name=args.wandb_project_name,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=None if args.keep_period == 0 else args.keep_period,
        seed=args.seed,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=args.action_horizon,
            action_dim=args.action_dim,
            discrete_state_input=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.weight_loader_path),
        data=C.RobomimicHDF5DataConfig(
            repo_id=args.repo_id,
            robomimic_sources=sources,
            robomimic_low_dim_keys=low_dim_keys,
            robomimic_rgb_keys=rgb_keys,
            robomimic_action_key=args.action_key,
            robomimic_normalize_weights_by_ds_size=not args.no_normalize_weights_by_ds_size,
            # Quick-start: bypass required precomputed norm stats.
            base_config=C.DataConfig(),
        ),
    )

    train.main(cfg)


if __name__ == "__main__":
    main()
