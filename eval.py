"""
WindHoverEnv の評価スクリプト。
ベース: https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/hover_eval.py

実行例:
  # 最新のランの最新チェックポイントを評価
  python eval.py -e wind-hovering --run latest

  # 特定のランの特定チェックポイントを評価
  python eval.py -e wind-hovering --run 20261004_120000 --ckpt 300
"""

import argparse
import os
import pickle
from importlib import metadata
import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from wind_hover_env import WindHoverEnv
from run_utils import find_latest_run_dir, find_latest_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="wind-hovering")
    parser.add_argument("--run", type=str, default="latest", help="'latest'、またはタイムスタンプ(run_id)を指定")
    parser.add_argument("--ckpt", type=int, default=None, help="イテレーション番号。省略時はそのランの最新チェックポイント")
    parser.add_argument("-r", "--record", action="store_true", help="Record the rollout to video")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = find_latest_run_dir(args.exp_name) if args.run == "latest" else os.path.join("logs", args.exp_name, args.run)
    ckpt_path = (
        os.path.join(log_dir, f"model_{args.ckpt}.pt") if args.ckpt is not None else find_latest_checkpoint(log_dir)
    )

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, wind_cfg, train_cfg = pickle.load(f)

    reward_cfg["reward_scales"] = {}
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60

    env = WindHoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        wind_cfg=wind_cfg,
        show_viewer=True,
    )

    print(f"=== 評価対象: {ckpt_path} ===")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    with torch.no_grad():
        if args.record:
            env.cam.start_recording(save_to_filename="out/wind_hover.mp4", fps=env_cfg["max_visualize_FPS"])
            for _ in range(max_sim_step):
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)
            env.cam.stop_recording()
        else:
            for _ in range(max_sim_step):
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# 最新のランを評価
python eval.py -e wind-hovering --run latest

# 録画したい場合
python eval.py -e wind-hovering --run latest --record
"""