"""
WindHoverEnv の評価スクリプト。
ベース: https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/hover_eval.py

実行例:
  python eval.py -e wind-hovering --ckpt 300
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="wind-hovering")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("-r", "--record", action="store_true", help="Record the rollout to video")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
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

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
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
# 評価実行
python eval.py -e wind-hovering --ckpt 300

# 録画したい場合
python eval.py -e wind-hovering --ckpt 300 --record
"""