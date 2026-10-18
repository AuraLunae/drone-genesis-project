"""
WindHoverEnv の評価スクリプト。
ベース: https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/hover_eval.py

実行例:
  # 最新のランの最新チェックポイントを評価(既定で60秒間)
  python eval.py -e wind-hovering --run latest

  # 実行時間を変える(秒)
  python eval.py --time 120

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
from wind_flag import ALL_STYLES, legend
from wind_hover_env import WindHoverEnv
from run_utils import find_latest_run_dir, find_latest_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="wind-hovering")
    parser.add_argument("--run", type=str, default="latest", help="'latest'、またはタイムスタンプ(run_id)を指定")
    parser.add_argument("--ckpt", type=int, default=None, help="イテレーション番号。省略時はそのランの最新チェックポイント")
    parser.add_argument("-r", "--record", action="store_true", help="Record the rollout to video")
    parser.add_argument(
        "--flag-style",
        "--wind-flag",
        dest="flag_style",
        nargs="?",
        const="cloth",
        choices=ALL_STYLES,
        default="cloth",
        metavar="{cloth,cloth-mesh,plate}",
        help="風を映す旗。既定はPBD布。cloth-meshは面表示、plateは剛体板",
    )
    parser.add_argument("-t", "--time", dest="duration", type=float, default=60.0,
                        help="実行時間[秒]。既定60秒(旧版は約9秒だった)。録画時もこの長さになる")
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
    # 風の強さと向きが誰にでも分かるよう、後方に衝突判定のない白い旗を立てる(視点も旗が入る位置になる)
    env_cfg["visualize_wind_flag"] = True
    env_cfg["wind_flag_style"] = args.flag_style
    env_cfg["max_visualize_FPS"] = 60
    # 実行時間が学習時のエピソード長(15秒)より長い場合、途中でタイムアウトして初期位置に
    # 戻されないよう、エピソード長も実行時間に合わせて延ばす(墜落・範囲外では従来通りリセットされる)
    env_cfg["episode_length_s"] = max(env_cfg["episode_length_s"], args.duration)

    env = WindHoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        wind_cfg=wind_cfg,
        show_viewer=True,
    )

    if env.wind_flag is not None:
        print(legend(env.wind_flag.style))
    print(f"=== 評価対象: {ckpt_path} ===")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    # 実行時間[秒]をシミュレーションのステップ数に変換する(1ステップ = env.dt = 0.01秒)
    max_sim_step = int(args.duration / env.dt)

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

# 風を旗で可視化(既定で布旗。布が使えない環境では板旗に自動切替)
python eval.py -e wind-hovering --run latest

# PBD布ではなく剛体板の旗を使う
python eval.py -e wind-hovering --run latest --flag-style plate
"""