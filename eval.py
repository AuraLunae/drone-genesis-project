"""
WindHoverEnv の評価スクリプト。
ベース: https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/hover_eval.py

実行例:
  # 最新のランの最新チェックポイントを評価(既定で60秒間)
  python eval.py -e wind-hovering --run latest

  # 実行時間を変える(秒)
  python eval.py --time 120

  # 風なしで、ゴールを次々にタッチしていく(タッチするたびに次のゴールが出る)
  python eval.py --touch

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
from genesis_compat import RealtimePacer, fast_init
from wind_flag import ALL_STYLES, legend
from wind_hover_env import WindHoverEnv
from run_utils import find_latest_run_dir, find_latest_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="wind-hovering")
    parser.add_argument("--run", type=str, default="latest", help="'latest'、またはタイムスタンプ(run_id)を指定")
    parser.add_argument("--ckpt", type=int, default=None, help="イテレーション番号。省略時はそのランの最新チェックポイント")
    parser.add_argument("-r", "--record", action="store_true", help="Record the rollout to video")
    parser.add_argument("--no-wind-flag", action="store_true", help="風を表示する旗を非表示にする")
    parser.add_argument(
        "--flag-style",
        "--wind-flag",
        dest="flag_style",
        nargs="?",
        const="surface",
        choices=ALL_STYLES,
        default=None,  # 未指定なら surface(--pbd が付けば cloth-mesh)
        metavar="{surface,cloth-mesh,cloth,cloth-vel,plate}",
        help="風を映す旗。既定は式で形を計算するスプライン曲面。cloth-meshはPBD布、clothは粒子表示、"
             "cloth-velはPBD布(粒子の速度に風を足す方式)、plateは剛体板",
    )
    parser.add_argument("--pbd", action="store_true",
                        help="旗をPBD布の物理計算にする(--flag-style cloth-mesh の短縮形)")
    parser.add_argument("-t", "--time", dest="duration", type=float, default=60.0,
                        help="実行時間[秒]。既定60秒(旧版は約9秒だった)。録画時もこの長さになる")
    parser.add_argument("--touch", action="store_true",
                        help="タッチチャレンジ(同じモデルで動く): 風なしで、ゴール(赤い球)にタッチすると次のゴールが出る。"
                             "タッチ回数と間隔を表示する")
    parser.add_argument("--mirror", action="store_true", help="床を暗い鏡面にする(既定はGenesis標準の床。反射の描画ぶん重くなる)")
    parser.add_argument("--record-res", type=str, default="1920x1080", help="録画の解像度(幅x高さ)。既定1920x1080")
    parser.add_argument("--record-fps", type=int, default=100,
                        help="録画のfps。既定100(物理が100Hzなので、毎ステップ1コマ。これ以上は上げても同じコマが増えるだけ)")
    parser.add_argument("--full-physics", action="store_true",
                        help="学習時と同じ物理設定(床との衝突判定あり)に戻す。既定は、評価では結果が変わらない衝突判定を切って軽くする")
    parser.add_argument("--unpaced", action="store_true",
                        help="実時間(1倍速)に合わせず、できるだけ速く進める。既定は1倍速(速すぎる分は眠る)。--record の時は常にこちら")
    args = parser.parse_args()
    if args.pbd and args.flag_style not in (None, "cloth-mesh"):
        parser.error("--pbd は --flag-style cloth-mesh と同じ意味です。他の旗のスタイルとは併用できません")
    args.flag_style = "cloth-mesh" if args.pbd else (args.flag_style or "surface")
    try:
        record_w, record_h = (int(v) for v in args.record_res.lower().split("x"))
    except ValueError:
        parser.error("--record-res は 1920x1080 のように指定してください")

    fast_init(gs.cpu)  # 32bit・performance_mode・PyTorchのスレッド1(軽量設定)

    log_dir = find_latest_run_dir(args.exp_name) if args.run == "latest" else os.path.join("logs", args.exp_name, args.run)
    ckpt_path = (
        os.path.join(log_dir, f"model_{args.ckpt}.pt") if args.ckpt is not None else find_latest_checkpoint(log_dir)
    )

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, wind_cfg, train_cfg = pickle.load(f)

    reward_cfg["reward_scales"] = {}
    if args.touch:
        # 風なし: 風の強さ・乱れを0にする(旗は垂れ下がったままになる)
        wind_cfg = dict(wind_cfg, strength_range=[0.0, 0.0], sigma_range=[0.0, 0.0])
        # 触れた瞬間にゴールが変わるルールにする(学習時の既定は、時間でゴールが変わる"hover")。
        # 観測は「ゴールまでの相対位置」だけなので、同じモデルがそのまま使える
        env_cfg["goal_mode"] = "touch"
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    # 風の強さと向きが誰にでも分かるよう、後方に衝突判定のない白い旗を立てる(視点も旗が入る位置になる)
    env_cfg["visualize_wind_flag"] = not (args.no_wind_flag or args.touch)  # --touch は風なしなので、旗は出さない
    env_cfg["wind_flag_style"] = args.flag_style
    env_cfg["wind_flag_xy"] = (-4.5, 0.0)   # 旗は奥に置く(遠近法で小さく見える)
    env_cfg["wind_flag_scale"] = 1.2
    env_cfg["max_visualize_FPS"] = args.record_fps if args.record else 60   # 録画のfps(start_recordingに渡る)
    env_cfg["record_res"] = (record_w, record_h)                            # 録画の解像度。録画の時は影も描く
    env_cfg["floor_mirror"] = args.mirror                   # 既定は標準の床。--mirror の時だけ暗い鏡面
    env_cfg["lite_physics"] = not args.full_physics         # 床との衝突判定を切る(評価では墜落判定が高さなので結果は同じ)
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

    touch = TouchCounter(env.dt) if args.touch else None
    pacer = None if (args.record or args.unpaced) else RealtimePacer(env.dt)  # 軽くしたぶん早送りにならないよう、1倍速に合わせる
    if touch:
        print("=== タッチチャレンジ(風なし)。ゴールにタッチすると次のゴールが出ます ===")

    def step_once(obs_dict):
        before = env.commands[0].clone() if touch else None
        actions = policy(obs_dict)
        obs_dict, rews, dones, infos = env.step(actions)
        if touch:
            touch.observe(before, env.commands[0], bool(dones[0]), float(env.rel_pos[0].norm()))
        if pacer is not None:
            pacer.wait()
        return obs_dict

    with torch.no_grad():
        if args.record:
            env.cam.start_recording(save_to_filename="out/wind_hover.mp4", fps=env_cfg["max_visualize_FPS"])
            for _ in range(max_sim_step):
                obs_dict = step_once(obs_dict)
            env.cam.stop_recording()
        else:
            for _ in range(max_sim_step):
                obs_dict = step_once(obs_dict)
    if touch:
        touch.summary()


class TouchCounter:
    """タッチ回数の集計。env.step() の前後でゴールが変わったら「タッチした」とみなす。

    goal_mode="touch" の WindHoverEnv は、機体がゴールの at_target_threshold(既定0.1 m)以内に入ると、
    同じステップでゴールを新しい位置に引き直す。そのため、ゴールが変わった = タッチ。
    ただし、墜落などのリセット(dones)でもゴールは引き直されるので、その場合は数えない。
    """

    NEAR_M = 0.2   # この距離[m]以内で、タッチ(0.1 m)せずにいた時間を「待機」として数える

    def __init__(self, dt):
        self.dt = dt
        self.near_steps = 0
        self.steps = 0
        self.last_touch_step = 0
        self.count = 0
        self.crashes = 0
        self.intervals = []

    def observe(self, before, after, done, dist=None):
        self.steps += 1
        if done:
            self.crashes += 1
            self.last_touch_step = self.steps
            print(f"  [{self.steps * self.dt:6.1f}s] 墜落・範囲外でリセットされました")
            return
        if dist is not None and 0.1 <= dist < self.NEAR_M:
            self.near_steps += 1
        if bool((before != after).any()):
            self.count += 1
            interval = (self.steps - self.last_touch_step) * self.dt
            self.last_touch_step = self.steps
            self.intervals.append(interval)
            print(f"  [{self.steps * self.dt:6.1f}s] タッチ {self.count} 回目 (前のタッチから {interval:.2f} 秒)")

    def summary(self):
        print("=== 結果 ===")
        print(f"実行時間 {self.steps * self.dt:.1f} 秒 / タッチ {self.count} 回 / 墜落 {self.crashes} 回")
        if self.intervals:
            print(f"タッチの間隔: 平均 {sum(self.intervals) / len(self.intervals):.2f} 秒 / "
                  f"最短 {min(self.intervals):.2f} 秒 / 最長 {max(self.intervals):.2f} 秒")
        near = self.near_steps * self.dt
        print(f"ゴールの{self.NEAR_M:.1f} m以内で、タッチせずにいた時間: {near:.1f} 秒")
        if self.steps > 0 and near > 0.3 * self.steps * self.dt:
            print("[注意] ゴール手前(0.1〜0.2 m)で待っている時間が長い。位置の精度が足りず、タッチの半径(0.1 m)に入れていない可能性がある")


if __name__ == "__main__":
    main()

"""
# 最新のランを評価
python eval.py -e wind-hovering --run latest

# 録画したい場合
python eval.py -e wind-hovering --run latest --record

# 旗を非表示にする場合
python eval.py -e wind-hovering --run latest --no-wind-flag

# 風を旗で可視化(既定はスプライン曲面の旗。使えない環境ではPBD布→板旗に自動で切り替わる)
python eval.py -e wind-hovering --run latest

# PBD布の物理計算にする場合(--pbd は --flag-style cloth-mesh の短縮形)
python eval.py -e wind-hovering --run latest --pbd

# PBD布の、粒子の速度に風を足す方式にする場合
python eval.py -e wind-hovering --run latest --flag-style cloth-vel

# 風なしのタッチチャレンジ(ゴールに触れると次のゴールが出る)
python eval.py -e wind-hovering --run latest --touch

# 床を暗い鏡面にする(既定はGenesis標準の床)
python eval.py -e wind-hovering --run latest --mirror

# 録画(既定は1920x1080・100fps・影あり。変える場合)
python eval.py -e wind-hovering --run latest --record --time 10 --record-res 2560x1440 --record-fps 100

# 学習時と同じ物理設定(衝突判定あり)に戻す / 実時間に合わせず最速で進める
python eval.py -e wind-hovering --run latest --full-physics
python eval.py -e wind-hovering --run latest --unpaced
"""