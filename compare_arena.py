"""
compare_arena.py

AI・PID・人間の3者が、全く同じ課題を「同時に」行い、点数化して比較するアリーナ。

  - AI    : 学習済みの方策(train.py で学習したもの)。eval.py と同じ入力・出力で動かす
  - PID   : カスケードPID(manual_pid_fly.py と同じ。位置→速度→姿勢→ミキサー)が自動で飛ぶ
  - 人間  : あなたがキーボードで操縦する(manual_raw_fly.py と同じ。モード2送信機相当の4軸操作)

同じ条件の保証:
  - 同じ機体(CF2X)、同じ物理設定(学習時と同じ RigidOptions)
  - 同じ開始位置(各レーンの原点、高さ1.0 m)、同じ目標(レーン原点からの相対座標が同一)
  - 同じ風(同じOU過程の風を、毎ステップ全機の重心に加える。--no-wind で無効)
  - 同じ失敗条件と採点(arena_scoring.py)。失敗条件はAIの学習時の終了条件と同じ

課題: 順番に出てくる複数の目標へ、できるだけ速く正確に近づく(制限時間あり)。
採点の詳細は arena_scoring.py の冒頭を参照(到達・スピード・接近・完走・失敗ペナルティ)。

流れ:
  待機 → [Enter] → 3秒のカウントダウン → 競技開始(人間は自動でアーム) → 終了 → 結果表示
  終了後に [Enter] で次のラウンド(目標と風が変わる。--seed 指定時は seed が1ずつ増える)。

人間の操作方法(manual_raw_fly.py と同じ。↑↓←→ は起動時のカメラ基準):
  Space / X   : 目標の上昇速度を上げる / 下げる(離しても保持)
  B           : 目標上下速度を即座に0へ(ホバリング。標準キーと衝突したら別のキーに自動変更)
  Q / E       : 左旋回 / 右旋回(ヨー)
  ↑ / ↓       : 奥 / 手前へ傾ける(ロール)
  ← / →       : 左 / 右へ傾ける(ピッチ)
  Enter       : ラウンド開始 / 結果表示後は次のラウンド
  Esc         : 終了

画面: 左から AI(青)・PID(緑)・人間(橙)のレーン。球は各自の「現在の目標」(色はレーンと対応)。
      3機の後方に立つ白い旗は、風の強さと向きを映す(衝突判定なし)。垂れ下がる=無風、持ち上がるほど強風、
      旗が向く方向=風下。詳しくは wind_flag.py を参照。コンソールにも風の強さ[N]を表示する。

実行:
  uv run python compare_arena.py                  # AI・PID・人間の3者(AIは最新の学習ランを使う)
  uv run python compare_arena.py --no-ai          # AI無し(PIDと人間だけ。学習前でも遊べる)
  uv run python compare_arena.py --time-limit 90 --waypoints 5 --seed 3
"""

import argparse
import csv
import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat
from genesis.vis.keybindings import Key, KeyAction, Keybind

from arena_scoring import ContestantScore, FailureRules, ScoringConfig, generate_waypoints, rank
from genesis_compat import PropellerVisual, apply_com_force, add_floor, reflective_vis_options
from manual_pid_fly import (
    BASE_RPM,
    DT,
    MAX_RPM as PID_MAX_RPM,
    MIN_RPM as PID_MIN_RPM,
    OFFICIAL_PID_PARAMS,
    DronePIDController,
    SingleDroneWind,
)
from manual_raw_fly import ManualFlightController, check_native_attitude_torque, register_first_free_key
from run_utils import find_latest_checkpoint, find_latest_run_dir
from wind_flag import ALL_STYLES, create_wind_flag, legend, wind_flag_scene_kwargs

# ---------------------------------------------------------------------- #
# 設定
# ---------------------------------------------------------------------- #
LANE_X = {"AI": -2.6, "PID": 0.0, "HUMAN": 2.6}  # 3機を横に並べる(目標は各レーン原点から±1 mなので重ならない)
LABELS = {"AI": "AI", "PID": "PID", "HUMAN": "人間"}
MARKER_COLORS = {"AI": (0.3, 0.55, 1.0), "PID": (0.3, 0.9, 0.4), "HUMAN": (1.0, 0.6, 0.2)}
START_HEIGHT = 1.0
COUNTDOWN_S = 3.0
MAX_ABS_RPM = 25000.0
AI_RPM_SCALE = 0.8  # wind_hover_env.py の step() と同じ: rpm = (1 + action * 0.8) * BASE_RPM
RESULTS_DIR = os.path.join("logs", "arena")
# 風を映す白い旗(衝突判定なし)。カメラ(y=-6から+y方向を見る)の反対側、つまり3機の後方(背景側)に立てる。
# カメラから遠い(水平で約9.5 m)ので、eval.py(約6.7 m)より大きくしてある。
WIND_FLAG_XY = (0.0, 3.5)
WIND_FLAG_SCALE = 1.6
# 旗が最初に伸びる向き。カメラは-y側から+y方向を見るので、広い面が見えるよう、最初はx方向へ伸ばす
WIND_FLAG_ALONG = (1.0, 0.0)
NO_WIND = np.zeros(3)

# 状態
READY, COUNTDOWN, RUNNING, FINISHED = "待機", "カウントダウン", "競技中", "終了"


# ---------------------------------------------------------------------- #
# AIの方策の読み込み
# ---------------------------------------------------------------------- #
class _PolicyLoaderEnv:
    """OnPolicyRunnerを生成して重みを読み込むためだけの最小限の環境(シミュレーションはしない)。

    rsl-rl の OnPolicyRunner は、生成時に環境から num_envs / num_actions / get_observations()
    だけを使う(step() は学習時にしか呼ばない)ので、それだけを備えている。"""

    def __init__(self, obs_dim, num_actions):
        self.num_envs = 1
        self.num_actions = num_actions
        self.device = gs.device
        self.max_episode_length = 1
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=gs.device)
        self.cfg = {}
        self._obs = TensorDict({"policy": torch.zeros((1, obs_dim), device=gs.device)}, batch_size=[1])

    def get_observations(self):
        return self._obs

    def step(self, actions):
        raise NotImplementedError("この環境は方策の読み込み専用で、シミュレーションはできません")


def load_ai_policy(exp_name, run, ckpt):
    """学習済みの方策を読み込む。戻り値: (方策, env_cfg, obs_cfg, チェックポイントのパス)

    学習済みのランやチェックポイントが無ければ FileNotFoundError。"""
    from rsl_rl.runners import OnPolicyRunner

    log_dir = find_latest_run_dir(exp_name) if run == "latest" else os.path.join("logs", exp_name, run)
    ckpt_path = os.path.join(log_dir, f"model_{ckpt}.pt") if ckpt is not None else find_latest_checkpoint(log_dir)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, _reward_cfg, _command_cfg, _wind_cfg, train_cfg = pickle.load(f)

    num_actions = env_cfg["num_actions"]
    loader_env = _PolicyLoaderEnv(obs_dim=13 + num_actions, num_actions=num_actions)  # 13 = 位置3+姿勢4+速度3+角速度3
    runner = OnPolicyRunner(loader_env, train_cfg, None, device=gs.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=gs.device)
    return policy, env_cfg, obs_cfg, ckpt_path


# ---------------------------------------------------------------------- #
# 3者の操縦(ドライバ)。どれも rpms(target_abs) -> 4つのRPM を返す
# ---------------------------------------------------------------------- #
class AIDriver:
    """学習済み方策。観測の作り方は WindHoverEnv._update_observation と同一にしてある。"""

    def __init__(self, drone, policy, obs_cfg):
        self.drone = drone
        self.policy = policy
        self.scales = obs_cfg["obs_scales"]
        # WindHoverEnv は、観測に「1ステップ前の行動」ではなく「2ステップ前の行動」を入れている
        # (obs を作った後で last_actions を更新する順序のため)。学習時と同じ分布にするため、
        # その挙動をそのまま再現する(prev=直前の行動, prev_prev=その前の行動)。
        self.prev_actions = torch.zeros((1, 4), device=gs.device)
        self.prev_prev_actions = torch.zeros((1, 4), device=gs.device)

    def start(self):
        pass

    def stop(self):
        pass

    def rpms(self, target_abs):
        pos = self.drone.get_pos().reshape(1, 3)
        quat = self.drone.get_quat().reshape(1, 4)
        vel = self.drone.get_vel().reshape(1, 3)
        ang = self.drone.get_ang().reshape(1, 3)
        target = torch.as_tensor(target_abs, dtype=pos.dtype, device=pos.device).reshape(1, 3)

        inv_q = inv_quat(quat)
        lin_vel_body = transform_by_quat(vel, inv_q)  # 機体座標系の速度・角速度(学習時と同じ)
        ang_vel_body = transform_by_quat(ang, inv_q)
        obs = torch.cat(
            [
                torch.clip((target - pos) * self.scales["rel_pos"], -1, 1),
                quat,
                torch.clip(lin_vel_body * self.scales["lin_vel"], -1, 1),
                torch.clip(ang_vel_body * self.scales["ang_vel"], -1, 1),
                self.prev_prev_actions,
            ],
            dim=-1,
        )
        with torch.no_grad():
            actions = self.policy(TensorDict({"policy": obs}, batch_size=[1]))
        actions = torch.clip(actions, -1.0, 1.0)
        self.prev_prev_actions = self.prev_actions
        self.prev_actions = actions.clone()
        rpms = (1 + actions[0] * AI_RPM_SCALE) * BASE_RPM
        return [max(0.0, min(float(r), MAX_ABS_RPM)) for r in rpms]


class PIDDriver:
    """カスケードPID(manual_pid_fly.py と同じ。ゲインは公式の検証済み値)が、現在の目標を自動で追う。"""

    def __init__(self, drone):
        self.controller = DronePIDController(drone=drone, dt=DT, base_rpm=BASE_RPM, pid_params=OFFICIAL_PID_PARAMS)

    def start(self):
        pass

    def stop(self):
        pass

    def rpms(self, target_abs):
        rpms = self.controller.update(np.asarray(target_abs, dtype=float))
        return [max(PID_MIN_RPM, min(float(r), PID_MAX_RPM)) for r in rpms]


class HumanDriver:
    """人間がキーボードで操縦する(manual_raw_fly.py の ManualFlightController を使う)。"""

    def __init__(self, controller):
        self.controller = controller

    def start(self):
        self.controller.set_armed(True)   # 競技開始と同時に自動でアームする(全員が同時スタートになるように)

    def stop(self):
        self.controller.set_armed(False)

    def rpms(self, target_abs):
        return [max(0.0, min(float(r), MAX_ABS_RPM)) for r in self.controller.update(DT)]


# ---------------------------------------------------------------------- #
# 競技者(1機分のドローン・目標の球・採点・ドライバをまとめたもの)
# ---------------------------------------------------------------------- #
class Contestant:
    def __init__(self, key, drone, marker, enable_prop_fix):
        self.key = key
        self.label = LABELS[key]
        self.lane_x = LANE_X[key]
        self.drone = drone
        self.marker = marker
        self.com_link_idx = drone.get_link("base_link").idx
        self.start_pos = drone.get_pos().clone()
        self.start_quat = drone.get_quat().clone()
        self.prop_visual = PropellerVisual(drone, DT, enabled=enable_prop_fix)
        self.score = None
        self.driver = None
        self.freeze_pos = None
        self.freeze_quat = None
        self.last_dist = float("nan")

    def pin(self, pos=None, quat=None):
        """指定の位置・姿勢(既定は開始位置)に固定し、プロペラを止める"""
        self.drone.set_pos(self.start_pos if pos is None else pos, zero_velocity=True)
        self.drone.set_quat(self.start_quat if quat is None else quat, zero_velocity=True)
        self.drone.set_propellers_rpm([0.0, 0.0, 0.0, 0.0])

    def freeze_here(self):
        self.freeze_pos = self.drone.get_pos().clone()
        self.freeze_quat = self.drone.get_quat().clone()

    def update_marker(self):
        target = torch.as_tensor(self.score.current_target, dtype=torch.float32, device=gs.device)
        self.marker.set_pos(target, zero_velocity=True)


# ---------------------------------------------------------------------- #
# 結果の表示・保存
# ---------------------------------------------------------------------- #
def format_results(ranking, meta):
    lines = [
        f"seed={meta['seed']}  制限時間={meta['time_limit']:.0f}秒  目標数={meta['waypoints']}  "
        f"風={'あり' if meta['wind'] else 'なし'}",
        "",
        "順位  名前   合計点  到達   完走/終了時刻  平均誤差   内訳(到達/速度/接近/完走/ペナルティ)   状態",
    ]
    for i, (key, sc) in enumerate(ranking, 1):
        s = sc.summary()
        end = "-" if s["end_time"] is None else f"{s['end_time']:.1f}s"
        err = "-" if s["mean_error_m"] is None else f"{s['mean_error_m']:.2f}m"
        status = s["status"] + (f": {s['fail_reason']}" if s["fail_reason"] else "")
        lines.append(
            f"{i:>3}   {LABELS[key]:<4} {s['total']:>7.1f}  {s['captured']}/{s['waypoints']}   {end:>9}    {err:>7}   "
            f"{s['pts_capture']:.0f}/{s['pts_speed']:.0f}/{s['pts_near']:.0f}/{s['pts_finish']:.0f}/{s['pts_penalty']:.0f}"
            f"   {status}"
        )
    return "\n".join(lines)


def save_results(ranking, meta, ai_ckpt):
    """結果を logs/arena/ に保存する(ラウンドごとのレポート + 全ラウンドを通した results.csv)"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESULTS_DIR, f"round_{stamp}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 比較アリーナ 結果\n\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"AIのチェックポイント: {ai_ckpt if ai_ckpt else '(AIなし)'}\n\n")
        f.write("```\n" + format_results(ranking, meta) + "\n```\n")

    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["日時", "seed", "制限時間", "目標数", "風", "順位", "名前", "合計点", "到達数", "終了時刻",
                             "平均誤差m", "到達点", "速度点", "接近点", "完走点", "ペナルティ", "状態", "失敗理由",
                             "AIチェックポイント"])
        for i, (key, sc) in enumerate(ranking, 1):
            s = sc.summary()
            writer.writerow([stamp, meta["seed"], meta["time_limit"], meta["waypoints"], int(meta["wind"]), i,
                             LABELS[key], s["total"], s["captured"], s["end_time"], s["mean_error_m"], s["pts_capture"],
                             s["pts_speed"], s["pts_near"], s["pts_finish"], s["pts_penalty"], s["status"],
                             s["fail_reason"], ai_ckpt or ""])
    return report_path, csv_path


# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="wind-hovering", help="AIの実験名")
    parser.add_argument("--run", type=str, default="latest", help="AIのラン('latest'またはタイムスタンプ)")
    parser.add_argument("--ckpt", type=int, default=None, help="AIのイテレーション番号。省略時は最新")
    parser.add_argument("--no-ai", action="store_true", help="AI無し(PIDと人間だけ)で実行する")
    parser.add_argument("--time-limit", type=float, default=60.0, help="1ラウンドの制限時間[秒]")
    parser.add_argument("--waypoints", type=int, default=3, help="1ラウンドの目標の数")
    parser.add_argument("--seed", type=int, default=None, help="目標と風の乱数シード。省略時はランダム(画面に表示)")
    parser.add_argument("--no-wind", action="store_true", help="風を無効にする")
    parser.add_argument("--gpu", action="store_true", help="GPUで実行する(既定はCPU。3機だけなのでCPUで十分)")
    parser.add_argument("--no-prop-fix", action="store_true", help="プロペラ回転の表示補正を無効にする")
    parser.add_argument("--flag-style", choices=ALL_STYLES, default=None,
                        help="風を映す旗の種類。surface=式で形を計算するスプライン曲面(既定) / cloth-mesh=PBD布 / cloth=PBD粒子 / "
                             "cloth-vel=PBD布(粒子の速度に風を足す方式) / plate=剛体板")
    parser.add_argument("--pbd", action="store_true",
                        help="旗をPBD布の物理計算にする(--flag-style cloth-mesh の短縮形)")
    args = parser.parse_args()
    if args.pbd and args.flag_style not in (None, "cloth-mesh"):
        parser.error("--pbd は --flag-style cloth-mesh と同じ意味です。他の旗のスタイルとは併用できません")
    args.flag_style = "cloth-mesh" if args.pbd else (args.flag_style or "surface")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    # ---- AIの読み込み(学習済みのランが無ければ、AI無しで続行する) ----
    ai = None
    if not args.no_ai:
        try:
            ai = load_ai_policy(args.exp_name, args.run, args.ckpt)
            print(f"=== AIを読み込みました: {ai[3]} ===")
        except FileNotFoundError as e:
            print(f"[注意] 学習済みのAIが見つからないため、AI無し(PIDと人間)で実行します: {e}")
    ai_policy, ai_env_cfg, ai_obs_cfg, ai_ckpt = ai if ai else (None, None, None, None)

    rules = FailureRules.from_env_cfg(ai_env_cfg) if ai_env_cfg else FailureRules()
    scoring_cfg = ScoringConfig()

    # ---- シーン(物理設定は学習時の WindHoverEnv と同じ) ----
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        rigid_options=gs.options.RigidOptions(
            dt=DT / 2,
            enable_collision=True,
            enable_joint_limit=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        vis_options=reflective_vis_options(show_world_frame=False),  # 床を若干反射させる
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -6.0, 2.0),
            camera_lookat=(0.0, 0.0, 1.2),
            camera_fov=50,
            realtime_factor=1.0,
            run_in_thread=False,
        ),
        show_viewer=True,
        **wind_flag_scene_kwargs(args.flag_style, WIND_FLAG_SCALE),  # 布の旗用: 布と機体・地面の連成を切るなど
    )
    add_floor(scene)

    keys = (["AI"] if ai else []) + ["PID", "HUMAN"]
    drones, markers = {}, {}
    for key in keys:
        drones[key] = scene.add_entity(
            morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(LANE_X[key], 0.0, START_HEIGHT)),
        )
        markers[key] = scene.add_entity(
            morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.06, fixed=False, collision=False),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=MARKER_COLORS[key])),
        )
    wind_flag = create_wind_flag(scene, base_xy=WIND_FLAG_XY, scale=WIND_FLAG_SCALE, style=args.flag_style,
                                 along=WIND_FLAG_ALONG)
    scene.build()
    wind_flag.after_build()  # 布の旗は、build の後に支柱へ留める

    contestants = [Contestant(k, drones[k], markers[k], not args.no_prop_fix) for k in keys]
    by_key = {c.key: c for c in contestants}

    if not check_native_attitude_torque(scene, by_key["PID"].drone, by_key["PID"].start_pos, by_key["PID"].start_quat):
        gs.logger.error(
            "このGenesisでは、差動RPMがロール/ピッチのトルクを生成しません(1.2.3以前の不具合)。"
            "genesis-world>=1.4.1 に更新してください(verify_physics.py のテスト4で確認できます)。"
        )
        return

    human_controller = ManualFlightController(by_key["HUMAN"].drone)

    # ---- キー操作(人間用) ----
    state = READY
    pending_enter = False
    is_running = True

    def request_enter():
        nonlocal pending_enter
        pending_enter = True

    def stop():
        nonlocal is_running
        is_running = False

    def axis_keybinds(name, key_pos, key_neg, setter):
        return [
            Keybind(f"{name}_pos_hold", key_pos, KeyAction.HOLD, callback=setter, args=(1.0,)),
            Keybind(f"{name}_pos_release", key_pos, KeyAction.RELEASE, callback=setter, args=(0.0,)),
            Keybind(f"{name}_neg_hold", key_neg, KeyAction.HOLD, callback=setter, args=(-1.0,)),
            Keybind(f"{name}_neg_release", key_neg, KeyAction.RELEASE, callback=setter, args=(0.0,)),
        ]

    scene.viewer.register_keybinds(
        *axis_keybinds("pitch", Key.RIGHT, Key.LEFT, human_controller.set_pitch),
        *axis_keybinds("roll", Key.DOWN, Key.UP, human_controller.set_roll),
        *axis_keybinds("yaw", Key.E, Key.Q, human_controller.set_yaw),
        *axis_keybinds("throttle", Key.SPACE, Key.X, human_controller.set_throttle),
        Keybind("enter", Key.RETURN, KeyAction.RELEASE, callback=request_enter),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )
    hover_key = register_first_free_key(
        scene.viewer, ["B", "V", "G", "C", "N", "M"], "hover_hold", human_controller.zero_vertical_speed
    )

    # ---- ラウンドの準備 ----
    base_seed = args.seed if args.seed is not None else random.randint(0, 9999)
    round_no = 0
    wind = None
    meta = {}

    def setup_round():
        nonlocal wind, meta
        seed = base_seed + round_no
        waypoints_rel = generate_waypoints(seed, args.waypoints)
        wind = SingleDroneWind(seed=seed) if not args.no_wind else None
        meta = {"seed": seed, "time_limit": args.time_limit, "waypoints": args.waypoints, "wind": not args.no_wind}
        for c in contestants:
            offset = np.array([c.lane_x, 0.0, 0.0])
            c.score = ContestantScore(waypoints_rel + offset, args.time_limit, scoring_cfg, rules)
            c.freeze_pos = c.freeze_quat = None
            if c.key == "AI":
                c.driver = AIDriver(c.drone, ai_policy, ai_obs_cfg)
            elif c.key == "PID":
                c.driver = PIDDriver(c.drone)
            else:
                human_controller.set_armed(False)
                c.driver = HumanDriver(human_controller)
            c.update_marker()
        print(f"\n=== ラウンド{round_no + 1}  seed={seed}  目標数={args.waypoints}  制限時間={args.time_limit:.0f}秒  "
              f"風={'なし' if args.no_wind else 'あり'} ===")
        print("準備ができたら Enter キーでスタート(3秒のカウントダウン後、全員が同時に始まります)")

    setup_round()

    print("\n=== 操作方法(人間) ===")
    print("Space / X : 目標の上昇速度を上げる / 下げる(離しても保持)")
    if hover_key is not None:
        print(f"{hover_key}         : 目標上下速度を即座に0へ(ホバリング)")
    print("Q / E     : 左旋回 / 右旋回")
    print("↑ / ↓     : 奥 / 手前へ傾ける     ← / → : 左 / 右へ傾ける")
    print("Enter     : スタート / 次のラウンド      Esc : 終了")
    print("レーン: 左から " + " / ".join(f"{LABELS[k]}" for k in keys))
    print(legend(args.flag_style))

    # ---- メインループ ----
    countdown_steps = int(COUNTDOWN_S / DT)
    countdown_left = 0
    sim_time = 0.0
    step_count = 0
    last_shown_count = None

    try:
        while is_running:
            # --- 状態遷移(Enterキー) ---
            if pending_enter:
                pending_enter = False
                if state == READY:
                    state, countdown_left = COUNTDOWN, countdown_steps
                    last_shown_count = None
                elif state == FINISHED:
                    round_no += 1
                    setup_round()
                    state = READY

            # --- 待機・カウントダウン: 全機を開始位置に固定して、プロペラは止めておく ---
            if state in (READY, COUNTDOWN):
                for c in contestants:
                    c.pin()
                scene.step()
                wind_flag.update(NO_WIND, DT)   # 競技前は風が無いので、旗は垂れ下がる
                if state == COUNTDOWN:
                    countdown_left -= 1
                    seconds_left = int(np.ceil(countdown_left * DT))
                    if seconds_left != last_shown_count and countdown_left > 0:
                        print(f"  {seconds_left} ...")
                        last_shown_count = seconds_left
                    if countdown_left <= 0:
                        print("  スタート!")
                        for c in contestants:
                            c.driver.start()
                        sim_time = 0.0
                        step_count = 0
                        state = RUNNING
                continue

            # --- 競技終了後: 結果を表示したまま、全機を固定して待つ ---
            if state == FINISHED:
                for c in contestants:
                    c.pin(c.freeze_pos, c.freeze_quat)
                scene.step()
                wind_flag.update(NO_WIND, DT)
                continue

            # --- 競技中 ---
            for c in contestants:
                if c.score.running:
                    rpms = c.driver.rpms(c.score.current_target)
                    c.drone.set_propellers_rpm(rpms)
                    c.prop_visual.apply(rpms)  # 表示用のプロペラ角度だけを上書き(物理には影響しない)
                else:
                    c.pin(c.freeze_pos, c.freeze_quat)

            # 同じ風を、競技中の全機の重心に加える
            wind_force = NO_WIND
            if wind is not None:
                wind_force = wind.step(DT)
                for c in contestants:
                    if c.score.running:
                        apply_com_force(c.drone, c.com_link_idx, wind_force)
            wind_flag.update(wind_force, DT)   # 旗にも同じ風を映す(強さと向きがそのまま分かる)

            scene.step()
            sim_time += DT
            step_count += 1

            # 採点(このステップ終了後の状態で)
            for c in contestants:
                if not c.score.running:
                    continue
                pos = np.array([float(v) for v in c.drone.get_pos()])
                roll, pitch, _yaw = (float(a) for a in quat_to_xyz(c.drone.get_quat(), rpy=True, degrees=True))
                ang_speed = float(torch.norm(c.drone.get_ang()))
                c.last_dist = float(np.linalg.norm(pos - c.score.current_target))
                for event in c.score.update(sim_time, DT, pos, roll, pitch, ang_speed):
                    print(f"\n[{sim_time:5.1f}s] {c.label}: {event}")
                if c.score.running:
                    c.update_marker()  # 目標が進んだ場合に、球を次の目標へ動かす
                else:
                    c.freeze_here()    # 完走・失敗した機体は、その場に固定する
                    c.driver.stop()

            # 制限時間・全員終了の判定
            for c in contestants:
                c.score.time_up(sim_time)
                if not c.score.running and c.freeze_pos is None:
                    c.freeze_here()
            if all(not c.score.running for c in contestants):
                state = FINISHED
                ranking = rank({c.key: c.score for c in contestants})
                print("\n\n" + "=" * 78 + "\n" + format_results(ranking, meta) + "\n" + "=" * 78)
                report_path, csv_path = save_results(ranking, meta, ai_ckpt)
                print(f"結果を保存しました: {report_path}  /  累計: {csv_path}")
                print("次のラウンドは Enter キー(目標と風が変わります)。終了は Esc。")
                continue

            # 0.25秒ごとに状態を1行で表示
            if step_count % 25 == 0:
                parts = []
                for c in contestants:
                    sc = c.score
                    if sc.running:
                        parts.append(f"{c.label} {sc.total:6.1f}点 目標{sc.wp_index + 1}/{args.waypoints} 距離{c.last_dist:4.2f}m")
                    else:
                        parts.append(f"{c.label} {sc.total:6.1f}点 [{sc.status}]")
                print(f"\r残り{max(0.0, args.time_limit - sim_time):5.1f}s | 風{wind_flag.strength:5.3f}N | "
                      + " | ".join(parts) + "   ", end="", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        print()
        gs.logger.info("終了しました。")


if __name__ == "__main__":
    main()