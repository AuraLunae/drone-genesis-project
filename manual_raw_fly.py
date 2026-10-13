"""
manual_raw_fly.py (完全手動 / 実機のモード2送信機に準拠)

実際のドローン操縦機(モード2)に合わせた4軸操作:

  - 上下(Space / X) : 目標の「上下速度」を変える。離しても目標は保持され、
                      ホバリングキー(下記)で0に戻せる。
  - ヨー(Q / E)     : スティック位置は「回転速度」の指示。離せばその場の
                      機首方向を保持する(絶対角度を指示しているわけではない)。
  - ロール/ピッチ(矢印キー): スティック位置が目標の傾斜角になり、離せば水平に
                      戻る(セルフレベル=Angleモード。多くの民生ドローンの既定)。

さらに実機同様の「アーム(武装)」操作がある: 起動直後はプロペラが回転して
おらず、Enterキーでアームするまで飛行できない。

操作方法:
  Space      : 目標の上昇速度を上げる(離しても目標速度は保持される)
  X          : 目標速度を下げる(0で停止、負で下降)
  B          : 目標上下速度を即座に0へ戻す(ホバリング)。Genesisのビューアが
               標準機能に使っているキーと衝突する場合は、V, G, C, N, M の順に
               空いているキーへ自動で割り当てる(割り当て結果は起動時に表示)
  Q / E      : 左旋回 / 右旋回(ヨー、離すと機首方向を保持)
  ↑ / ↓      : 起動時のカメラから見て 奥 / 手前 へ傾ける(ロール)
  ← / →      : 起動時のカメラから見て 左 / 右 へ傾ける(ピッチ)
  Enter      : アーム / ディスアーム(どちらも開始位置に戻る)
  Esc        : 終了
  ※ 傾きは機体基準(実機と同じ)なので、ヨーで向きを変えると、矢印キーで進む
    方向もそれに合わせて変わる。

オプション:
  --wind     : AI/PIDモードと同じOU過程の風外乱を有効にする
  --drag K   : 線形の空力抵抗(1/s)を加える。既定0(=抵抗なし)。AI/PIDモードには
               抵抗が無いため、それらと比較する時は指定しないこと。目安は0.5程度。

Genesis 1.4.x 前提:
  1.2.3 では差動RPMがロール/ピッチのトルクを生まない不具合があった(verify_physics.py
  で実測確認)。1.4.1 では物理側が正しくトルクを出すため、手動のトルク補正は不要
  (残すと二重計上になる)。起動時に自己診断を行い、古い版なら理由を表示して止まる。

ベース(ミキサー式・PIDの考え方):
  https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/quadcopter_controller.py
"""

import argparse
import math

import numpy as np
import torch

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind
from genesis.utils.geom import quat_to_xyz

DT = 0.01
G = 9.81
BASE_RPM = 14468.429183500699  # CF2Xのホバリング基準RPM(verify_physics.pyで較正を確認済み)
MIN_RPM = 0.0
MAX_RPM = 25000.0  # interactive_drone.py と同じ、モーターの物理的な上限
MASS_KG = 0.027    # verify_physics.pyの実測(0.05 N / 1.8518 m/s^2 = 27.0 g)


def wrap_deg(angle):
    """角度を[-180, 180)に正規化する"""
    return (angle + 180.0) % 360.0 - 180.0


def approach(value, target, max_step):
    """valueをtargetへ、1回あたり最大max_stepだけ近づける"""
    if value < target:
        return min(value + max_step, target)
    return max(value - max_step, target)


# ---------------------------------------------------------------------- #
# 姿勢PID(微分項は誤差ではなく「測定値」の微分を使う: 目標が急変しても微分キックが出ない)
# ---------------------------------------------------------------------- #
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt, measurement_rate=None):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if measurement_rate is None else -measurement_rate
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


# ---------------------------------------------------------------------- #
# 風(OU過程)。他のスクリプトと同一モデル
# ---------------------------------------------------------------------- #
class SingleDroneWind:
    def __init__(self, strength_range=(0.02, 0.08), theta_range=(0.5, 2.0),
                 sigma_range=(0.02, 0.08), max_norm_multiplier=3.0, seed=None):
        rng = np.random.default_rng(seed)
        direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction) + 1e-8
        self.mean = direction * rng.uniform(*strength_range)
        self.theta = rng.uniform(*theta_range)
        self.sigma = rng.uniform(*sigma_range)
        self.max_norm = strength_range[1] * max_norm_multiplier
        self.vec = np.zeros(3)
        self._rng = rng

    def step(self, dt):
        noise = self._rng.normal(0, 1, 3)
        self.vec += self.theta * (self.mean - self.vec) * dt + self.sigma * math.sqrt(dt) * noise
        norm = np.linalg.norm(self.vec)
        if norm > self.max_norm:
            self.vec *= self.max_norm / norm
        return self.vec


# ---------------------------------------------------------------------- #
# 実機のモード2送信機を模した手動フライトコントローラ
# ---------------------------------------------------------------------- #
class ManualFlightController:
    MAX_TILT_DEG = 20.0        # ロール/ピッチの最大傾斜角
    YAW_RATE_DEG_S = 90.0      # ヨースティック最大時の回転速度
    STICK_RAMP_PER_S = 5.0     # キー入力が0→±1に達するまで約0.2秒(アナログスティックに近づける)
    VERTICAL_SPEED_RATE = 1.0  # [m/s per s] Space/X押下中に目標上下速度が変わる速さ
    MAX_VERTICAL_SPEED = 1.5   # [m/s]
    VERTICAL_SPEED_KP = 8.0
    MAX_VERTICAL_ACCEL = 6.0   # [m/s^2]
    MIN_TILT_COS = 0.5         # 傾き補償の下限(極端な傾きで推力が発散しないように)

    def __init__(self, drone):
        self.drone = drone
        # 姿勢PIDゲイン(verify_physics.pyの実測から再計算した値)
        #   roll/pitch: ±2170 RPMの差動で角加速度158.7 rad/s^2 → PID出力1 RPMあたり
        #     G = 4.19 deg/s^2。閉ループの固有角周波数 w = sqrt(G*Kp) = 6.5 rad/s、
        #     減衰比 z = G*Kd/(2w) = 0.32*Kd。Kd=2.5 で z≈0.8(程よい減衰)。
        #     (公式のKd=1は z≈0.3で振動的、補正トルクで効きが倍だった旧版のKd=5は z≈1.6で過減衰)
        #   yaw: 同じ差動で91.9 rad/s^2 → G = 2.43 deg/s^2/RPM。Kp=2で w=2.2 rad/s、
        #     Kd=1.5 で z≈0.8(公式のKd=0.2は z≈0.1でほぼ減衰しない)。
        self._pid_roll = PIDController(10.0, 0.0, 2.5)
        self._pid_pitch = PIDController(10.0, 0.0, 2.5)
        self._pid_yaw = PIDController(2.0, 0.0, 1.5)
        self.armed = False
        self._clear_inputs()

    def _clear_inputs(self):
        # キーの状態(-1/0/+1)
        self.roll_key = 0.0
        self.pitch_key = 0.0
        self.yaw_key = 0.0
        self.throttle_key = 0.0
        # なめらかに追従させたスティック値
        self.roll_stick = 0.0
        self.pitch_stick = 0.0
        self.yaw_stick = 0.0
        self.target_yaw_deg = 0.0
        self.target_vertical_speed = 0.0
        self._prev_attitude = None
        self._pid_roll.reset()
        self._pid_pitch.reset()
        self._pid_yaw.reset()

    # ---- キーコールバック ---- #
    def set_roll(self, v):
        self.roll_key = v

    def set_pitch(self, v):
        self.pitch_key = v

    def set_yaw(self, v):
        self.yaw_key = v

    def set_throttle(self, v):
        self.throttle_key = v

    def zero_vertical_speed(self):
        self.target_vertical_speed = 0.0

    def read_attitude(self):
        att = quat_to_xyz(self.drone.get_quat(), rpy=True, degrees=True)
        return np.array([float(a) for a in att], dtype=np.float64)

    def set_armed(self, armed):
        self._clear_inputs()
        self.armed = armed
        if armed:
            attitude = self.read_attitude()
            self.target_yaw_deg = float(attitude[2])  # 現在の機首方向を基準にする
            self._prev_attitude = attitude

    # ---- 毎ステップ呼び出し ---- #
    def update(self, dt):
        if not self.armed:
            return [0.0, 0.0, 0.0, 0.0]

        attitude = self.read_attitude()
        if self._prev_attitude is None:
            attitude_rate = np.zeros(3, dtype=np.float64)
        else:
            delta = attitude - self._prev_attitude
            delta[2] = wrap_deg(delta[2])
            attitude_rate = delta / dt
        self._prev_attitude = attitude
        roll, pitch, yaw = attitude

        # キーのオン/オフ入力を、スルーレート制限でアナログスティック風に滑らかにする
        ramp = self.STICK_RAMP_PER_S * dt
        self.roll_stick = approach(self.roll_stick, self.roll_key, ramp)
        self.pitch_stick = approach(self.pitch_stick, self.pitch_key, ramp)
        self.yaw_stick = approach(self.yaw_stick, self.yaw_key, ramp)

        # ヨー: スティックは回転速度の指示。離せば機首方向を保持する。
        # 目標ヨーは[-180, 180)に正規化して持ち、誤差も正規化する(折り返しで暴れない)
        self.target_yaw_deg = wrap_deg(self.target_yaw_deg + self.yaw_stick * self.YAW_RATE_DEG_S * dt)

        # 上下: Space/Xで目標上下速度を変える(離しても保持される)
        self.target_vertical_speed += self.throttle_key * self.VERTICAL_SPEED_RATE * dt
        self.target_vertical_speed = max(
            -self.MAX_VERTICAL_SPEED, min(self.target_vertical_speed, self.MAX_VERTICAL_SPEED)
        )

        # ロール/ピッチ: スティック位置がそのまま目標傾斜角(セルフレベル)
        target_roll = self.roll_stick * self.MAX_TILT_DEG
        target_pitch = self.pitch_stick * self.MAX_TILT_DEG

        roll_del = self._pid_roll.update(target_roll - roll, dt, measurement_rate=attitude_rate[0])
        pitch_del = self._pid_pitch.update(target_pitch - pitch, dt, measurement_rate=attitude_rate[1])
        yaw_del = self._pid_yaw.update(
            wrap_deg(self.target_yaw_deg - yaw), dt, measurement_rate=attitude_rate[2]
        )

        # 集合推力: 目標上下加速度から、推力∝rpm^2 の関係で厳密に求める。
        # 傾けると鉛直成分が cos(傾き) 倍に減って沈むので、その分を補償する。
        vertical_velocity = float(self.drone.get_vel()[2])
        vertical_accel = self.VERTICAL_SPEED_KP * (self.target_vertical_speed - vertical_velocity)
        vertical_accel = max(-self.MAX_VERTICAL_ACCEL, min(vertical_accel, self.MAX_VERTICAL_ACCEL))
        cos_tilt = max(math.cos(math.radians(roll)) * math.cos(math.radians(pitch)), self.MIN_TILT_COS)
        thrust_ratio = max((G + vertical_accel) / (G * cos_tilt), 0.0)
        collective_rpm = BASE_RPM * math.sqrt(thrust_ratio)

        # quadcopter_controller.py のミキサーと同じ符号パターン
        # (x_vel/y_velは位置制御の外側ループが無いのでゼロ扱い)
        m1 = collective_rpm - roll_del - pitch_del - yaw_del
        m2 = collective_rpm - roll_del + pitch_del + yaw_del
        m3 = collective_rpm + roll_del + pitch_del - yaw_del
        m4 = collective_rpm + roll_del - pitch_del + yaw_del
        return [m1, m2, m3, m4]


class PropellerVisual:
    """プロペラの回転「表示」だけを、実際のRPMとは切り離して滑らかにする(物理には無関係)。

    Genesisは、各ステップでプロペラの実際の回転量(rpmに比例)を表示角度に足し込む。
    CF2Xは約14,500rpm(毎秒240回転)で回るため、100Hz前後のコマ送りで表示すると
    ワゴンホイール効果が起き、1コマあたりの回転が180°付近になるRPMでは、対称な
    プロペラは止まって見える(加速でRPMが上がると止まって見え、戻ると回って見える)。

    そこで、実際のRPMに比例しつつ、1ステップあたりの表示回転角を小さく抑えた
    表示用の角度を自前で積算し、Genesisの内部カウンタ(_propellers_revs)に上書きする。
    内部カウンタが見つからない版では何もしない(self.enabled=False)。
    """

    VISUAL_SCALE = 0.04  # 実際の回転量に対する表示回転量の比。ホバリング付近で約35度/ステップ

    def __init__(self, drone, dt, enabled=True):
        self.drone = drone
        revs = getattr(drone, "_propellers_revs", None)
        self.enabled = bool(enabled) and isinstance(revs, torch.Tensor)
        # Genesisが「1回転」に対応させているカウンタ周期。内部と同じ 60/solver.dt を使う。
        # (食い違っても、周期が整数倍なら継ぎ目なく、そうでなくても180度のずれで済み、
        #  ブレードが対称なプロペラでは目立たない)
        self._wrap = 60.0 / float(getattr(drone.solver, "dt", dt))
        self._revs = None

    def apply(self, rpms):
        """set_propellers_rpm の直後、scene.step() の前に毎ステップ呼ぶ"""
        if not self.enabled:
            return
        revs = self.drone._propellers_revs  # 形状 (プロペラ数, 環境数)
        if self._revs is None:
            self._revs = torch.zeros_like(revs)
        rpm_tensor = torch.tensor([float(r) for r in rpms], dtype=revs.dtype, device=revs.device).reshape(-1, 1)
        self._revs = (self._revs + rpm_tensor * self.VISUAL_SCALE) % self._wrap
        revs.copy_(self._revs)


def register_first_free_key(viewer, candidate_names, name, callback):
    """ビューアの標準キーと衝突しない最初のキーに、コールバックを登録する。

    Genesisのビューアは Z(reset_camera)など、いくつかのキーを標準機能に使っている。
    どれが使用済みかは版によって異なるため、候補を順に試し、登録に成功した
    最初のキー名を返す(全て失敗したらNone)。"""
    for key_name in candidate_names:
        key = getattr(Key, key_name, None)
        if key is None:
            continue
        try:
            viewer.register_keybinds(Keybind(name, key, KeyAction.RELEASE, callback=callback))
            return key_name
        except ValueError:  # 既に別の機能に割り当て済み
            continue
    return None


def check_native_attitude_torque(scene, drone, pos, quat):
    """差動RPMが本当にロール/ピッチのトルクを生むかを3ステップで確認する。
    生まない版(Genesis 1.2.3など)では、PIDが機体を傾けられず操縦不能になる。"""
    delta = 0.15 * BASE_RPM
    pitch_pattern = [BASE_RPM + delta, BASE_RPM - delta, BASE_RPM - delta, BASE_RPM + delta]
    for _ in range(3):
        drone.set_propellers_rpm(pitch_pattern)
        scene.step()
    pitch_rate = abs(float(drone.get_ang()[1]))
    drone.set_pos(pos, zero_velocity=True)
    drone.set_quat(quat, zero_velocity=True)
    return pitch_rate > 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wind", action="store_true", help="AI/PIDモードと同じOU過程の風外乱を有効にする")
    parser.add_argument("--seed", type=int, default=None, help="風のパラメータのシード")
    parser.add_argument("--drag", type=float, default=0.0,
                        help="線形の空力抵抗[1/s]。既定0(AI/PIDモードと同条件)。目安は0.5程度")
    parser.add_argument("--no-prop-fix", action="store_true",
                        help="プロペラ回転の表示補正(ワゴンホイール効果対策)を無効にする")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        rigid_options=gs.options.RigidOptions(dt=DT / 2),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
            realtime_factor=1.0,
            run_in_thread=False,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 0.5)),
    )
    scene.viewer.follow_entity(drone)

    scene.build()
    com_link_idx = drone.get_link("base_link").idx
    disarmed_pos = drone.get_pos().clone()
    disarmed_quat = drone.get_quat().clone()

    if not check_native_attitude_torque(scene, drone, disarmed_pos, disarmed_quat):
        gs.logger.error(
            "このGenesisでは、差動RPMがロール/ピッチのトルクを生成しません(1.2.3以前の不具合)。"
            "genesis-world>=1.4.1 に更新してください(verify_physics.py のテスト4で確認できます)。"
        )
        return

    controller = ManualFlightController(drone)
    wind = SingleDroneWind(seed=args.seed) if args.wind else None
    prop_visual = PropellerVisual(drone, DT, enabled=not args.no_prop_fix)

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    def toggle_arm():
        # アーム/ディスアームのどちらも、開始位置・姿勢に戻す
        drone.set_pos(disarmed_pos, zero_velocity=True)
        drone.set_quat(disarmed_quat, zero_velocity=True)
        controller.set_armed(not controller.armed)
        print(f"\n=== {'アーム' if controller.armed else 'ディスアーム'} ===")

    def axis_keybinds(name, key_pos, key_neg, setter):
        """押している間+1/-1、離すと0に戻る軸"""
        return [
            Keybind(f"{name}_pos_hold", key_pos, KeyAction.HOLD, callback=setter, args=(1.0,)),
            Keybind(f"{name}_pos_release", key_pos, KeyAction.RELEASE, callback=setter, args=(0.0,)),
            Keybind(f"{name}_neg_hold", key_neg, KeyAction.HOLD, callback=setter, args=(-1.0,)),
            Keybind(f"{name}_neg_release", key_neg, KeyAction.RELEASE, callback=setter, args=(0.0,)),
        ]

    keybinds = [
        *axis_keybinds("pitch", Key.RIGHT, Key.LEFT, controller.set_pitch),
        *axis_keybinds("roll", Key.DOWN, Key.UP, controller.set_roll),
        *axis_keybinds("yaw", Key.E, Key.Q, controller.set_yaw),
        *axis_keybinds("throttle", Key.SPACE, Key.X, controller.set_throttle),
        Keybind("arm_toggle", Key.RETURN, KeyAction.RELEASE, callback=toggle_arm),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    ]
    scene.viewer.register_keybinds(*keybinds)

    # ホバリング用キーは、ビューアの標準キー(Zはreset_cameraなど)と衝突しない最初の候補を使う
    hover_key = register_first_free_key(
        scene.viewer, ["B", "V", "G", "C", "N", "M"], "hover_hold", controller.zero_vertical_speed
    )

    print("\n=== 操作方法(完全手動。実機のモード2送信機に準拠) ===")
    print("Space       : 目標の上昇速度を上げる(離しても保持)")
    print("X           : 目標速度を下げる(0で停止、負で下降)")
    if hover_key is not None:
        print(f"{hover_key}           : 目標上下速度を即座に0へ(ホバリング)")
    else:
        print("(ホバリング用キーは、空いているキーが無く無効。Xで目標速度を戻してください)")
    print("Q / E       : 左旋回 / 右旋回(ヨー)")
    print("↑ / ↓       : 奥 / 手前へ傾ける(起動時のカメラ基準)")
    print("← / →       : 左 / 右へ傾ける(起動時のカメラ基準)")
    print("Enter       : アーム / ディスアーム(開始位置に戻る)")
    print("Esc         : 終了")
    print("\n起動直後はディスアーム状態です。Enterキーでアームしてください。")
    if wind is not None:
        print(f"風: 有効(ベース強さ {np.linalg.norm(wind.mean):.3f} N、AI/PIDモードと同じモデル)")
    else:
        print("風: 無効(--wind で有効化)")
    if args.drag > 0.0:
        print(f"空力抵抗: 有効(係数 {args.drag} 1/s)。AI/PIDモードには無い物理なので比較時は注意")
    if prop_visual.enabled:
        print("プロペラ表示補正: 有効(加速で止まって見える現象への対策。物理には影響しません)")
    else:
        print("プロペラ表示補正: 無効(--no-prop-fix 指定、またはこの版に内部カウンタが見つかりません)")

    step_count = 0
    try:
        while is_running:
            if not controller.armed:
                drone.set_pos(disarmed_pos, zero_velocity=True)
                drone.set_quat(disarmed_quat, zero_velocity=True)

            rpms = controller.update(DT)
            rpms = [max(MIN_RPM, min(r, MAX_RPM)) for r in rpms]
            drone.set_propellers_rpm(rpms)
            prop_visual.apply(rpms)  # 表示用のプロペラ角度だけを上書き(物理には影響しない)

            # 外力(風 + 任意の空力抵抗)は、1ステップに1回の呼び出しにまとめて重心に加える
            if controller.armed and (wind is not None or args.drag > 0.0):
                force = np.zeros(3)
                if wind is not None:
                    force += wind.step(DT)
                if args.drag > 0.0:
                    velocity = np.array([float(v) for v in drone.get_vel()])
                    force -= args.drag * MASS_KG * velocity
                drone.solver.apply_links_external_wrench(
                    force=torch.tensor([force.tolist()], dtype=torch.float32),
                    links_idx=[com_link_idx],
                    ref=gs.link_ref_frame.link_COM,
                    local=False,
                )

            scene.step()

            # 0.25秒ごとに状態を1行で表示(実機のOSDに相当)
            step_count += 1
            if controller.armed and step_count % 25 == 0:
                pos = [float(v) for v in drone.get_pos()]
                vel = [float(v) for v in drone.get_vel()]
                roll, pitch, yaw = controller.read_attitude()
                horizontal_speed = math.hypot(vel[0], vel[1])
                print(
                    f"\r高度 {pos[2]:5.2f}m | 上下 {vel[2]:+5.2f}(目標 {controller.target_vertical_speed:+4.2f})m/s"
                    f" | 水平 {horizontal_speed:4.2f}m/s | R{roll:+6.1f} P{pitch:+6.1f} Y{yaw:+7.1f}deg"
                    f" | {sum(rpms) / 4:6.0f}rpm   ",
                    end="", flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        print()
        gs.logger.info("終了しました。")


if __name__ == "__main__":
    main()