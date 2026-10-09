"""
manual_raw_fly.py (完全手動 / 本物の操縦感を追求した版)

実際のドローン操縦機(モード2)に合わせた4軸操作:

  - スロットル(W/S): 直接出力。離しても中央に戻らない(実機のスロットル
    スティックと同じ)。高度は自動では維持されない ― 触らなければ
    上昇/下降し続ける(Acroモードの実機と同じ、自動ホバリング無し)。
  - ヨー(A/D)   : スティック位置は「回転速度」の指示。離せばその場の
    機首方向をそのまま保持する(絶対角度を指示しているわけではない)。
  - ロール(←/→) : スティック位置が目標の傾斜角になり、離せば水平に戻る
    (セルフレベル=Angleモード。多くの民生ドローンのデフォルト)。
  - ピッチ(↑/↓) : 同上。

さらに実機同様の「アーム(武装)」操作を入れている: 起動直後はプロペラは
回転しておらず、Enterキーでアームするまで飛行できない。

ロール/ピッチ/ヨーの姿勢PIDゲインは、quadcopter_controller.py / fly_route.py
で検証済みの値をそのまま流用している(自前でゲインを推測すると、本プロジェクトで
過去に経験した通り不安定化のリスクがあるため)。

操作方法:
  W / S      : スロットルを上げる / 下げる(離しても値は保持される)
  A / D      : 左旋回 / 右旋回(ヨー、離すと機首方向を保持)
  ↑ / ↓      : 前傾 / 後傾(ピッチ、離すと水平に戻る)
  ← / →      : 左傾 / 右傾(ロール、離すと水平に戻る)
  Enter      : アーム / ディスアーム切り替え
  Esc        : 終了

ベース(姿勢PIDゲイン・ミキサー式を流用):
  https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/quadcopter_controller.py
  https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/fly_route.py
"""

import argparse
import math

import numpy as np
import torch

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind
from genesis.utils.geom import quat_to_xyz

DT = 0.01
BASE_RPM = 14468.429183500699  # CF2Xのホバリング基準RPM
MIN_RPM = 0.0
MAX_RPM = 25000.0  # interactive_drone.py と同じ、モーターの物理的な上限


# ---------------------------------------------------------------------- #
# 姿勢PID(quadcopter_controller.py の PIDController をそのまま使用)
# ---------------------------------------------------------------------- #
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


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
    MAX_TILT_DEG = 20.0          # ロール/ピッチの最大傾斜角
    YAW_RATE_DEG_S = 90.0        # ヨースティック最大時の回転速度
    THROTTLE_RATE_RPM_S = 6000.0  # スロットルスティックを倒している間の変化速度
    THROTTLE_MIN_OFFSET = -0.35 * BASE_RPM
    THROTTLE_MAX_OFFSET = 0.6 * BASE_RPM

    def __init__(self, drone):
        self.drone = drone
        self._pid_roll = PIDController(10.0, 0.0, 1.0)
        self._pid_pitch = PIDController(10.0, 0.0, 1.0)
        self._pid_yaw = PIDController(2.0, 0.0, 0.2)

        # スティック入力(-1.0 ~ 1.0)。キーのHOLD/RELEASEで更新される
        self.roll_stick = 0.0
        self.pitch_stick = 0.0
        self.yaw_stick = 0.0
        self.throttle_stick = 0.0

        self.target_yaw_deg = 0.0      # ヨーは絶対角度を積分で保持(レート制御の結果)
        self.throttle_offset = 0.0     # スロットルはセルフセンタリングしない
        self.armed = False

    # ---- キーコールバック ---- #
    def set_roll(self, v):
        self.roll_stick = v

    def set_pitch(self, v):
        self.pitch_stick = v

    def set_yaw(self, v):
        self.yaw_stick = v

    def set_throttle(self, v):
        self.throttle_stick = v

    def toggle_arm(self):
        self.armed = not self.armed
        if self.armed:
            # アーム時は積分項をリセットし、スロットルはアイドル(0オフセット=ホバリング相当)から開始
            self._pid_roll.integral = 0.0
            self._pid_pitch.integral = 0.0
            self._pid_yaw.integral = 0.0
            self.throttle_offset = 0.0
            att = quat_to_xyz(self.drone.get_quat(), rpy=True, degrees=True)
            self.target_yaw_deg = float(att[2])  # 現在の機首方向を基準にする
        print(f"=== {'アーム' if self.armed else 'ディスアーム'} ===")

    # ---- 毎ステップ呼び出し ---- #
    def update(self, dt):
        if not self.armed:
            return [0.0, 0.0, 0.0, 0.0]

        att = quat_to_xyz(self.drone.get_quat(), rpy=True, degrees=True)

        # ヨー: スティックは回転速度の指示。離せば機首方向を保持する
        self.target_yaw_deg += self.yaw_stick * self.YAW_RATE_DEG_S * dt

        # スロットル: セルフセンタリングしない直接出力(実機のスロットルと同じ)
        self.throttle_offset += self.throttle_stick * self.THROTTLE_RATE_RPM_S * dt
        self.throttle_offset = max(
            self.THROTTLE_MIN_OFFSET, min(self.throttle_offset, self.THROTTLE_MAX_OFFSET)
        )

        # ロール/ピッチ: スティック位置がそのまま目標傾斜角(セルフレベル)
        target_roll = self.roll_stick * self.MAX_TILT_DEG
        target_pitch = self.pitch_stick * self.MAX_TILT_DEG

        err_roll = target_roll - float(att[0])
        err_pitch = target_pitch - float(att[1])
        err_yaw = self.target_yaw_deg - float(att[2])

        roll_del = self._pid_roll.update(err_roll, dt)
        pitch_del = self._pid_pitch.update(err_pitch, dt)
        yaw_del = self._pid_yaw.update(err_yaw, dt)

        thrust = self.throttle_offset
        # quadcopter_controller.py のミキサー式と同じパターン(x_vel/y_velは
        # 位置制御の外側ループが無いのでゼロ扱い)
        m1 = BASE_RPM + (thrust - roll_del - pitch_del - yaw_del)
        m2 = BASE_RPM + (thrust - roll_del + pitch_del + yaw_del)
        m3 = BASE_RPM + (thrust + roll_del + pitch_del - yaw_del)
        m4 = BASE_RPM + (thrust + roll_del - pitch_del + yaw_del)
        return [m1, m2, m3, m4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wind", action="store_true", help="AI/PIDモードと同じOU過程の風外乱を有効にする")
    parser.add_argument("--seed", type=int, default=None, help="風のパラメータのシード")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
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

    controller = ManualFlightController(drone)
    wind = SingleDroneWind(seed=args.seed) if args.wind else None

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    def axis_keybinds(name, key_pos, key_neg, setter):
        """押している間+1/-1、離すと0に戻る軸(ロール/ピッチ/ヨー用)"""
        return [
            Keybind(f"{name}_pos_hold", key_pos, KeyAction.HOLD, callback=setter, args=(1.0,)),
            Keybind(f"{name}_pos_release", key_pos, KeyAction.RELEASE, callback=setter, args=(0.0,)),
            Keybind(f"{name}_neg_hold", key_neg, KeyAction.HOLD, callback=setter, args=(-1.0,)),
            Keybind(f"{name}_neg_release", key_neg, KeyAction.RELEASE, callback=setter, args=(0.0,)),
        ]

    scene.viewer.register_keybinds(
        *axis_keybinds("pitch", Key.UP, Key.DOWN, controller.set_pitch),
        *axis_keybinds("roll", Key.RIGHT, Key.LEFT, controller.set_roll),
        *axis_keybinds("yaw", Key.D, Key.A, controller.set_yaw),
        *axis_keybinds("throttle", Key.W, Key.S, controller.set_throttle),
        Keybind("arm_toggle", Key.RETURN, KeyAction.RELEASE, callback=controller.toggle_arm),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )

    print("\n=== 操作方法(完全手動。実機のモード2送信機に準拠) ===")
    print("W / S       : スロットル増加 / 減少(離しても値を保持)")
    print("A / D       : 左旋回 / 右旋回(ヨー)")
    print("↑ / ↓       : 前傾 / 後傾(ピッチ)")
    print("← / →       : 左傾 / 右傾(ロール)")
    print("Enter       : アーム / ディスアーム")
    print("Esc         : 終了")
    print("\n起動直後はディスアーム状態です。Enterキーでアームしてください。")
    if wind is not None:
        print(f"風: 有効(ベース強さ {np.linalg.norm(wind.mean):.3f} N、AI/PIDモードと同じモデル)")
    else:
        print("風: 無効(--wind で有効化)")

    try:
        while is_running:
            rpms = controller.update(DT)
            rpms = [max(MIN_RPM, min(r, MAX_RPM)) for r in rpms]
            drone.set_propellers_rpm(rpms)

            if wind is not None and controller.armed:
                w = wind.step(DT)
                drone.solver.apply_links_external_force(
                    torch.tensor([list(w)], dtype=torch.float32),
                    links_idx=[com_link_idx],
                    ref="link_com",
                    local=False,
                )

            scene.step()
    except KeyboardInterrupt:
        pass
    finally:
        gs.logger.info("終了しました。")


if __name__ == "__main__":
    main()