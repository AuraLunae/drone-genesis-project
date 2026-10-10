"""
manual_raw_fly.py (完全手動 / 本物の操縦感を追求した版)

実際のドローン操縦機(モード2)に合わせた4軸操作:

    - 上下速度(Space/X): 目標の上昇/下降速度を変える。目標を0に戻すと自動で減速する。
    - ヨー(Q/E)   : スティック位置は「回転速度」の指示。離せばその場の
    機首方向をそのまま保持する(絶対角度を指示しているわけではない)。
    - ロール(↑/↓) : スティック位置が目標の傾斜角になり、離せば水平に戻る
    (セルフレベル=Angleモード。多くの民生ドローンのデフォルト)。
    - ピッチ(←/→) : 同上。

さらに実機同様の「アーム(武装)」操作を入れている: 起動直後はプロペラは
回転しておらず、Enterキーでアームするまで飛行できない。

ロール/ピッチ/ヨーの姿勢PIDゲインは、quadcopter_controller.py / fly_route.py
で検証済みの値をそのまま流用している(自前でゲインを推測すると、本プロジェクトで
過去に経験した通り不安定化のリスクがあるため)。

操作方法:
    Space      : 上昇速度を上げる(離しても目標速度は保持される)
    X          : 目標速度を下げる(0で停止、負で下降)
    Q / E      : 左旋回 / 右旋回(ヨー、離すと機首方向を保持)
    ↑ / ↓      : 左傾 / 右傾(ロール、離すと水平に戻る)
    ← / →      : 後傾 / 前傾(ピッチ、離すと水平に戻る)
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
PROP_ARM_LENGTH = 0.028  # CF2X URDFのプロペラ腕長[m]
PROP_KF = 3.16e-10  # CF2X URDFの推力係数[N/RPM^2]


# ---------------------------------------------------------------------- #
# 姿勢PID(quadcopter_controller.py の PIDController をそのまま使用)
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
    MAX_TILT_DEG = 20.0          # ロール/ピッチの最大傾斜角
    YAW_RATE_DEG_S = 90.0        # ヨースティック最大時の回転速度
    VERTICAL_SPEED_RATE = 1.0  # m/s^2
    MAX_VERTICAL_SPEED = 1.5  # m/s
    VERTICAL_SPEED_KP = 8.0
    MAX_VERTICAL_ACCEL = 6.0  # m/s^2
    RPM_PER_VERTICAL_ACCEL = 740.0  # official/interactive_drone.py と同じ

    def __init__(self, drone):
        self.drone = drone
        self._pid_roll = PIDController(10.0, 0.0, 5.0)
        self._pid_pitch = PIDController(10.0, 0.0, 5.0)
        self._pid_yaw = PIDController(2.0, 0.0, 0.2)
        self._prev_attitude = None

        # スティック入力(-1.0 ~ 1.0)。キーのHOLD/RELEASEで更新される
        self.roll_stick = 0.0
        self.pitch_stick = 0.0
        self.yaw_stick = 0.0
        self.throttle_stick = 0.0

        self.target_yaw_deg = 0.0      # ヨーは絶対角度を積分で保持(レート制御の結果)
        self.target_vertical_speed = 0.0
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
        arm = not self.armed
        if not arm:
            self.armed = False
        self.roll_stick = 0.0
        self.pitch_stick = 0.0
        self.yaw_stick = 0.0
        self.throttle_stick = 0.0
        self.target_vertical_speed = 0.0
        self._pid_roll.reset()
        self._pid_pitch.reset()
        self._pid_yaw.reset()
        if arm:
            # スロットルはホバリング相当の基準RPMから開始
            att = quat_to_xyz(self.drone.get_quat(), rpy=True, degrees=True)
            self.target_yaw_deg = float(att[2])  # 現在の機首方向を基準にする
            self._prev_attitude = np.array([float(angle) for angle in att], dtype=np.float64)
            self.armed = True
        else:
            self._prev_attitude = None
        print(f"=== {'アーム' if arm else 'ディスアーム'} ===")

    # ---- 毎ステップ呼び出し ---- #
    def update(self, dt):
        if not self.armed:
            return [0.0, 0.0, 0.0, 0.0]

        att = quat_to_xyz(self.drone.get_quat(), rpy=True, degrees=True)
        attitude = np.array([float(angle) for angle in att], dtype=np.float64)
        previous_attitude = self._prev_attitude
        if previous_attitude is None:
            attitude_rate = np.zeros(3, dtype=np.float64)
        else:
            angle_delta = attitude - previous_attitude
            angle_delta[2] = (angle_delta[2] + 180.0) % 360.0 - 180.0
            attitude_rate = angle_delta / dt
        self._prev_attitude = attitude

        # ヨー: スティックは回転速度の指示。離せば機首方向を保持する
        self.target_yaw_deg += self.yaw_stick * self.YAW_RATE_DEG_S * dt

        # Space/Xで目標上下速度を変え、目標0では実速度を減衰させる
        self.target_vertical_speed += self.throttle_stick * self.VERTICAL_SPEED_RATE * dt
        self.target_vertical_speed = max(
            -self.MAX_VERTICAL_SPEED,
            min(self.target_vertical_speed, self.MAX_VERTICAL_SPEED),
        )

        # ロール/ピッチ: スティック位置がそのまま目標傾斜角(セルフレベル)
        target_roll = self.roll_stick * self.MAX_TILT_DEG
        target_pitch = self.pitch_stick * self.MAX_TILT_DEG

        err_roll = target_roll - float(att[0])
        err_pitch = target_pitch - float(att[1])
        err_yaw = self.target_yaw_deg - float(att[2])

        roll_del = self._pid_roll.update(err_roll, dt, measurement_rate=attitude_rate[0])
        pitch_del = self._pid_pitch.update(err_pitch, dt, measurement_rate=attitude_rate[1])
        yaw_del = self._pid_yaw.update(err_yaw, dt, measurement_rate=attitude_rate[2])

        vertical_velocity = float(self.drone.get_vel()[2])
        vertical_accel = self.VERTICAL_SPEED_KP * (self.target_vertical_speed - vertical_velocity)
        vertical_accel = max(-self.MAX_VERTICAL_ACCEL, min(vertical_accel, self.MAX_VERTICAL_ACCEL))
        thrust = self.RPM_PER_VERTICAL_ACCEL * vertical_accel
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
        rigid_options=gs.options.RigidOptions(dt=DT / 2),
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
    disarmed_pos = drone.get_pos().clone()
    disarmed_quat = drone.get_quat().clone()

    controller = ManualFlightController(drone)
    wind = SingleDroneWind(seed=args.seed) if args.wind else None

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    def toggle_arm():
        drone.set_pos(disarmed_pos, zero_velocity=True)
        drone.set_quat(disarmed_quat, zero_velocity=True)
        controller.toggle_arm()

    def axis_keybinds(name, key_pos, key_neg, setter):
        """押している間+1/-1、離すと0に戻る軸(ロール/ピッチ/ヨー用)"""
        return [
            Keybind(f"{name}_pos_hold", key_pos, KeyAction.HOLD, callback=setter, args=(1.0,)),
            Keybind(f"{name}_pos_release", key_pos, KeyAction.RELEASE, callback=setter, args=(0.0,)),
            Keybind(f"{name}_neg_hold", key_neg, KeyAction.HOLD, callback=setter, args=(-1.0,)),
            Keybind(f"{name}_neg_release", key_neg, KeyAction.RELEASE, callback=setter, args=(0.0,)),
        ]

    scene.viewer.register_keybinds(
        *axis_keybinds("pitch", Key.RIGHT, Key.LEFT, controller.set_pitch),
        *axis_keybinds("roll", Key.DOWN, Key.UP, controller.set_roll),
        *axis_keybinds("yaw", Key.E, Key.Q, controller.set_yaw),
        *axis_keybinds("throttle", Key.SPACE, Key.X, controller.set_throttle),
        Keybind("arm_toggle", Key.RETURN, KeyAction.RELEASE, callback=toggle_arm),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )

    print("\n=== 操作方法(完全手動。実機のモード2送信機に準拠) ===")
    print("Space      : 上昇速度を上げる(離しても目標速度は保持)")
    print("X          : 目標速度を下げる(0で停止、負で下降)")
    print("Q / E       : 左旋回 / 右旋回(ヨー)")
    print("↑ / ↓       : 左傾 / 右傾(ロール)")
    print("← / →       : 後傾 / 前傾(ピッチ)")
    print("Enter       : アーム / ディスアーム")
    print("Esc         : 終了")
    print("\n起動直後はディスアーム状態です。Enterキーでアームしてください。")
    if wind is not None:
        print(f"風: 有効(ベース強さ {np.linalg.norm(wind.mean):.3f} N、AI/PIDモードと同じモデル)")
    else:
        print("風: 無効(--wind で有効化)")

    try:
        while is_running:
            if not controller.armed:
                drone.set_pos(disarmed_pos, zero_velocity=True)
                drone.set_quat(disarmed_quat, zero_velocity=True)

            rpms = controller.update(DT)
            rpms = [max(MIN_RPM, min(r, MAX_RPM)) for r in rpms]
            drone.set_propellers_rpm(rpms)

            # CF2Xのゼロ質量プロペラリンクでは差動推力の腕長トルクが伝わらないため補う
            prop_forces = [PROP_KF * rpm**2 for rpm in rpms]
            roll_torque = PROP_ARM_LENGTH * (prop_forces[2] + prop_forces[3] - prop_forces[0] - prop_forces[1])
            pitch_torque = PROP_ARM_LENGTH * (prop_forces[1] + prop_forces[2] - prop_forces[0] - prop_forces[3])
            drone.solver.apply_links_external_wrench(
                torque=torch.tensor([[roll_torque, pitch_torque, 0.0]], dtype=torch.float32),
                links_idx=[drone.get_link("base_link").idx],
                ref="link_origin",
                local=True,
            )

            if wind is not None and controller.armed:
                w = wind.step(DT)
                drone.solver.apply_links_external_wrench(
                    force=torch.tensor([list(w)], dtype=torch.float32),
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