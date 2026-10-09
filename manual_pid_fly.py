"""
manual_pid_fly.py

カスケードPID制御(位置→速度→姿勢→ミキサー)で、人間がキーボードから
ドローンを操縦できるようにしたスクリプト。AI(train.py / WindHoverEnv)との
比較用に作成しており、以下を共通化している:

  - 同じCF2X(Crazyflie 2.X)機体・同じホバリング基準RPM
  - 同じOU過程の風外乱モデル(--wind で有効化。wind_hover_env.pyと同一の式・同じ上限クリップ)

ドローン自体を直接操作するのではなく、「目標地点(target)」をキーボードで
動かし、PIDコントローラがそれを追いかける構成にしている(実際の操縦に近い、
「行き先を指示する」操作感)。

操作方法:
  ↑ / ↓ / ← / →  : 目標地点を前後左右に動かす(押している間、一定速度で移動)
  Space            : 目標地点を上昇させる
  LShift           : 目標地点を下降させる
  Esc              : 終了

ベースにした公式コード(PIDゲイン・ミキサー式はそのまま踏襲):
  https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/quadcopter_controller.py
  https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/interactive_drone.py (キー操作の枠組み)
  https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/fly_route.py (PIDゲイン・RPMクランプ範囲)
"""

import argparse
import math

import numpy as np
import torch

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind
from genesis.utils.geom import quat_to_xyz

DT = 0.01  # train.py / wind_hover_env.py と同じ制御周期(100Hz)
BASE_RPM = 14468.429183500699  # CF2Xのホバリング基準RPM(公式サンプルと同一)
MIN_RPM = 0.9 * BASE_RPM  # 公式 fly_route.py のクランプ範囲をそのまま使用
MAX_RPM = 1.5 * BASE_RPM


# ---------------------------------------------------------------------- #
# quadcopter_controller.py を踏襲したカスケードPID(公式実装をそのまま移植)
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


class DronePIDController:
    """位置→速度→姿勢のカスケードPID + ミキサー。公式quadcopter_controller.pyと同一設計。"""

    def __init__(self, drone, dt, base_rpm, pid_params):
        self._pid_pos_x = PIDController(*pid_params[0])
        self._pid_pos_y = PIDController(*pid_params[1])
        self._pid_pos_z = PIDController(*pid_params[2])
        self._pid_vel_x = PIDController(*pid_params[3])
        self._pid_vel_y = PIDController(*pid_params[4])
        self._pid_vel_z = PIDController(*pid_params[5])
        self._pid_att_roll = PIDController(*pid_params[6])
        self._pid_att_pitch = PIDController(*pid_params[7])
        self._pid_att_yaw = PIDController(*pid_params[8])
        self.drone = drone
        self._dt = dt
        self._base_rpm = base_rpm

    def _mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel):
        m1 = self._base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        m2 = self._base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        m3 = self._base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        m4 = self._base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        return [m1, m2, m3, m4]

    def update(self, target):
        pos = self.drone.get_pos()
        vel = self.drone.get_vel()
        att = quat_to_xyz(self.drone.get_quat(), rpy=True, degrees=True)

        err_x = target[0] - float(pos[0])
        err_y = target[1] - float(pos[1])
        err_z = target[2] - float(pos[2])

        vel_des_x = self._pid_pos_x.update(err_x, self._dt)
        vel_des_y = self._pid_pos_y.update(err_y, self._dt)
        vel_des_z = self._pid_pos_z.update(err_z, self._dt)

        err_vel_x = vel_des_x - float(vel[0])
        err_vel_y = vel_des_y - float(vel[1])
        err_vel_z = vel_des_z - float(vel[2])

        x_vel_del = self._pid_vel_x.update(err_vel_x, self._dt)
        y_vel_del = self._pid_vel_y.update(err_vel_y, self._dt)
        thrust_des = self._pid_vel_z.update(err_vel_z, self._dt)

        err_roll = 0.0 - float(att[0])
        err_pitch = 0.0 - float(att[1])
        err_yaw = 0.0 - float(att[2])

        roll_del = self._pid_att_roll.update(err_roll, self._dt)
        pitch_del = self._pid_att_pitch.update(err_pitch, self._dt)
        yaw_del = self._pid_att_yaw.update(err_yaw, self._dt)

        return self._mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)


# ---------------------------------------------------------------------- #
# 風(OU過程)。wind_hover_env.py と同じ式・同じ上限クリップを単一機体向けに移植
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
# 目標地点(target)をキーボードで動かすコントローラ
# ---------------------------------------------------------------------- #
class TargetController:
    """ドローン自体ではなく目標地点の方を動かし、PIDが追従する形にすることで、
    現実の操縦(行き先を指示する)に近い操作感にする。"""

    MOVE_SPEED = 0.6  # m/s。キーを押しっぱなしの間、targetがこの速度で移動する

    def __init__(self, initial_target):
        self.target = np.array(initial_target, dtype=np.float64)
        self.cur_dir = np.zeros(3)

    def add_direction(self, direction):
        self.cur_dir += np.array(direction)

    def step(self, dt):
        self.target += np.clip(self.cur_dir, -1.0, 1.0) * self.MOVE_SPEED * dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wind", action="store_true", help="AIと同じOU過程の風外乱を有効にする")
    parser.add_argument("--seed", type=int, default=None, help="風のパラメータのシード")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 1.5),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=45,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 1.0)),
    )
    scene.viewer.follow_entity(drone)

    # 公式 fly_route.py の値をそのまま使用
    # ("parameters are tuned such that the drone can fly, not optimized" との注記あり)
    pid_params = [
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [20.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [25.0, 0.0, 20.0],
        [10.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
        [2.0, 0.0, 0.2],
    ]

    scene.build()

    com_link_idx = drone.get_link("base_link").idx  # wind_hover_env.pyと同じ回避策

    controller = DronePIDController(drone=drone, dt=DT, base_rpm=BASE_RPM, pid_params=pid_params)
    target_ctrl = TargetController(initial_target=(0.0, 0.0, 1.0))
    wind = SingleDroneWind(seed=args.seed) if args.wind else None

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    def direction_keybinds(name, key, direction):
        return [
            Keybind(f"{name}_hold", key, KeyAction.HOLD, callback=target_ctrl.add_direction, args=(direction,)),
            Keybind(
                f"{name}_release", key, KeyAction.RELEASE,
                callback=target_ctrl.add_direction, args=(tuple(-d for d in direction),),
            ),
        ]

    scene.viewer.register_keybinds(
        *direction_keybinds("move_forward", Key.UP, (1.0, 0.0, 0.0)),
        *direction_keybinds("move_backward", Key.DOWN, (-1.0, 0.0, 0.0)),
        *direction_keybinds("move_left", Key.LEFT, (0.0, 1.0, 0.0)),
        *direction_keybinds("move_right", Key.RIGHT, (0.0, -1.0, 0.0)),
        *direction_keybinds("move_up", Key.SPACE, (0.0, 0.0, 1.0)),
        *direction_keybinds("move_down", Key.LSHIFT, (0.0, 0.0, -1.0)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )

    print("\n=== 操作方法 ===")
    print("↑ / ↓ / ← / →  : 目標地点を前後左右に動かす")
    print("Space            : 目標地点を上昇")
    print("LShift           : 目標地点を下降")
    print("Esc              : 終了")
    if wind is not None:
        print(f"風: 有効(ベース強さ {np.linalg.norm(wind.mean):.3f} N、AIの学習時と同じモデル)")
    else:
        print("風: 無効(--wind で有効化)")

    try:
        while is_running:
            target_ctrl.step(DT)
            rpms = controller.update(target_ctrl.target)
            rpms = [max(MIN_RPM, min(r, MAX_RPM)) for r in rpms]
            drone.set_propellers_rpm(rpms)

            if wind is not None:
                w = wind.step(DT)
                drone.solver.apply_links_external_force(
                    torch.tensor([list(w)], dtype=torch.float32),  # (n_links_idx, 3) = (1, 3)
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