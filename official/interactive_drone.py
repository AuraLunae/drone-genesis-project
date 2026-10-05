import os
import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

# ---------------- 調整用パラメータ ----------------
HOVER_RPM = 14475.8     # ホバリング用の基準RPM
MAX_RPM = 25000.0

V_MAX = 1.5             # 水平の目標速度 [m/s]
Z_RATE = 0.6            # 上昇/下降の速さ [m/s]
KV = 2.5                # 速度誤差 -> 加速度
A_MAX = 2.5             # 水平加速度の上限 [m/s^2]
G = 9.81

KP_ATT = 225.0          # 姿勢 P
KD_ATT = 21.0           # 姿勢 D
KP_YAW = 8.0            # ヨーレート P
YAW_RATE_MAX = 1.5      # [rad/s]
KP_Z = 6.0
KD_Z = 4.0

RPM_PER_ANG_ACC = 14.0  # 角加速度 -> RPM差
RPM_PER_YAW_ACC = 24.0
RPM_PER_ACC_Z = 740.0   # 上下加速度 -> RPM差
MAX_ATT_DELTA = 1500.0
YAW_SIGN = -1.0         # CF2X のプロペラ回転方向とトルクの符号が逆なので反転

# cf2x 用のプロペラ配置固定値 (+x:前, +y:左)
FALLBACK_PROP_XY = np.array([[0.028, -0.028], [-0.028, -0.028], [0.028, 0.028], [-0.028, 0.028]])
FALLBACK_SPIN = np.array([1.0, -1.0, -1.0, 1.0])


def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).reshape(-1)


def quat_to_rot(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class DroneController:
    def __init__(self, prop_xy, spin, z0):
        # 機体座標系: +x=前, +y=左
        self.sx = np.sign(prop_xy[:, 1])  # +y側 (左側) のプロペラ: +1
        self.sy = np.sign(prop_xy[:, 0])  # +x側 (前側) のプロペラ: +1
        self.spin = spin
        self.z_target = z0
        self.keys = {k: False for k in ("up", "down", "left", "right", "climb", "descend", "yaw_l", "yaw_r")}

    def set_key(self, name: str, pressed: bool):
        if self.keys[name] != pressed:
            print(f"[key] {name}: {'ON' if pressed else 'OFF'}")
        self.keys[name] = pressed

    def compute_rpms(self, pos, quat, vel, ang, dt):
        k = self.keys
        R = quat_to_rot(quat)

        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        w_body = R.T @ ang

        # --- 1. 高度制御 ---
        min_z_target = max(0.03, pos[2] - 0.1)
        self.z_target += (float(k["climb"]) - float(k["descend"])) * Z_RATE * dt
        self.z_target = max(self.z_target, min_z_target)

        if k["climb"] and self.z_target < pos[2]:
            self.z_target = pos[2] + 0.05

        acc_z = np.clip(KP_Z * (self.z_target - pos[2]) - KD_Z * vel[2], -6.0, 6.0)
        base = HOVER_RPM + RPM_PER_ACC_Z * acc_z

        tilt_factor = np.sqrt(max(R[2, 2], 0.3))
        base /= tilt_factor

        # --- 2. 水平速度 & 姿勢目標の計算 ---
        # ワールド目標速度 (+x_w: 前, +y_w: 左)
        v_w_x = (float(k["up"]) - float(k["down"])) * V_MAX
        v_w_y = (float(k["left"]) - float(k["right"])) * V_MAX

        # ワールド目標加速度
        a_w_x = np.clip(KV * (v_w_x - vel[0]), -A_MAX, A_MAX)
        a_w_y = np.clip(KV * (v_w_y - vel[1]), -A_MAX, A_MAX)

        # ワールド加速度 -> 機体座標系加速度 (+x_b: 前, +y_b: 左)
        c, s = np.cos(yaw), np.sin(yaw)
        a_b_x = a_w_x * c + a_w_y * s   # 機体前方向の加速度
        a_b_y = -a_w_x * s + a_w_y * c  # 機体左方向の加速度

        if pos[2] < 0.08:
            pitch_t = 0.0
            roll_t = 0.0
        else:
            # 入力方向に対して機体を傾け、そこへ加速させる
            # +x: 前, +y: 左 のボディ座標で、押した方向の加速度と同じ符号で目標姿勢を作る
            pitch_t = np.clip(a_b_x / G, -0.3, 0.3)
            roll_t = np.clip(a_b_y / G, -0.3, 0.3)

        # --- 3. 姿勢 PD 制御 ---
        alpha_x = KP_ATT * (roll_t - roll) - KD_ATT * w_body[0]
        alpha_y = KP_ATT * (pitch_t - pitch) - KD_ATT * w_body[1]

        yaw_rate_t = (float(k["yaw_l"]) - float(k["yaw_r"])) * YAW_RATE_MAX
        alpha_z = KP_YAW * (yaw_rate_t - w_body[2])

        ux = np.clip(alpha_x * RPM_PER_ANG_ACC, -MAX_ATT_DELTA, MAX_ATT_DELTA)
        uy = np.clip(alpha_y * RPM_PER_ANG_ACC, -MAX_ATT_DELTA, MAX_ATT_DELTA)
        uz = np.clip(alpha_z * RPM_PER_YAW_ACC, -MAX_ATT_DELTA, MAX_ATT_DELTA)

        rpms = base + self.sx * ux + self.sy * uy + self.spin * YAW_SIGN * uz
        return np.clip(rpms, 0.0, MAX_RPM)


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-2.0, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    start_z = 0.5
    drone = scene.add_entity(
        morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, start_z)),
    )
    scene.viewer.follow_entity(drone)

    scene.build()

    controller = DroneController(FALLBACK_PROP_XY, FALLBACK_SPIN, start_z)

    def key_binds(name: str, *key_names: str):
        binds = []
        for kn in key_names:
            key = getattr(Key, kn, None)
            if key is None:
                continue
            binds += [
                Keybind(f"{name}_{kn}_press", key, KeyAction.PRESS, callback=controller.set_key, args=(name, True)),
                Keybind(f"{name}_{kn}_release", key, KeyAction.RELEASE, callback=controller.set_key, args=(name, False)),
            ]
        return binds

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    # 矢印キーおよび WASD キー両方に対応
    scene.viewer.register_keybinds(
        *key_binds("up", "UP", "W"),
        *key_binds("down", "DOWN", "S"),
        *key_binds("left", "LEFT", "A"),
        *key_binds("right", "RIGHT", "D"),
        *key_binds("climb", "SPACE", "PAGEUP"),
        *key_binds("descend", "LSHIFT", "RSHIFT", "PAGEDOWN"),
        *key_binds("yaw_l", "Q"),
        *key_binds("yaw_r", "E"),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )

    print("\nDrone Controls:")
    print("W / S / Up / Down      - 前進 / 後退")
    print("A / D / Left / Right   - 左移動 / 右移動")
    print("Space / Shift          - 上昇 / 下降")
    print("Q / E                  - 左旋回 / 右旋回")
    print("Esc                    - 終了")

    dt = 0.01
    try:
        while is_running:
            rpms = controller.compute_rpms(
                to_np(drone.get_pos()),
                to_np(drone.get_quat()),
                to_np(drone.get_vel()),
                to_np(drone.get_ang()),
                dt,
            )
            drone.set_propellers_rpm(rpms)
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()