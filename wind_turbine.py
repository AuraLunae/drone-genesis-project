"""Visualize the simulated wind with a non-colliding wind turbine.

風の強さと向きを映す風車(衝突判定なし)。旗(wind_flag.py)の代わりに、compare_arena.py の後方に立つ。

  * 羽根の回る速さ = 風の強さ(約0.13 N以上で最大の2.6回転/秒。無風では止まる。加速・減速には慣性がある)
  * 風車の向き     = 風向き。羽根が風上を向き、尾翼が風下を向く(風見鶏と同じ。向きは少し遅れて追従する)

部品(assets/wind_turbine/ の tower.stl / head.stl / rotor.stl)は、prepare_wind_turbine_assets.py で作る。
風車は表示専用で、ドローンの物理・採点には影響しない。
"""

import json
import math
import os

import genesis as gs
import numpy as np
import torch

FULL_SCALE_N = 0.13  # この強さ以上で、羽根が最大の速さで回る(旗と同じ基準)
CALM_N = 0.003  # これ未満は無風とみなす
MAX_SPIN_REV_S = 2.6  # 羽根の最大の回転 [回転/秒](初版は1.6。画面の更新が60回/秒なら、1コマで約16度回る)
MAX_SPIN_RAD_S = 2.0 * math.pi * MAX_SPIN_REV_S
SPIN_UP_S = 1.0  # 羽根が加速する時間の目安 [s]
SPIN_DOWN_S = 1.8  # 羽根が減速する時間の目安 [s](回り続ける慣性)
YAW_TAU_S = 0.7  # 風車が風上へ向きを変える時間の目安 [s]
YAW_MAX_RAD_S = 1.6  # 風車が向きを変える最大の速さ [rad/s]
HEIGHT_PER_SCALE = 2.2 / 1.6  # 旗の scale(compare_arena の1.6)から風車の高さ[m]を決める比(scale 1.6 → 2.2 m、eval.py の1.2 → 1.65 m)
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "wind_turbine")
BODY_COLOR = (0.93, 0.93, 0.95)
ROTOR_COLOR = (0.85, 0.22, 0.18)  # 回っているのが見えやすい色(レーンの色=青・緑・橙とは別)


def missing_assets(asset_dir=None):
    """風車の部品のうち、見つからないファイル名のリスト(全部あれば空)。"""
    asset_dir = asset_dir or ASSET_DIR
    needed = ("tower.stl", "head.stl", "rotor.stl", "layout.json")
    return [name for name in needed if not os.path.exists(os.path.join(asset_dir, name))]


def legend():
    """風車の見方を簡潔に説明する。"""
    return (
        "風車 = 風の見える化(衝突判定なし): "
        f"羽根の回る速さ=風の強さ(約{FULL_SCALE_N:.2f} N以上で最大・無風で停止) / "
        "羽根が向く方=風上(尾翼が向く方=風下)"
    )


def quat_mul(a, b):
    """wxyz順のクォータニオンの積 a*b。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_z(angle):
    return np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])


def quat_x(angle):
    return np.array([math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0])


def wrap_pi(angle):
    """角度を -pi〜pi に直す(一番近い回り方で追従させるため)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def target_yaw(wind):
    """羽根を風上へ向ける向き(頭の +x が向く方位 [rad])。風は力の向き=風下なので、その反対。"""
    return math.atan2(-wind[1], -wind[0])


class NullWindTurbine:
    """部品が使えない時の代わり(何も描かず、風の強さだけ覚える)。比較そのものは止めない。"""

    style = "none"

    def __init__(self):
        self.strength = 0.0

    def after_build(self):
        pass

    def update(self, wind_vec, dt):
        self.strength = float(np.linalg.norm(np.asarray(wind_vec, dtype=float).reshape(3)))


class WindTurbine:
    """支柱(固定) + 頭(風向きに合わせて回る) + 羽根(風の強さで回る)。

    頭と羽根は fixed=False(重力を受ける)・collision=False の物体で、旗の板と同じく、
    **毎ステップ** set_pos / set_quat で位置と向きを入れ直す(落ちないように)。"""

    style = "turbine"

    def __init__(self, scene, base_xy, height=2.2, asset_dir=None, n_envs=0):
        """n_envs: 環境の数(0なら、環境を並べない普通のシーン)。1以上なら、全環境に同じ風車を置く。"""
        asset_dir = asset_dir or ASSET_DIR
        self.n_envs = int(n_envs)
        with open(os.path.join(asset_dir, "layout.json"), encoding="utf-8") as f:
            layout = json.load(f)
        self.scale = float(height) / float(layout["height_m"])
        self.base = np.array([float(base_xy[0]), float(base_xy[1]), 0.0])
        self.pivot = np.array([0.0, 0.0, float(layout["pivot_z_m"])]) * self.scale
        self.mount = np.array(layout["rotor_mount_m"], dtype=float) * self.scale
        self.spin_sign = 1.0 if float(layout.get("spin_sign", -1)) > 0 else -1.0
        self.rotor_radius = float(layout["rotor_radius_m"]) * self.scale
        self.hub_height = float(self.pivot[2] + self.mount[2])

        self.yaw = 0.0
        self.spin_angle = 0.0
        self.spin_rate = 0.0  # 羽根の回転(+x軸まわり、向きの符号を含む)[rad/s]
        self.strength = 0.0
        self._failed = False

        body = gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=BODY_COLOR))
        blade = gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=ROTOR_COLOR))

        def add(name, surface, pos, fixed):
            return scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=os.path.join(asset_dir, name),
                    scale=self.scale,
                    pos=tuple(float(v) for v in pos),
                    fixed=fixed,
                    collision=False,
                ),
                surface=surface,
            )

        head_pos, rotor_pos, _ = self._pose()
        self.tower = add("tower.stl", body, self.base, True)
        self.head = add("head.stl", body, head_pos, False)
        self.rotor = add("rotor.stl", blade, rotor_pos, False)

    def after_build(self):
        pass  # 旗(布)と違い、build の後に必要な処理は無い

    def _pose(self):
        """いまの向き・羽根の角度から、頭と羽根の(位置, 向き)を求める。"""
        q_yaw = quat_z(self.yaw)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        mount_world = np.array(
            [c * self.mount[0] - s * self.mount[1], s * self.mount[0] + c * self.mount[1], self.mount[2]]
        )
        head_pos = self.base + self.pivot
        rotor_pos = head_pos + mount_world
        q_rotor = quat_mul(q_yaw, quat_x(self.spin_angle))
        return head_pos, rotor_pos, (q_yaw, q_rotor)

    def update(self, wind_vec, dt):
        """風の力のベクトル [N] を、風車の動きに反映する(毎ステップ、scene.step() の前に呼ぶ)。"""
        wind = np.asarray(wind_vec, dtype=float).reshape(3)
        self.strength = float(np.linalg.norm(wind))
        horizontal = float(np.linalg.norm(wind[:2]))

        # 向き: 風があれば、羽根を風上へ。一番近い回り方で、遅れて追いかける
        if self.strength > CALM_N and horizontal > 1e-9:
            diff = wrap_pi(target_yaw(wind) - self.yaw)
            step = diff * min(1.0, dt / YAW_TAU_S)
            limit = YAW_MAX_RAD_S * dt
            self.yaw = wrap_pi(self.yaw + max(-limit, min(limit, step)))

        # 羽根: 風の強さに比例した速さへ。加速より減速をゆっくりにする(慣性)
        level = min(self.strength / FULL_SCALE_N, 1.0) if self.strength > CALM_N else 0.0
        target_rate = self.spin_sign * MAX_SPIN_RAD_S * level
        tau = SPIN_UP_S if abs(target_rate) > abs(self.spin_rate) else SPIN_DOWN_S
        self.spin_rate += (target_rate - self.spin_rate) * min(1.0, dt / tau)
        self.spin_angle = (self.spin_angle + self.spin_rate * dt) % (2.0 * math.pi)

        self._apply()

    def _apply(self):
        if self._failed:
            return
        head_pos, rotor_pos, (q_yaw, q_rotor) = self._pose()

        def tensor(a):
            a = np.asarray(a, dtype=float)
            if self.n_envs > 0:
                a = np.tile(a, (self.n_envs, 1))  # 環境を並べたシーンの set_pos / set_quat は (環境数, 3または4)
            return torch.as_tensor(a, dtype=torch.float32, device=gs.device)

        try:
            self.head.set_pos(tensor(head_pos), zero_velocity=True)
            self.head.set_quat(tensor(q_yaw), zero_velocity=True)
            self.rotor.set_pos(tensor(rotor_pos), zero_velocity=True)
            self.rotor.set_quat(tensor(q_rotor), zero_velocity=True)
        except Exception as error:  # noqa: BLE001  版差で動かせない場合は、風車だけ止めて続行する
            self._failed = True
            print(f"[警告] 風車の位置・向きを更新できなくなりました(風車は止まります): {type(error).__name__}: {error}")


def create_wind_turbine(scene, base_xy, height=2.2, asset_dir=None, n_envs=0):
    """風車を作る。部品のファイルが無い・作れない時は、警告を出して何も描かない風車を返す(比較は止めない)。

    ※ wind_flag.create_wind_flag(style="turbine"、既定)は、部品が無い時に何も描かない風車ではなく、板の旗に切り替える。"""
    asset_dir = asset_dir or ASSET_DIR
    missing = missing_assets(asset_dir)
    if missing:
        print(
            f"[警告] 風車の部品がありません({', '.join(missing)} / 場所: {asset_dir})。風車なしで続行します。"
            " 先に `uv run python prepare_wind_turbine_assets.py WindTurbineWithBallBearingsV2_stls.zip` を実行してください"
        )
        return NullWindTurbine()
    try:
        return WindTurbine(scene, base_xy, height=height, asset_dir=asset_dir, n_envs=n_envs)
    except Exception as error:  # noqa: BLE001
        print(f"[警告] 風車を作成できません({type(error).__name__}: {error})。風車なしで続行します")
        return NullWindTurbine()