"""Visualize the simulated wind with a non-colliding flag."""

import math
import os
import tempfile

import genesis as gs
import numpy as np
import torch

G = 9.81
FULL_SCALE_N = 0.13
SMOOTHING_S = 0.15
CALM_N = 0.003
ALL_STYLES = ("procedural", "cloth", "cloth-mesh", "plate")
DEFAULT_STYLE = "procedural"
_CLOTH_VIS_MODE = {"cloth": "particle", "cloth-mesh": "visual"}


def legend(style=DEFAULT_STYLE):
    """旗の見方を簡潔に説明する。"""
    if style in _CLOTH_VIS_MODE:
        kind = "PBDソルバーの布"
    elif style == "procedural":
        kind = "進行波で動く分割旗"
    else:
        kind = "剛体の板"
    return (
        f"白い旗 = 風の見える化({kind}・衝突判定なし): "
        f"垂れ下がる=ほぼ無風 / 持ち上がるほど強風 / "
        f"旗が向く方向=風下(約{FULL_SCALE_N:.2f} N以上でほぼ水平)"
    )


def _normalize(vec, default=(1.0, 0.0)):
    """ゼロ除算を避ける安全な正規化。"""
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.asarray(default, dtype=float)
    return arr / norm


def _wind_to_3d(wind_vec):
    """風入力を 3 要素 ndarray に変換し、torch.Tensor も安全に扱う。"""
    if isinstance(wind_vec, torch.Tensor):
        arr = wind_vec.detach().reshape(-1).cpu().numpy()
    else:
        arr = np.asarray(wind_vec, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"wind_vec must have 3 elements, got shape {arr.shape}")
    return arr.astype(float)


def matrix_to_quat(rotation):
    """Convert a 3x3 rotation matrix to Genesis wxyz quaternion order."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    quat = np.array([w, x, y, z], dtype=np.float64)
    norm_q = np.linalg.norm(quat)
    if not np.isfinite(norm_q) or norm_q < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm_q


def flag_pose(azimuth, level, hinge, flag_length, flutter_rad=0.0):
    """Return the center and orientation of the rigid flag."""
    alpha = min(max(level * (math.pi / 2.0) + flutter_rad, 0.0), math.pi / 2.0)
    direction = np.array(
        [math.sin(alpha) * azimuth[0], math.sin(alpha) * azimuth[1], -math.cos(alpha)]
    )
    width_axis = np.array([-azimuth[1], azimuth[0], 0.0])
    normal = np.cross(width_axis, direction)
    rotation = np.column_stack([direction, normal, width_axis])
    center = np.asarray(hinge, dtype=float) + direction * (flag_length / 2.0)
    return center, matrix_to_quat(rotation)


def yaw_pitch_to_quat(yaw, pitch_up):
    """Quaternion (wxyz) that points a box's local +X along the (yaw, pitch_up) direction.

    pitch_up は「上向きが正」(水平=0、真下=-pi/2)。Genesis/標準の右手系では
    Y軸まわりの正回転は +X を下向きに倒すため、Y軸まわりの角度は -pitch_up を使う。
    これで `direction_from_yaw_pitch(yaw, pitch_up)` の向きと一致する。
    """
    pitch_q = -pitch_up
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch_q * 0.5), math.sin(pitch_q * 0.5)
    quat = np.array([cp * cy, -sp * sy, sp * cy, cp * sy], dtype=np.float64)
    norm_q = float(np.linalg.norm(quat))
    if not np.isfinite(norm_q) or norm_q < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm_q


def direction_from_yaw_pitch(yaw, pitch_up):
    """Unit vector for (yaw, pitch_up); pitch_up is positive upward."""
    return np.array(
        [
            math.cos(pitch_up) * math.cos(yaw),
            math.cos(pitch_up) * math.sin(yaw),
            math.sin(pitch_up),
        ],
        dtype=np.float64,
    )


def _add_pole(scene, base_xy, pole_height, scale):
    gray = gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.72, 0.72, 0.72)))
    return scene.add_entity(
        morph=gs.morphs.Box(
            pos=(base_xy[0], base_xy[1], pole_height / 2.0),
            size=(0.04 * scale, 0.04 * scale, pole_height),
            fixed=True,
            collision=False,
        ),
        surface=gray,
    )


class ProceduralWindFlag:
    """進行波(Traveling Wave)で動く超軽量・高リアルなプロシージャル旗"""

    FULL_LEVEL_PITCH_DEG = 0.0  # 強風(level=1)で旗はほぼ水平(Plate と同じ基準)
    HANG_PITCH_DEG = -90.0  # 無風(level=0)で真下に垂れ下がる

    def __init__(self, scene, base_xy, scale=1.0, n_envs=0, n_segments=6):
        self.n_envs = n_envs
        self.scale = scale
        self.pole_height = 2.6 * scale
        self.total_length = 0.85 * scale
        self.flag_height = 0.45 * scale
        self.n_segments = n_segments
        self.seg_length = self.total_length / max(1, n_segments)
        self.hinge = np.array([base_xy[0], base_xy[1], self.pole_height], dtype=float)

        self._azimuth = np.array([1.0, 0.0], dtype=float)
        self._level = 0.0
        self._time = 0.0
        self.strength = 0.0
        self.style = "procedural"

        _add_pole(scene, base_xy, self.pole_height, scale)

        white = gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.96, 0.96, 0.96)))
        self.segments = []
        for _ in range(n_segments):
            seg = scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(base_xy[0], base_xy[1], self.pole_height),
                    size=(self.seg_length * 1.03, 0.006 * scale, self.flag_height),
                    fixed=False,
                    collision=False,
                ),
                surface=white,
            )
            self.segments.append(seg)

    def after_build(self):
        """build 直後に無風の垂れ下がり姿勢を置く(最初の step までの間、全セグメントが
        蝶番の位置に重なって見えるのを避ける)。"""
        try:
            self._apply_pose(self._compute_pose())
        except Exception as error:
            print(f"[警告] 旗の初期姿勢を設定できませんでした: {error}")

    def _compute_pose(self):
        """現在の _level / _azimuth / _time から、各セグメントの (中心, quat) を返す。"""
        base_pitch = math.radians(
            self.HANG_PITCH_DEG
            + (self.FULL_LEVEL_PITCH_DEG - self.HANG_PITCH_DEG) * (self._level ** 0.8)
        )
        yaw_angle = math.atan2(self._azimuth[1], self._azimuth[0])

        wave_freq = 2.5 + 4.5 * self._level
        k = 2.0 * math.pi / (0.45 * self.total_length)
        max_amplitude = math.radians(22.0 * self._level)

        curr_pos = self.hinge.copy()
        poses = []
        for i in range(self.n_segments):
            x_rel = (i + 0.5) * self.seg_length
            amp_factor = (x_rel / self.total_length) ** 1.3
            phase = k * x_rel - 2.0 * math.pi * wave_freq * self._time
            wave_angle = max_amplitude * amp_factor * math.sin(phase)

            seg_pitch = base_pitch + wave_angle * 0.4  # 上向き正
            seg_yaw = yaw_angle + wave_angle * 0.8

            quat = yaw_pitch_to_quat(seg_yaw, seg_pitch)
            dir_vec = direction_from_yaw_pitch(seg_yaw, seg_pitch)
            seg_center = curr_pos + dir_vec * (self.seg_length * 0.5)
            curr_pos = curr_pos + dir_vec * self.seg_length
            poses.append((seg_center, quat))
        return poses

    def _apply_pose(self, poses):
        device = getattr(gs, "device", torch.device("cpu"))
        for seg, (center, quat) in zip(self.segments, poses):
            if self.n_envs > 0:
                center = np.tile(center, (self.n_envs, 1))
                quat = np.tile(quat, (self.n_envs, 1))
            seg.set_pos(torch.as_tensor(center, dtype=torch.float32, device=device), zero_velocity=True)
            seg.set_quat(torch.as_tensor(quat, dtype=torch.float32, device=device), zero_velocity=True)

    def update(self, wind_vec, dt):
        """現在の風ベクトル[N]を受けて旗の形状・姿勢を更新する"""
        wind = _wind_to_3d(wind_vec)
        self.strength = float(np.linalg.norm(wind))
        self._time += float(dt)

        horizontal = wind[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if self.strength > CALM_N and horizontal_norm > 1e-9:
            self._azimuth = _normalize(horizontal, default=(1.0, 0.0))

        target_level = min(self.strength / FULL_SCALE_N, 1.0)
        self._level += (target_level - self._level) * min(1.0, dt / SMOOTHING_S)

        self._apply_pose(self._compute_pose())


class PlateWindFlag:
    """Non-colliding rigid flag, used directly or as a cloth fallback."""

    FLUTTER_RAD = 0.10
    FLUTTER_HZ = 3.0
    style = "plate"

    def __init__(self, scene, base_xy, scale=1.0, n_envs=0):
        self.n_envs = n_envs
        self.pole_height = 2.6 * scale
        self.flag_length = 0.70 * scale
        self.hinge = np.array([base_xy[0], base_xy[1], self.pole_height], dtype=float)
        self._azimuth = np.array([1.0, 0.0])
        self._level = 0.0
        self._time = 0.0
        self.strength = 0.0

        white = gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(1.0, 1.0, 1.0)))
        _add_pole(scene, base_xy, self.pole_height, scale)
        self.flag = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(base_xy[0], base_xy[1], self.pole_height - self.flag_length / 2.0),
                size=(self.flag_length, 0.015 * scale, 0.45 * scale),
                fixed=False,
                collision=False,
            ),
            surface=white,
        )

    def after_build(self):
        """build 直後に垂れ下がり姿勢を置く(初期の quat は単位で、横向きに見えてしまうため)。"""
        try:
            self._apply_pose(0.0)
        except Exception as error:
            print(f"[警告] 板の旗の初期姿勢を設定できませんでした: {error}")

    def _apply_pose(self, flutter):
        center, quat = flag_pose(self._azimuth, self._level, self.hinge, self.flag_length, flutter)
        if self.n_envs > 0:
            center = np.tile(center, (self.n_envs, 1))
            quat = np.tile(quat, (self.n_envs, 1))
        device = getattr(gs, "device", torch.device("cpu"))
        self.flag.set_pos(torch.as_tensor(center, dtype=torch.float32, device=device), zero_velocity=True)
        self.flag.set_quat(torch.as_tensor(quat, dtype=torch.float32, device=device), zero_velocity=True)

    def update(self, wind_vec, dt):
        """Apply the current force vector [N] to the flag visualization."""
        wind = _wind_to_3d(wind_vec)
        self.strength = float(np.linalg.norm(wind))
        self._time += dt

        horizontal = wind[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if self.strength > CALM_N and horizontal_norm > 1e-9:
            self._azimuth = _normalize(horizontal, default=(1.0, 0.0))

        target_level = min(self.strength / FULL_SCALE_N, 1.0)
        self._level += (target_level - self._level) * min(1.0, dt / SMOOTHING_S)
        flutter = self.FLUTTER_RAD * self._level * math.sin(2.0 * math.pi * self.FLUTTER_HZ * self._time)
        self._apply_pose(flutter)


def write_flag_mesh(path, length, width, nx, nz):
    """Write a vertical rectangular OBJ cloth mesh."""
    if nx < 2 or nz < 2:
        raise ValueError("nx and nz must be >= 2 to generate a mesh")
    lines = []
    for iz in range(nz):
        z = -width * iz / (nz - 1)
        for ix in range(nx):
            lines.append(f"v {length * ix / (nx - 1):.6f} 0.000000 {z:.6f}")
    for iz in range(nz - 1):
        for ix in range(nx - 1):
            a = iz * nx + ix + 1
            b, c = a + 1, a + nx
            d = c + 1
            lines.extend((f"f {a} {b} {d}", f"f {a} {d} {c}"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii") as mesh_file:
        mesh_file.write("# wind flag cloth mesh (generated by wind_flag.py)\n" + "\n".join(lines) + "\n")
    return path


def hoist_world_positions(base_xy, pole_height, cell, nz):
    """Return cloth particles along the flag's hoist edge."""
    return [(base_xy[0], base_xy[1], pole_height - j * cell) for j in range(nz)]


class WindBank:
    """Approximate continuous acceleration using on/off force fields in binary weights."""

    def __init__(self, a_max, bits):
        if bits < 1:
            raise ValueError("WindBank.bits must be >= 1")
        self.bits = bits
        self.levels = 2**bits - 1
        self.step = a_max / self.levels

    def quantize(self, value):
        return int(min(round(abs(value) / self.step), self.levels))

    def active_bits(self, value):
        quantized = self.quantize(value)
        return [(quantized >> bit) & 1 for bit in range(self.bits)]


class ClothWindFlag:
    """PBD cloth flag driven by the current wind force."""

    LENGTH = 0.90
    WIDTH = 0.55
    CELL = 0.05
    POLE_HEIGHT = 2.6
    THETA_MAX_DEG = 75.0
    BANK_BITS = 6
    FLUTTER_AMP = 0.15
    FLUTTER_HZ = 3.0

    def __init__(self, scene, base_xy, scale=1.0, n_envs=0, style="cloth"):
        self.style = style
        self.base_xy = base_xy
        self.pole_height = self.POLE_HEIGHT * scale
        length, width, self.cell = self.LENGTH * scale, self.WIDTH * scale, self.CELL * scale
        self.nx = int(round(length / self.cell)) + 1
        self.nz = int(round(width / self.cell)) + 1
        self._azimuth = np.array([1.0, 0.0])
        self._level = 0.0
        self._time = 0.0
        self.strength = 0.0
        self._built = False
        self._fields = {}
        self._active = {}
        self.wind_enabled = False

        mesh_path = os.path.join(
            tempfile.gettempdir(),
            "drone_wind_flag",
            f"flag_{self.nx}x{self.nz}_{length:.3f}x{width:.3f}.obj",
        )
        write_flag_mesh(mesh_path, length, width, self.nx, self.nz)
        self.cloth = scene.add_entity(
            material=gs.materials.PBD.Cloth(),
            morph=gs.morphs.Mesh(file=mesh_path, pos=(base_xy[0], base_xy[1], self.pole_height)),
            surface=gs.surfaces.Default(
                color=(1.0, 1.0, 1.0, 1.0), vis_mode=_CLOTH_VIS_MODE[style]
            ),
        )

        a_max = G * math.tan(math.radians(self.THETA_MAX_DEG))
        self.bank = WindBank(a_max, self.BANK_BITS)
        try:
            for axis, vector in (("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0))):
                for sign in (1, -1):
                    direction = tuple(sign * component for component in vector)
                    for bit in range(self.BANK_BITS):
                        self._fields[(axis, sign, bit)] = scene.add_force_field(
                            gs.force_fields.Constant(
                                direction=direction, strength=self.bank.step * (2**bit)
                            )
                        )
            self.wind_enabled = True
        except Exception as error:
            print(f"[警告] 布の旗に風を反映できません(力場を登録できませんでした): {error}")
            self._fields = {}

        _add_pole(scene, base_xy, self.pole_height, scale)

    def after_build(self):
        """Pin the cloth's hoist edge after scene.build()."""
        positions = hoist_world_positions(self.base_xy, self.pole_height, self.cell, self.nz)
        try:
            for position in positions:
                self.cloth.fix_particles(self.cloth.find_closest_particle(position))
        except Exception as error:
            try:
                self.cloth.fix_particles(particles_idx_local=[j * self.nx for j in range(self.nz)])
            except Exception as fallback_error:
                print(f"[警告] 布を支柱に固定できませんでした(旗が落ちる可能性があります): {error} / {fallback_error}")
        self._built = True

    def _set_axis(self, axis, value):
        bits = self.bank.active_bits(value)
        positive = value >= 0
        for sign, on_side in ((1, positive), (-1, not positive)):
            for bit in range(self.bank.bits):
                key = (axis, sign, bit)
                want = bool(bits[bit]) if on_side else False
                if self._active.get(key, False) != want:
                    field = self._fields[key]
                    field.activate() if want else field.deactivate()
                    self._active[key] = want

    def update(self, wind_vec, dt):
        """Apply the current force vector [N] to the cloth visualization."""
        wind = _wind_to_3d(wind_vec)
        self.strength = float(np.linalg.norm(wind))
        self._time += dt
        horizontal = wind[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if self.strength > CALM_N and horizontal_norm > 1e-9:
            self._azimuth = _normalize(horizontal, default=(1.0, 0.0))

        target_level = min(self.strength / FULL_SCALE_N, 1.0)
        self._level += (target_level - self._level) * min(1.0, dt / SMOOTHING_S)
        if not (self._built and self.wind_enabled):
            return

        theta = math.radians(self.THETA_MAX_DEG) * self._level
        acceleration = G * math.tan(theta)
        acceleration *= 1.0 + self.FLUTTER_AMP * self._level * math.sin(
            2.0 * math.pi * self.FLUTTER_HZ * self._time
        )
        self._set_axis("x", acceleration * self._azimuth[0])
        self._set_axis("y", acceleration * self._azimuth[1])


def wind_flag_scene_kwargs(style=DEFAULT_STYLE, scale=1.0):
    """Return optional Genesis scene settings needed by PBD cloth."""
    if style not in _CLOTH_VIS_MODE:
        return {}
    kwargs = {}
    for options_name in ("LegacyCouplerOptions", "CouplerOptions"):
        try:
            kwargs["coupler_options"] = getattr(gs.options, options_name)(rigid_pbd=False)
            break
        except (AttributeError, TypeError):
            continue
    try:
        kwargs["pbd_options"] = gs.options.PBDOptions(particle_size=ClothWindFlag.CELL * scale)
    except (AttributeError, TypeError):
        pass
    return kwargs


def create_wind_flag(scene, base_xy, scale=1.0, n_envs=0, style=DEFAULT_STYLE):
    """Create the requested flag, falling back to a rigid plate if cloth fails."""
    if style not in ALL_STYLES:
        raise ValueError(f"style must be one of {ALL_STYLES}: {style}")
    if style == "procedural":
        return ProceduralWindFlag(scene, base_xy, scale=scale, n_envs=n_envs)
    if style in _CLOTH_VIS_MODE:
        try:
            return ClothWindFlag(scene, base_xy, scale=scale, n_envs=n_envs, style=style)
        except Exception as error:
            print(f"[警告] PBD布を作成できないため、剛体板の旗に切り替えます: {error}")
    return PlateWindFlag(scene, base_xy, scale=scale, n_envs=n_envs)
