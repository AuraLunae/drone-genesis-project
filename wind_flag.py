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
ALL_STYLES = ("cloth", "cloth-mesh", "plate")
_CLOTH_VIS_MODE = {"cloth": "particle", "cloth-mesh": "visual"}


def legend(style="cloth-mesh"):
    """旗の見方を簡潔に説明する。"""
    cloth = style in _CLOTH_VIS_MODE
    kind = "PBDソルバーの布" if cloth else "剛体の板"
    return (
        f"白い旗 = 風の見える化({kind}・衝突判定なし): "
        f"垂れ下がる=ほぼ無風 / 持ち上がるほど強風 / "
        f"旗が向く方向=風下(約{FULL_SCALE_N:.2f} N以上でほぼ水平)"
    )


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
    return quat / np.linalg.norm(quat)


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
        pass

    def update(self, wind_vec, dt):
        """Apply the current force vector [N] to the flag visualization."""
        wind = np.asarray(wind_vec, dtype=float).reshape(3)
        self.strength = float(np.linalg.norm(wind))
        self._time += dt

        horizontal = wind[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if self.strength > CALM_N and horizontal_norm > 1e-9:
            self._azimuth = horizontal / horizontal_norm

        target_level = min(self.strength / FULL_SCALE_N, 1.0)
        self._level += (target_level - self._level) * min(1.0, dt / SMOOTHING_S)
        flutter = self.FLUTTER_RAD * self._level * math.sin(2.0 * math.pi * self.FLUTTER_HZ * self._time)
        center, quat = flag_pose(self._azimuth, self._level, self.hinge, self.flag_length, flutter)

        if self.n_envs > 0:
            center = np.tile(center, (self.n_envs, 1))
            quat = np.tile(quat, (self.n_envs, 1))
        self.flag.set_pos(torch.as_tensor(center, dtype=torch.float32, device=gs.device), zero_velocity=True)
        self.flag.set_quat(torch.as_tensor(quat, dtype=torch.float32, device=gs.device), zero_velocity=True)


def write_flag_mesh(path, length, width, nx, nz):
    """Write a vertical rectangular OBJ cloth mesh."""
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
        wind = np.asarray(wind_vec, dtype=float).reshape(3)
        self.strength = float(np.linalg.norm(wind))
        self._time += dt
        horizontal = wind[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if self.strength > CALM_N and horizontal_norm > 1e-9:
            self._azimuth = horizontal / horizontal_norm

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


def wind_flag_scene_kwargs(style="cloth-mesh", scale=1.0):
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


def create_wind_flag(scene, base_xy, scale=1.0, n_envs=0, style="cloth-mesh"):
    """Create the requested flag, falling back to a rigid plate if cloth fails."""
    if style not in ALL_STYLES:
        raise ValueError(f"style must be one of {ALL_STYLES}: {style}")
    if style in _CLOTH_VIS_MODE:
        try:
            return ClothWindFlag(scene, base_xy, scale=scale, n_envs=n_envs, style=style)
        except Exception as error:
            print(f"[警告] PBD布を作成できないため、剛体板の旗に切り替えます: {error}")
    return PlateWindFlag(scene, base_xy, scale=scale, n_envs=n_envs)
