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
ALL_STYLES = ("surface", "cloth-mesh", "cloth", "cloth-vel", "plate")
_CLOTH_VIS_MODE = {"surface": "visual", "cloth-mesh": "visual", "cloth": "particle", "cloth-vel": "visual"}
_STYLE_LABEL = {
    "surface": "式で形を計算したスプライン曲面",
    "cloth-mesh": "PBDソルバーの布",
    "cloth": "PBDソルバーの布(粒子表示)",
    "cloth-vel": "PBDソルバーの布(粒子の速度へ風を加える方式)",
    "plate": "剛体の板",
}
VELOCITY_PBD_LOWER = (-8.0, -8.0, 0.0)  # cloth-vel: PBDの粒子を動かす範囲 [m]
VELOCITY_PBD_UPPER = (8.0, 8.0, 6.0)
SWIVEL_S = 0.35  # 旗が風下へ向きを変える時間の目安 [s]
GRAVITY_FEEL = 1.0  # 旗の「ほんの少しの重さ」の強さ。0で重力感なし、2で倍
STRETCH_STATIC = 0.012  # 自重で布がわずかに伸びる量(掛け縁で最大の比率)
STRETCH_DYN = 0.05  # 風に引かれて布が伸びる量(掛け縁で最大の比率。伸び全体はこの約半分)
LIFT_OMEGA = 2.0 * math.pi * 1.3  # 垂れ角の揺り戻しの固有角周波数 [rad/s]
LIFT_ZETA = 0.38  # 同・減衰比(小さいほど跳ね返る)
TENSION_OMEGA = 2.0 * math.pi * 3.0  # 布の伸び縮みの固有角周波数 [rad/s]
TENSION_ZETA = 0.25  # 同・減衰比


def legend(style="surface"):
    """旗の見方を簡潔に説明する。"""
    kind = _STYLE_LABEL.get(style, "旗")
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


def _unit_xy(along):
    """水平方向(長さ2)を単位ベクトルにする。"""
    vector = np.asarray(along, dtype=float).reshape(2)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("along must be a non-zero horizontal direction")
    return vector / norm


def _cloth_surface(vis_mode):
    """Cloth surface with smoothed normals, visible from both sides.

    視点によっては布の裏側を見ることになる(旗の向きは視線に対して決まる)ので、両面表示にする。
    この版に `smooth` や `double_sided` が無ければ、通るものだけで作る。
    """
    options = (
        dict(smooth=True, double_sided=True),
        dict(smooth=True),
        dict(double_sided=True),
        dict(),
    )
    for extra in options:
        try:
            return gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0), vis_mode=vis_mode, **extra)
        except TypeError:
            continue
    return gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0), vis_mode=vis_mode)


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

    def __init__(self, scene, base_xy, scale=1.0, n_envs=0, along=None):
        """along: 布の旗と同じ指定(布が最初に伸びる水平方向)。板は広い面が風下の向きに直交するので、
        布の旗と同じ側から広い面が見えるよう、along を90度回した向きを最初の風下の向きにする。
        None なら従来どおり (1, 0)。"""
        self.n_envs = n_envs
        self.pole_height = 2.6 * scale
        self.flag_length = 0.70 * scale
        self.hinge = np.array([base_xy[0], base_xy[1], self.pole_height], dtype=float)
        if along is None:
            self._azimuth = np.array([1.0, 0.0])
        else:
            unit = _unit_xy(along)
            self._azimuth = np.array([unit[1], -unit[0]])
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


def write_flag_mesh(path, length, width, nx, nz, along=(1.0, 0.0)):
    """Write a vertical rectangular OBJ cloth mesh.

    along: 支柱から旗の先端へ向かう水平の向き。既定の (1, 0) は +x 方向(従来どおり)。
    """
    ax, ay = _unit_xy(along)
    lines = []
    for iz in range(nz):
        z = -width * iz / (nz - 1)
        for ix in range(nx):
            s = length * ix / (nx - 1)
            lines.append(f"v {s * ax:.6f} {s * ay:.6f} {z:.6f}")
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



# ----------------------------------------------------------------------------
# 式で形を計算するスプライン曲面(自由形状変形, FFD)
#   旗の形を「17x9個の代表点」だけ式で計算し、三次スプラインで滑らかな曲面にして
#   全ての粒子の位置に写す。毎フレームの計算は小さな行列積だけで、とても軽い。
#   (FFDの外部ライブラリ PyGeM / geomdl / splinepy と同じ考え方を、numpyだけで実装)
# ----------------------------------------------------------------------------
def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bspline_basis(x, n_ctrl, degree=3):
    """Clamped uniform B-spline basis values at x in [0, 1]; returns (len(x), n_ctrl)."""
    p = min(degree, n_ctrl - 1)
    inner = np.linspace(0.0, 1.0, n_ctrl - p + 1)[1:-1]
    knots = np.concatenate([np.zeros(p + 1), inner, np.ones(p + 1)])
    x = np.clip(np.asarray(x, dtype=float).reshape(-1), 0.0, 1.0)
    basis = np.zeros((x.size, knots.size - 1))
    for i in range(knots.size - 1):
        basis[:, i] = (x >= knots[i]) & (x < knots[i + 1])
    basis[x >= 1.0, n_ctrl - 1] = 1.0
    for d in range(1, p + 1):
        nxt = np.zeros((x.size, knots.size - 1 - d))
        for i in range(knots.size - 1 - d):
            left = knots[i + d] - knots[i]
            right = knots[i + d + 1] - knots[i + 1]
            if left > 0.0:
                nxt[:, i] += (x - knots[i]) / left * basis[:, i]
            if right > 0.0:
                nxt[:, i] += (knots[i + d + 1] - x) / right * basis[:, i + 1]
        basis = nxt
    return basis, knots, p


def spline_interp_matrix(x, n_samples, degree=3):
    """Matrix M (len(x), n_samples) so that M @ samples is the cubic spline through the samples.

    The samples sit at evenly spaced parameters linspace(0, 1, n_samples), and the spline
    passes exactly through them (so the hoist edge stays on the pole).
    """
    basis_x, _, _ = bspline_basis(x, n_samples, degree)
    sample_x = np.linspace(0.0, 1.0, n_samples)
    basis_s, _, _ = bspline_basis(sample_x, n_samples, degree)
    return basis_x @ np.linalg.inv(basis_s), sample_x


def flag_shape_params(level, lift=None):
    """Wind level (0..1) -> shape parameters of the flag.

    lift: wind level with cloth inertia (spring-smoothed, may over/undershoot a little).
    Only the droop angle uses it, so the flag swings back when the wind changes suddenly.
    """
    level = float(np.clip(level, 0.0, 1.0))
    lift = level if lift is None else float(np.clip(lift, -0.2, 1.0))
    return {
        "droop_root": math.radians(72.0) * (1.0 - lift) ** 1.3,  # 付け根の垂れ角(水平から下向き)
        "droop_tip": math.radians(18.0) * (1.0 - lift),  # 自由端はさらに垂れる
        "freq": 0.5 + 3.3 * level,  # はためきの周波数 [Hz]。風が強いほど高い
        "wavelength": 1.15 - 0.65 * level,  # 波長 / 旗の長さ。強風ほど波が詰まる
        "amp": math.radians(80.0) * float(smoothstep(0.10, 0.70, level)) + math.radians(7.0) * (1.0 - level),
        "tip_amp": math.radians(14.0) * float(smoothstep(0.25, 0.85, level)),  # 自由端の速いはためき
        "billow": 0.07 * float(smoothstep(0.1, 0.5, level)),  # 布が横にふくらむ量(旗の幅に対する比)
    }


def flag_samples(level, phase, slow, length, width, hoist_z, nu=17, nv=9, nf=129,
                 lag=0.5, ripple=0.0, tip_cycles=0.5, billow_scale=1.0,
                 lift=None, stretch=0.0, gravity=None):
    """Representative points of the flag in local axes (x downwind, y crosswind, z up).

    Returns an (nu, nv, 3) array. Each row is a curve integrated from direction angles,
    so its length is preserved: phi = droop angle, psi = crosswind slope of the travelling waves.
    phase: primary wave phase [rad] (integrated over time, so frequency changes cause no jumps).
    slow: slow clock [rad] for the gust-like amplitude changes.
    lift: level with cloth inertia (droop angle). stretch: how much the cloth is stretched at the hoist
    (0.05 = 5%), falling to 0 at the free end. gravity: strength of the slight sag (default GRAVITY_FEEL).
    """
    g = GRAVITY_FEEL if gravity is None else gravity
    q = flag_shape_params(level, lift)
    lv = float(np.clip(level, 0.0, 1.0))
    s = np.linspace(0.0, 1.0, nf)[None, :]
    v = np.linspace(0.0, 1.0, nv)[:, None]
    ds = length / (nf - 1)
    calm = 1.0 - lv

    # 垂れ角: 自由端ほど、また下の行ほど少し垂れる。無風時はゆっくり揺れる
    sway = math.radians(5.0) * calm * math.sin(0.8 * slow + 0.4)
    phi = q["droop_root"] + sway + q["droop_tip"] * s**2 + math.radians(10.0) * calm * v * s
    # 重力: 強風でも自由端と下辺がほんの少し垂れる(布の重さ)
    phi = phi + g * (math.radians(3.0) * s**2 + math.radians(1.0) * v * s**1.5)
    phi = np.minimum(phi, math.radians(88.0))

    # 横向きの波。掛け縁で0、自由端ほど急に大きくなる(実際の旗と同じ)
    k = 2.0 * math.pi / q["wavelength"]
    env = (np.exp(1.7 * s) - 1.0) / (math.exp(1.7) - 1.0)
    # 突風: 周期の違うゆっくりした揺らぎを重ねて、単調な繰り返しに見えないようにする
    breathing = 1.0 + 0.18 * math.sin(slow) + 0.08 * math.sin(2.31 * slow + 1.1) + 0.05 * math.sin(5.7 * slow + 0.3)
    # 下の行ほど位相が遅れる(旗が斜めにねじれる)
    delay = lag * v * (0.4 + 0.6 * s)
    # 比が無理数に近い波を重ねる: 波形が周期的に揃わず、自然に崩れる
    wave = (
        np.cos(k * s - phase - delay)
        + 0.38 * np.cos(1.62 * k * s - 1.62 * phase + 1.3 - 1.4 * delay)
        + 0.20 * np.cos(2.41 * k * s - 2.41 * phase + 2.9 - 1.7 * delay)
    )
    # 縦方向のしわ: 縦にも波長を持ち、風下へ進む
    rip = np.cos(1.7 * k * s - 1.3 * phase + 2.0 * math.pi * 1.3 * v) * float(smoothstep(0.1, 0.6, lv))
    # 自由端の速いはためき(自由端の縁が縦方向に波打つ)
    tip = q["tip_amp"] * s**3 * np.cos(2.6 * k * s - 2.6 * phase + 2.0 * math.pi * tip_cycles * v + 1.0)
    psi = q["amp"] * breathing * env * ((1.0 - 0.2 * v) * wave / 1.4 + ripple * rip) + tip

    direction = np.stack([np.cos(phi) * np.cos(psi), np.sin(psi), -np.sin(phi) * np.cos(psi)], axis=-1)
    # 布の伸び: 掛け縁ほど強く引かれて伸び、自由端では伸びない
    step = ds * (1.0 + stretch * (1.0 - s))
    points = np.zeros((nv, nf, 3))
    points[:, 1:, :] = np.cumsum(direction[:, 1:, :] * step[..., None][:, 1:, :], axis=1)
    # 自重で、上の行ほど(下の行を支えるぶん)わずかに縦に伸びる
    points[:, :, 2] += hoist_z - width * (v + 0.02 * g * (v - 0.5 * v**2))
    # 布のふくらみ: 幅の中ほどが横に膨らむ(旗に立体感が出る)
    points[:, :, 1] += billow_scale * q["billow"] * width * np.sin(math.pi * v) * env * (0.6 + 0.4 * math.cos(0.5 * phase))
    columns = np.arange(nu) * (nf - 1) // (nu - 1)
    return np.transpose(points[:, columns, :], (1, 0, 2))


class SplineFlagSurface:
    """Maps particle (u, v) coordinates to flag positions with a tensor-product cubic spline."""

    NU = 17
    NV = 9

    def __init__(self, u, v, length, width, hoist_z):
        self.length, self.width, self.hoist_z = length, width, hoist_z
        self.mu, _ = spline_interp_matrix(u, self.NU)
        self.mv, _ = spline_interp_matrix(v, self.NV)

    def positions(self, level, phase, slow, lift=None, stretch=0.0):
        samples = flag_samples(level, phase, slow, self.length, self.width, self.hoist_z, self.NU, self.NV,
                               lift=lift, stretch=stretch)
        nu, nv = samples.shape[0], samples.shape[1]
        grid = np.ascontiguousarray(samples.transpose(1, 0, 2)).reshape(nv, nu * 3)
        across = (self.mv @ grid).reshape(-1, nu, 3)  # 幅方向を先に補間(行列積)
        return np.einsum("ni,nic->nc", self.mu, across)  # 続いて長さ方向


class ClothWindFlag:
    """PBD cloth flag driven by the current wind force."""

    LENGTH = 0.90
    WIDTH = 0.55
    CELL = 0.05  # 粒子(シミュレーション)の間隔
    VISUAL_SUBDIV = 2  # 描画メッシュを粒子間隔の何分の1にするか。大きいほど滑らか。1で従来の粗さ
    POLE_HEIGHT = 2.6
    THETA_MAX_DEG = 75.0
    BANK_BITS = 6
    FLUTTER_AMP = 0.15
    FLUTTER_HZ = 3.0
    USE_FORCE_FIELDS = True  # False の旗(cloth-vel)は、定常力場を登録しない

    def __init__(self, scene, base_xy, scale=1.0, n_envs=0, style="cloth", along=(1.0, 0.0)):
        """along: 布が最初に伸びる水平方向。カメラの視線に垂直にすると、旗の広い面が見える。
        eval.py の視点は +x 側から -x 方向を見るので (0, 1)、compare_arena.py は -y 側から見るので (1, 0)。"""
        self.style = style
        self.base_xy = base_xy
        self._along = _unit_xy(along)
        self.pole_height = self.POLE_HEIGHT * scale
        length, width, self.cell = self.LENGTH * scale, self.WIDTH * scale, self.CELL * scale
        self.length, self.width = length, width
        self.visual_cell = self.cell / max(1, int(self.VISUAL_SUBDIV))
        self.nx = int(round(length / self.visual_cell)) + 1
        self.nz = int(round(width / self.visual_cell)) + 1
        self._azimuth = self._along.copy()  # 最初の向き(風が吹くまで保つ)
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
            f"flag_{self.nx}x{self.nz}_{length:.3f}x{width:.3f}_{self._along[0]:+.3f}_{self._along[1]:+.3f}.obj",
        )
        write_flag_mesh(mesh_path, length, width, self.nx, self.nz, along=self._along)
        self.cloth = scene.add_entity(
            material=gs.materials.PBD.Cloth(),
            morph=gs.morphs.Mesh(file=mesh_path, pos=(base_xy[0], base_xy[1], self.pole_height)),
            surface=_cloth_surface(_CLOTH_VIS_MODE[style]),
        )

        a_max = G * math.tan(math.radians(self.THETA_MAX_DEG))
        self.bank = WindBank(a_max, self.BANK_BITS)
        if self.USE_FORCE_FIELDS:
            self._register_force_fields(scene)

        _add_pole(scene, base_xy, self.pole_height, scale)

    def _register_force_fields(self, scene):
        """風の加速度を二進の重みで近似するための、定常力場(x・y方向、正負、各BANK_BITS個)を登録する。"""
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

    def _pin_hoist_edge(self):
        """布の根元(支柱側の縦の辺)の粒子を固定する。Genesisの版による名前の違いに備え、候補を順に試す。"""
        positions = hoist_world_positions(self.base_xy, self.pole_height, self.visual_cell, self.nz)
        errors = []
        # 1. 一番近い粒子を探して fix_particles(複数形)で固定
        try:
            for position in positions:
                self.cloth.fix_particles(self.cloth.find_closest_particle(position))
            return True
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        # 2. 粒子ごとに fix_particle(単数形)で固定
        try:
            for position in positions:
                try:
                    idx = self.cloth.find_closest_particle(position)
                except TypeError:
                    idx = self.cloth.find_closest_particle(position, 0)
                idx = int(idx.item()) if hasattr(idx, "item") else int(idx)
                self.cloth.fix_particle(idx)
            return True
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        # 3. 頂点の並び順(各行の先頭)を粒子の番号とみなして固定
        try:
            self.cloth.fix_particles(particles_idx_local=[j * self.nx for j in range(self.nz)])
            return True
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        print("[警告] 布を支柱に固定できませんでした(旗が落ちる可能性があります): " + " / ".join(str(e) for e in errors))
        return False

    def after_build(self):
        """Pin the cloth's hoist edge after scene.build()."""
        self._pin_hoist_edge()
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

    def _track_wind(self, wind_vec, dt):
        """Update strength, direction and smoothed level from the force vector [N]."""
        wind = np.asarray(wind_vec, dtype=float).reshape(3)
        self.strength = float(np.linalg.norm(wind))
        self._time += dt
        horizontal = wind[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if self.strength > CALM_N and horizontal_norm > 1e-9:
            self._azimuth = horizontal / horizontal_norm
        target_level = min(self.strength / FULL_SCALE_N, 1.0)
        self._level += (target_level - self._level) * min(1.0, dt / SMOOTHING_S)

    def update(self, wind_vec, dt):
        """Apply the current force vector [N] to the cloth visualization."""
        self._track_wind(wind_vec, dt)
        if not (self._built and self.wind_enabled):
            return

        theta = math.radians(self.THETA_MAX_DEG) * self._level
        acceleration = G * math.tan(theta)
        acceleration *= 1.0 + self.FLUTTER_AMP * self._level * math.sin(
            2.0 * math.pi * self.FLUTTER_HZ * self._time
        )
        self._set_axis("x", acceleration * self._azimuth[0])
        self._set_axis("y", acceleration * self._azimuth[1])


def spring_step(x, v, target, omega, zeta, dt):
    """Damped spring (cloth inertia). Substeps keep it stable for large omega * dt."""
    n = max(1, int(math.ceil(omega * dt / 0.25)))
    h = dt / n
    for _ in range(n):
        v += (omega * omega * (target - x) - 2.0 * zeta * omega * v) * h
        x += v * h
    return x, v


def _to_numpy(array):
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=float)


class SurfaceWindFlag(ClothWindFlag):
    """Flag whose shape is computed by formula and written to the cloth particles.

    The PBD cloth only carries the smooth render mesh: every particle is fixed and moved by
    set_particles_pos, so no cloth solving is needed. If the setup fails (API differences),
    it falls back to the physical PBD cloth driven by force fields.
    """

    def __init__(self, scene, base_xy, scale=1.0, n_envs=0, style="surface", along=(1.0, 0.0)):
        super().__init__(scene, base_xy, scale=scale, n_envs=n_envs, style=style, along=along)
        self.n_envs = n_envs
        self.kinematic = False
        self._surface = None
        self._angle = math.atan2(self._along[1], self._along[0])  # 最初の向き(along)
        self._phase = 0.0
        self._slow = 0.0
        self._broken = False
        self._lift, self._lift_v = 0.0, 0.0  # 垂れ角用の風の強さ(慣性つき)
        self._tension, self._tension_v = 0.0, 0.0  # 布の伸びの元になる張力(ばね)
        self.stretch = 0.0

    def _write(self, positions):
        poses = np.asarray(positions, dtype=np.float32)
        if self.n_envs > 0:
            poses = np.tile(poses[None], (self.n_envs, 1, 1))
        self.cloth.set_particles_pos(torch.as_tensor(poses, dtype=torch.float32, device=gs.device))

    def _world_positions(self):
        local = self._surface.positions(self._level, self._phase, self._slow, self._lift, self.stretch)
        c, s = math.cos(self._angle), math.sin(self._angle)
        world = np.empty_like(local)
        world[:, 0] = self.base_xy[0] + c * local[:, 0] - s * local[:, 1]
        world[:, 1] = self.base_xy[1] + s * local[:, 0] + c * local[:, 1]
        world[:, 2] = local[:, 2]
        return world

    def _setup_kinematic(self):
        rest = _to_numpy(self.cloth.get_particles_pos())
        if rest.ndim == 3:
            rest = rest[0]
        offset = rest[:, :2] - np.asarray(self.base_xy, dtype=float)[None, :]
        u = (offset @ self._along) / self.length  # 支柱から旗の先端へ(along方向)
        side = offset @ np.array([-self._along[1], self._along[0]])  # alongに垂直な水平方向
        v = (self.pole_height - rest[:, 2]) / self.width
        flat = np.ptp(side) < 0.05 * self.length
        spans = u.min() < 0.05 and u.max() > 0.95 and v.min() < 0.05 and v.max() > 0.95
        if not (flat and spans):
            raise ValueError("旗の粒子の初期配置が想定と違います")
        self._surface = SplineFlagSurface(u, v, self.length, self.width, self.pole_height)
        self._write(self._world_positions())
        written = _to_numpy(self.cloth.get_particles_pos())
        if written.ndim == 3:
            written = written[0]
        if np.abs(written - self._world_positions()).max() > 1e-3:
            raise RuntimeError("粒子位置を書き込めませんでした")
        indices = list(range(rest.shape[0]))
        try:
            self.cloth.fix_particles(particles_idx_local=indices)
        except (TypeError, ValueError):
            self.cloth.fix_particles(torch.as_tensor(indices, dtype=torch.int64, device=gs.device))

    def after_build(self):
        try:
            self._setup_kinematic()
            self.kinematic = True
            self._built = True
        except Exception as error:
            print(f"[警告] 式で形を計算する旗を使えないため、PBD布の物理計算に切り替えます: {error}")
            self.style = "cloth-mesh"  # 凡例の表示を、実際の動作に合わせる
            super().after_build()

    def update(self, wind_vec, dt):
        if not self.kinematic:
            return super().update(wind_vec, dt)
        if self._broken:
            return
        self._track_wind(wind_vec, dt)
        # 風下へ向きを変える: 角度を最短経路でなめらかに追従(支柱まわりの旋回)
        wobble = 0.09 * self._level * (math.sin(1.3 * self._slow + 0.5) + 0.5 * math.sin(3.1 * self._slow))
        target = math.atan2(self._azimuth[1], self._azimuth[0]) + wobble  # 強風ほど左右に少し振れる
        delta = (target - self._angle + math.pi) % (2.0 * math.pi) - math.pi
        self._angle += delta * min(1.0, dt / SWIVEL_S)
        # 位相は周波数を時間積分して進める(風が変わっても波形が飛ばない)
        self._phase += 2.0 * math.pi * flag_shape_params(self._level)["freq"] * dt
        self._slow += 2.0 * math.pi * 0.23 * dt
        if GRAVITY_FEEL > 0.0:
            # 布の慣性: 風が急に弱まると垂れ込んで揺り戻し、急に強まると少し行き過ぎて戻る
            self._lift, self._lift_v = spring_step(self._lift, self._lift_v, self._level, LIFT_OMEGA, LIFT_ZETA, dt)
            # 布の伸び縮み: 風に引かれて伸び、はためきのたびにぴんと張って、少し跳ねる
            target = self._level**1.2 * (0.65 + 0.35 * math.cos(self._phase - 0.6))
            self._tension, self._tension_v = spring_step(
                self._tension, self._tension_v, target, TENSION_OMEGA, TENSION_ZETA, dt
            )
            self.stretch = GRAVITY_FEEL * (STRETCH_STATIC + STRETCH_DYN * max(self._tension, 0.0))
        else:
            self._lift, self.stretch = self._level, 0.0
        try:
            self._write(self._world_positions())
        except Exception as error:
            self._broken = True
            print(f"[警告] 旗の位置を更新できなくなりました(旗は止まります): {error}")


class _ParticleAccess:
    """PBD布エンティティの粒子の位置・速度を読み書きするメソッドを選ぶ薄いアダプタ。

    Genesisの版によって、メソッド名が違うことがある。候補を順に試し、使えるものを選ぶ。
    全部だめな時は ok=False にして理由を保持する(旗を止めるだけで、全体は止めない)。
    """

    POS_GETTERS = ("get_particles_pos", "get_particles_position", "get_particle_pos")
    VEL_GETTERS = ("get_particles_vel", "get_particles_velocity", "get_particle_vel")
    VEL_SETTERS = ("set_particles_vel", "set_particles_velocity", "set_particle_vel")

    def __init__(self, entity):
        self.entity = entity
        self.get_pos = self._pick(self.POS_GETTERS)
        self.get_vel = self._pick(self.VEL_GETTERS)
        self.set_vel = self._pick(self.VEL_SETTERS)
        missing = [
            name
            for name, function in (
                ("粒子位置の取得", self.get_pos),
                ("粒子速度の取得", self.get_vel),
                ("粒子速度の設定", self.set_vel),
            )
            if function is None
        ]
        self.ok = not missing
        self.reason = ("、".join(missing) + "のメソッドが見つからない") if missing else ""

    def _pick(self, names):
        for name in names:
            function = getattr(self.entity, name, None)
            if callable(function):
                return function
        return None


class VelocityClothWindFlag(ClothWindFlag):
    """PBD cloth flag whose wind is added to the particle velocities every step (cloth-vel).

    cloth-mesh は定常力場のオン/オフで風を近似するが、こちらは布の粒子の速度に、風下向きの
    加速度を毎ステップ直接足す。風の強さに比例した加速度に、速度に比例する減衰(空気抵抗の代わり)、
    強風時に支柱から先端へ進む波打ち、速度の上限(物理の発散を防ぐ安全弁)をつけてある。
    粒子の読み書きのメソッドが見つからない版では、旗は垂れ下がったまま(警告を1度だけ表示)。
    """

    USE_FORCE_FIELDS = False
    WIND_ACCEL = 25.0  # [m/s^2] 風が FULL_SCALE_N の時の風下向き加速度。重力(9.8)との釣り合いで、
    #   風の強さ0.5で約50度、1.0で約68度持ち上がる
    DAMPING_PER_S = 1.5  # [1/s] 粒子速度の減衰(ばたつきすぎを抑える)
    MAX_SPEED = 8.0  # [m/s] 粒子速度の上限
    FLUTTER_ACCEL = 14.0  # [m/s^2] 強風時に布を横方向へ波打たせる加速度の大きさ
    FLUTTER_HZ = 2.5  # はためきの周波数
    FLUTTER_WAVE = 9.0  # [rad/m] 波が支柱から先端へ進む時の、距離あたりの位相の進み

    def __init__(self, scene, base_xy, scale=1.0, n_envs=0, style="cloth-vel", along=(1.0, 0.0)):
        super().__init__(scene, base_xy, scale=scale, n_envs=n_envs, style=style, along=along)
        self._access = None
        self._warned = False

    def _warn(self, message):
        if not self._warned:
            print(f"[注意] 風の旗: {message}")
            self._warned = True

    def after_build(self):
        super().after_build()
        self._access = _ParticleAccess(self.cloth)
        if not self._access.ok:
            self._warn(f"風を旗に伝えられません({self._access.reason})。旗は垂れ下がったままになります")

    def wind_accel(self, positions, velocities, dt):
        """布の粒子に加える速度の変化 dv (N,3) を返す(Genesis非依存。単体テストできる)。

        positions  : (N,3) 粒子の位置(はためきの位相を、支柱からの距離で決めるのに使う)
        velocities : (N,3) 粒子の速度(減衰に使う)
        dv = ( 風下向きの加速度 - 減衰 x 速度 + 波打ち ) x dt
        """
        azimuth = np.array([self._azimuth[0], self._azimuth[1], 0.0])
        side = np.array([-self._azimuth[1], self._azimuth[0], 0.0])  # 水平で風向きに垂直
        push = self.WIND_ACCEL * self._level * azimuth
        base = np.asarray(self.base_xy, dtype=float)
        distance = np.linalg.norm(positions[:, :2] - base[None, :], axis=1)
        phase = 2.0 * math.pi * self.FLUTTER_HZ * self._time - self.FLUTTER_WAVE * distance
        flutter = (self.FLUTTER_ACCEL * self._level * np.sin(phase))[:, None] * side[None, :]
        return (push[None, :] - self.DAMPING_PER_S * velocities + flutter) * dt

    def update(self, wind_vec, dt):
        self._track_wind(wind_vec, dt)
        if self._built and self._access is not None and self._access.ok:
            self._push_wind_to_particles(dt)

    def _push_wind_to_particles(self, dt):
        access = self._access
        try:
            velocity = _to_numpy(access.get_vel())
            position = _to_numpy(access.get_pos())
            n = position.shape[-2]
            v = velocity.reshape(-1, n, 3)
            p = position.reshape(-1, n, 3)[0]  # はためきの位相は環境0の位置で決める(全環境に同じ旗)
            new_v = v + self.wind_accel(p, v[0], dt)[None, :, :]
            speed = np.linalg.norm(new_v, axis=-1, keepdims=True)
            new_v = new_v * np.minimum(1.0, self.MAX_SPEED / (speed + 1e-9))
            access.set_vel(torch.as_tensor(new_v.reshape(velocity.shape), dtype=torch.float32, device=gs.device))
        except Exception as error:  # noqa: BLE001  版差で形状・引数が合わない場合は、旗だけ止めて続行する
            access.ok = False
            self._warn(f"粒子の速度を更新できませんでした({type(error).__name__}: {error})。旗への風の反映を止めます")


def wind_flag_scene_kwargs(style="surface", scale=1.0):
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
    pbd_candidates = [dict(particle_size=ClothWindFlag.CELL * scale)]
    if style == "surface":
        # surfaceの旗は、粒子を全部固定して式で動かすだけで、布の計算は要らない。粒子の動ける範囲を旗の周りに絞り、
        # 拘束の反復を最小にして、PBDソルバーが毎ステップにする仕事を減らす(通らない版では、下の候補に進む)。
        # ※ surface が物理のPBD布に切り替わった時(after_buildの警告)は、反復が少ないぶん布が少し伸びる
        lite = dict(lower_bound=VELOCITY_PBD_LOWER, upper_bound=VELOCITY_PBD_UPPER,
                    particle_size=ClothWindFlag.CELL * scale)
        iterations = dict(max_stretch_solver_iterations=1, max_bending_solver_iterations=1,
                          max_volume_solver_iterations=1, max_density_solver_iterations=1,
                          max_viscosity_solver_iterations=1)
        pbd_candidates[:0] = [dict(lite, **iterations), dict(lite)]
    if style == "cloth-vel":
        # 粒子の範囲を実際に使う範囲に絞り、伸びの拘束の反復を増やす(通らない版では、下の候補に進む)
        pbd_candidates.insert(
            0,
            dict(
                lower_bound=VELOCITY_PBD_LOWER,
                upper_bound=VELOCITY_PBD_UPPER,
                particle_size=ClothWindFlag.CELL * scale,
                max_stretch_solver_iterations=8,
            ),
        )
    for candidate in pbd_candidates:
        try:
            kwargs["pbd_options"] = gs.options.PBDOptions(**candidate)
            break
        except Exception:  # noqa: BLE001  (TypeErrorだけでなく、pydanticの検証エラーも拾う)
            continue
    return kwargs


def create_wind_flag(scene, base_xy, scale=1.0, n_envs=0, style="surface", along=(1.0, 0.0)):
    """Create the requested flag; fall back surface -> PBD cloth -> rigid plate on failure.

    along: 旗が最初に伸びる水平方向。カメラの視線に垂直にすると、旗の広い面が見える
           (eval.py の視点なら (0, 1)、compare_arena.py の視点なら (1, 0))。
    """
    if style not in ALL_STYLES:
        raise ValueError(f"style must be one of {ALL_STYLES}: {style}")
    if style == "surface":
        try:
            return SurfaceWindFlag(scene, base_xy, scale=scale, n_envs=n_envs, style="surface", along=along)
        except Exception as error:
            print(f"[警告] スプライン曲面の旗を作成できないため、PBD布の旗に切り替えます: {error}")
            style = "cloth-mesh"
    if style == "cloth-vel":
        try:
            return VelocityClothWindFlag(scene, base_xy, scale=scale, n_envs=n_envs, style="cloth-vel", along=along)
        except Exception as error:
            print(f"[警告] 粒子の速度で風を伝える布の旗を作成できないため、定常力場のPBD布の旗に切り替えます: {error}")
            style = "cloth-mesh"
    if style in _CLOTH_VIS_MODE:
        try:
            return ClothWindFlag(scene, base_xy, scale=scale, n_envs=n_envs, style=style, along=along)
        except Exception as error:
            print(f"[警告] PBD布を作成できないため、剛体板の旗に切り替えます: {error}")
    return PlateWindFlag(scene, base_xy, scale=scale, n_envs=n_envs, along=along)
