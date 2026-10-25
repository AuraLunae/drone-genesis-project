"""
genesis_compat.py

Genesisの「版による違い」を1か所に集約する共通モジュール。

このプロジェクトでは、Genesisの版が変わるたびにAPIの食い違いで何度も止まった
(COM_link_idxの未初期化、外力APIの置き場所と名前の変更など)。
版に依存する呼び出しはここだけに置き、各スクリプトはここを経由する。
今後また版差で壊れた時は、このファイルだけを直せばよい。

確認済みの事実:
  - Genesis 1.2.3 : 外力は solver.apply_links_external_force(force, links_idx, ref="link_com", local=False)。
                    差動RPMがロール/ピッチのトルクを生まない不具合がある。
  - Genesis 1.4.x : 外力は solver.apply_links_external_wrench(force=..., torque=..., links_idx=...,
                    ref=gs.link_ref_frame.link_COM, local=False)。差動RPMのトルクは正しく生じる。
"""

import inspect
import time

import torch

import genesis as gs


def apply_com_force(drone, link_idx, force, n_envs=0):
    """ドローンの重心(link_com)に、並進力だけを加える(トルクは誘発しない)。

    drone    : DroneEntity
    link_idx : 力を加えるリンクのインデックス(drone.get_link("base_link").idx)
    force    : n_envs==0(単一環境)なら長さ3、バッチ環境なら形状 (n_envs, 3) のテンソル/配列 [N]
    n_envs   : 環境数。単一環境(scene.build()をn_envs無しで呼んだ場合)は0

    外力は毎ステップ自動でクリアされる(verify_physics.pyのテスト9で確認済み)ので、
    1ステップに1回、必要な力を加え直す。
    """
    force = torch.as_tensor(force, dtype=torch.float32, device=gs.device)
    if n_envs == 0:
        force = force.reshape(1, 3)          # (リンク数=1, 3)
    else:
        force = force.reshape(n_envs, 1, 3)  # (環境数, リンク数=1, 3)

    solver = drone.solver
    if hasattr(solver, "apply_links_external_wrench"):  # Genesis 1.4.x
        solver.apply_links_external_wrench(
            force=force,
            links_idx=[link_idx],
            ref=gs.link_ref_frame.link_COM,
            local=False,
        )
    else:  # Genesis 1.2.x
        solver.apply_links_external_force(
            force,
            links_idx=[link_idx],
            ref="link_com",
            local=False,
        )


class PropellerVisual:
    """プロペラの回転「表示」だけを、実際のRPMとは切り離して滑らかにする(物理には無関係)。

    Genesisは、各ステップでプロペラの実際の回転量(rpmに比例)を表示角度に足し込む。
    CF2Xは約14,500rpm(毎秒240回転)で回るため、100Hz前後のコマ送りで表示すると
    ワゴンホイール効果が起き、1コマあたりの回転が180°付近になるRPMでは、対称な
    プロペラは止まって見える(加速でRPMが上がると止まって見え、戻ると回って見える)。

    そこで、実際のRPMに比例しつつ、1ステップあたりの表示回転角を小さく抑えた
    表示用の角度を自前で積算し、Genesisの内部カウンタ(_propellers_revs)に上書きする。
    内部カウンタが見つからない版では何もしない(self.enabled=False)。

    使い方: scene.build() の後に生成し、毎ステップ
            drone.set_propellers_rpm(rpms) の直後、scene.step() の前に apply(rpms) を呼ぶ。
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
        """rpms: 長さ4のリスト/配列(単一環境)、または形状 (環境数, 4) のテンソル(バッチ環境)"""
        if not self.enabled:
            return
        revs = self.drone._propellers_revs  # 形状 (プロペラ数, 環境数)
        if self._revs is None:
            self._revs = torch.zeros_like(revs)
        rpm_tensor = torch.as_tensor(rpms, dtype=revs.dtype, device=revs.device).detach()
        rpm_tensor = rpm_tensor.reshape(-1, 1) if rpm_tensor.ndim == 1 else rpm_tensor.T
        self._revs = (self._revs + rpm_tensor * self.VISUAL_SCALE) % self._wrap
        revs.copy_(self._revs)


# ---------------------------------------------------------------------- #
# 鏡面の床と、描画・実行の軽量化(見た目と速さだけ。lite_rigid_options以外は物理の結果を変えない)
# ---------------------------------------------------------------------- #
FLOOR_COLOR = (0.07, 0.08, 0.10, 1.0)  # 床の色。白いと映り込みが見えないので暗くしてある(明るくしたければここ)
FLOOR_ROUGHNESS = 0.02                 # 小さいほど鏡に近い(0.0〜0.05の範囲で試す)
VIEW_EVERY_N_STEPS = 2                 # 画面の更新間隔[物理ステップ]。物理100Hzなら、2で50fps / 3で約33fps / 1で毎ステップ(重い)


def fast_init(backend, **kwargs):
    """gs.init を軽量設定で呼ぶ(32bit・警告以外のログなし・performance_mode)。通らない引数があれば外してやり直す。

    あわせて、小さなテンソルばかりの処理でスレッドの取り合いが起きないよう、PyTorchのスレッド数を1にする
    (評価・アリーナ用。学習は train.py が自前で gs.init を呼ぶ)。
    """
    options = dict(precision="32", logging_level="warning", performance_mode=True)
    options.update(kwargs)
    try:
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        gs.init(backend=backend, **options)
    except TypeError:
        gs.init(backend=backend, **kwargs)


def _first_working(makers):
    """作る関数を順に試し、最初に通ったものを返す。TypeErrorだけでなく、pydanticの検証エラー(ValueError)も拾う。"""
    last = None
    for make in makers:
        try:
            return make()
        except Exception as error:  # noqa: BLE001
            last = error
    raise last


def lite_vis_options(mirror=True, **kwargs):
    """画面を出す時の VisOptions。影を切り、座標軸を消し、床を鏡面反射にする(見た目だけ)。

    通らない引数は、影 → 座標軸 → 反射 の順にあきらめて作り直す。
    """
    options = dict(shadow=False, show_world_frame=False)
    if mirror:
        options["plane_reflection"] = True
    options.update(kwargs)
    notes = {
        "shadow": "影を切れない",
        "show_world_frame": "座標軸を消せない",
        "plane_reflection": "床の反射ができない(鏡面になりません)",
    }
    while True:
        try:
            return gs.options.VisOptions(**options)
        except Exception:  # noqa: BLE001
            for key in ("shadow", "show_world_frame", "plane_reflection"):
                if key in options:
                    del options[key]
                    print(f"[注意] この版の VisOptions では{notes[key]}ため、その設定なしで続けます")
                    break
            else:
                raise


def lite_viewer_options(**kwargs):
    """画面を出す時の ViewerOptions。左上の操作ヒント(毎フレームの文字描画)を消す。通らなければ、そのまま作る。"""
    try:
        return gs.options.ViewerOptions(disable_help_text=True, **kwargs)
    except Exception:  # noqa: BLE001
        return gs.options.ViewerOptions(**kwargs)


def lite_rigid_options(dt):
    """評価・アリーナ用の RigidOptions。床との衝突判定と関節制限を切る(毎ステップの接触計算が無駄なので)。

    墜落の判定は高さ(termination_if_close_to_ground など)で行い、1機が自由飛行するだけなので、
    地面に触れる前に終わる。学習時と同じ dt/2 のサブステップは変えない(方策が学習した物理を保つため)。
    """
    return _first_working(
        [
            lambda: gs.options.RigidOptions(
                dt=dt,
                enable_collision=False,
                enable_joint_limit=False,
                constraint_solver=gs.constraint_solver.Newton,
            ),
            lambda: gs.options.RigidOptions(dt=dt, enable_collision=False, enable_joint_limit=False),
        ]
    )


_STEP_CAN_SKIP_DRAW = {}


def step_scene(scene, render=True):
    """scene.step() を呼ぶ。render=False の時は、画面(ビューア)の更新だけを省く(物理は同じだけ進む)。

    `update_visualizer` 引数が無い版では、普通に scene.step() を呼ぶ。
    """
    if render:
        scene.step()
        return
    if "ok" not in _STEP_CAN_SKIP_DRAW:
        try:
            _STEP_CAN_SKIP_DRAW["ok"] = "update_visualizer" in inspect.signature(scene.step).parameters
        except (TypeError, ValueError):
            _STEP_CAN_SKIP_DRAW["ok"] = False
    if _STEP_CAN_SKIP_DRAW["ok"]:
        scene.step(update_visualizer=False)
    else:
        scene.step()


class RealtimePacer:
    """シミュレーションの進みを実時間(1倍速)に合わせる。速く進みすぎた分だけ眠る(待つ間はCPUを使わない)。

    軽くしたぶん、実時間より速く進んで早送りに見えないようにする。遅れている時は何もしない。
    """

    def __init__(self, dt):
        self.dt = float(dt)
        self._t0 = None
        self._n = 0

    def wait(self):
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
            return
        self._n += 1
        ahead = self._t0 + self._n * self.dt - now
        if ahead > 0.002:
            time.sleep(ahead - 0.001)
        elif ahead < -0.25:  # 大きく遅れた(読み込みなど)時は、基準を取り直す
            self._t0, self._n = now, 0


def mirror_floor_surface():
    """暗くてつるつるの床の表面。白い床では、白い旗や明るい機体の映り込みが見えない。"""
    return _first_working(
        [
            lambda: gs.surfaces.Default(color=FLOOR_COLOR, roughness=FLOOR_ROUGHNESS),
            lambda: gs.surfaces.Default(color=FLOOR_COLOR),
        ]
    )


def add_floor(scene, reflective=True):
    """床(Plane)を足す。暗い色の床にする(鏡面にするのは VisOptions の plane_reflection=True、lite_vis_options)。

    reflective は互換のために残してある(今は常に暗い床。反射の有無は VisOptions で決まる)。
    表面の設定が通らなければ、通常の床にする。
    """
    try:
        return scene.add_entity(gs.morphs.Plane(), surface=mirror_floor_surface())
    except Exception as e:  # noqa: BLE001
        print(f"[注意] 床の表面設定が通らないため、通常の床にします: {e}")
    return scene.add_entity(gs.morphs.Plane())


def reflective_vis_options(**kwargs):
    """互換用。床を鏡面反射にした VisOptions(lite_vis_options と同じ)。"""
    return lite_vis_options(mirror=True, **kwargs)
