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
# 床の反射(評価・アリーナ用の見た目だけ。物理には影響しない)
# ---------------------------------------------------------------------- #
FLOOR_ROUGHNESS = 0.55  # 小さいほど鏡に近い。若干の反射にするため、やや大きめ(暫定値。実機で見て調整)


def reflective_vis_options(**kwargs):
    """床の反射を有効にした VisOptions を作る(plane_reflection が無い版では、無しで作る)。"""
    try:
        return gs.options.VisOptions(plane_reflection=True, **kwargs)
    except TypeError:
        print("[注意] この版の VisOptions には plane_reflection がないため、床の反射なしで続けます")
        return gs.options.VisOptions(**kwargs)


def add_floor(scene, reflective=True):
    """床(Plane)を足す。reflective=True なら、若干つやのある床にする(通らなければ通常の床)。"""
    if reflective:
        try:
            surface = gs.surfaces.Default(color=(0.92, 0.92, 0.94, 1.0), roughness=FLOOR_ROUGHNESS)
            return scene.add_entity(gs.morphs.Plane(), surface=surface)
        except (TypeError, AttributeError) as e:
            print(f"[注意] 床の表面設定が通らないため、通常の床にします: {e}")
    return scene.add_entity(gs.morphs.Plane())