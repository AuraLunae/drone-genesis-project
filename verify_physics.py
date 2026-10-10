"""
verify_physics.py

このプロジェクトで「おそらくこう」「未検証」としてきたGenesis/CF2Xの物理的な
主張を、実際にシミュレーションを動かして測定し、白黒つけるための検証スクリプト。

特にフェーズ10で未解決だった「差動推力だけ(手動トルク補正無し)でロール/
ピッチのトルクが本当に物理的に伝わるか」を筆頭に、これまでの会話で
「確認できなかった」としてきた物理的性質を一通り測定する。

各テストは独立しており、1つのシーン・1機のドローンを使い回して
reset_drone()で初期化し直しながら順番に実行する(シーンの再構築は行わない)。

重要な制約: Genesisの set_propellers_rpm は「1ステップにつき1回しか呼べない」。
そのため、全テストで必ず PhysicsVerifier.step() 経由で
「RPM指令 → (任意で外力) → scene.step()」を1セットで実行する。

実行:
  uv run python verify_physics.py
  uv run python verify_physics.py --vis   # ビューアで見ながら実行(遅くなる)

結果は標準出力と logs/physics_verification_report.md の両方に出力する。
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import quat_to_xyz

DT = 0.01
BASE_RPM = 14468.429183500699  # CF2Xのホバリング基準RPM(公式サンプルの値)


class PhysicsVerifier:
    def __init__(self, drone, com_link_idx):
        self.drone = drone
        self.com_link_idx = com_link_idx
        self.results = []  # (タイトル, 詳細テキストのリスト)

    # ------------------------------------------------------------------ #
    # 基本ヘルパー
    # ------------------------------------------------------------------ #
    def reset_drone(self, pos=(0.0, 0.0, 5.0), quat=(1.0, 0.0, 0.0, 0.0)):
        """地面から十分離した高さで、静止・水平姿勢にリセットする"""
        self.drone.set_pos(torch.tensor(pos, dtype=torch.float32), zero_velocity=True)
        self.drone.set_quat(torch.tensor(quat, dtype=torch.float32), zero_velocity=True)

    def step(self, scene, rpms, ext_force=None):
        """RPM指令 → (任意で外力) → scene.step() を1セットで実行する。

        set_propellers_rpm は1ステップに1回しか呼べないため、全てのテストは
        必ずこのヘルパーを通して1ステップを進める。
        rpms: スカラー(4つ全て同じRPM)、または長さ4のシーケンス
        ext_force: 重心に加える外力 (fx, fy, fz) [N]。省略時は加えない"""
        if np.isscalar(rpms):
            rpms = [float(rpms)] * 4
        self.drone.set_propellers_rpm([float(r) for r in rpms])
        if ext_force is not None:
            self.drone.solver.apply_links_external_wrench(
                force=torch.tensor([list(ext_force)], dtype=torch.float32),
                links_idx=[self.com_link_idx],
                ref="link_com",
                local=False,
            )
        scene.step()

    def get_state(self):
        pos = np.array([float(v) for v in self.drone.get_pos()])
        vel = np.array([float(v) for v in self.drone.get_vel()])
        ang = np.array([float(v) for v in self.drone.get_ang()])
        att = np.array([float(v) for v in quat_to_xyz(self.drone.get_quat(), rpy=True, degrees=True)])
        return pos, vel, ang, att

    def log(self, title, lines):
        print(f"\n--- {title} ---")
        for line in lines:
            print(line)
        self.results.append((title, lines))

    # ------------------------------------------------------------------ #
    # テスト0: 重力
    # ------------------------------------------------------------------ #
    def test_gravity(self, scene):
        """0: プロペラ停止・自由落下の加速度が重力(-9.81 m/s^2)と一致するか"""
        self.reset_drone()
        _, vel0, _, _ = self.get_state()
        n_steps = 10
        for _ in range(n_steps):
            self.step(scene, 0.0)
        _, vel1, _, _ = self.get_state()
        accel_z = (vel1[2] - vel0[2]) / (n_steps * DT)
        self.log("0. 重力加速度(自由落下)", [
            f"測定された鉛直加速度: {accel_z:.3f} m/s^2",
            "期待値(標準重力): -9.81 m/s^2",
            f"判定: {'一致' if abs(accel_z - (-9.81)) < 0.5 else '不一致'}",
        ])
        return accel_z

    # ------------------------------------------------------------------ #
    # テスト1・2: ホバリング較正と推力則
    # ------------------------------------------------------------------ #
    def _measure_vertical_accel(self, scene, rpm, n_steps=15):
        self.reset_drone()
        _, vel0, _, _ = self.get_state()
        for _ in range(n_steps):
            self.step(scene, rpm)
        _, vel1, _, _ = self.get_state()
        return (vel1[2] - vel0[2]) / (n_steps * DT)

    def test_hover_calibration(self, scene):
        """1: BASE_RPMでの正味の鉛直加速度が0に近いか(ホバリング較正の確認)"""
        accel = self._measure_vertical_accel(scene, BASE_RPM)
        self.log("1. ホバリング較正(BASE_RPM)", [
            f"BASE_RPM = {BASE_RPM:.1f} での正味鉛直加速度: {accel:.4f} m/s^2",
            f"判定: {'ホバリング相当(ほぼ0)' if abs(accel) < 0.3 else '較正ズレの可能性あり'}",
        ])
        return accel

    def test_thrust_law(self, scene):
        """2: 推力がRPMの2乗に比例しているか(複数RPM倍率での回帰)"""
        multipliers = [0.6, 0.8, 1.0, 1.2, 1.4]
        g = 9.81
        points = []
        for k in multipliers:
            accel = self._measure_vertical_accel(scene, k * BASE_RPM, n_steps=8)
            points.append((k, accel + g))  # 加速度+重力 ≈ 推力による加速度成分

        # log(推力相当加速度) を log(k) に対して線形回帰 → 傾きが「指数」
        xs = np.log(np.array([k for k, _ in points]))
        ys = np.log(np.array([max(t, 1e-6) for _, t in points]))
        slope, _ = np.polyfit(xs, ys, 1)

        lines = [f"倍率k={k:.1f}: 推力相当加速度(a+g) = {t:.3f} m/s^2" for k, t in points]
        lines.append(f"回帰で得られた指数(傾き): {slope:.2f} (理論値 2.0 が F∝rpm^2 を意味する)")
        lines.append(f"判定: {'F∝rpm^2と整合' if 1.6 < slope < 2.4 else '単純な2乗則と乖離'}")
        self.log("2. 推力則(F ∝ rpm^n の n を測定)", lines)
        return slope

    # ------------------------------------------------------------------ #
    # テスト3: モーター応答遅れ
    # ------------------------------------------------------------------ #
    def test_motor_lag(self, scene):
        """3: RPMコマンドへの応答に一次遅れ(モーターの立ち上がり時間)があるか"""
        self.reset_drone()
        for _ in range(5):  # 定常ホバリング状態を作ってから
            self.step(scene, BASE_RPM)

        target_rpm = 1.4 * BASE_RPM
        _, vel, _, _ = self.get_state()
        vz_history = [vel[2]]
        for _ in range(8):
            self.step(scene, target_rpm)
            _, vel, _, _ = self.get_state()
            vz_history.append(vel[2])

        # 最初の1ステップでの加速度 vs 後半の平均加速度を比較
        accel_first = (vz_history[1] - vz_history[0]) / DT
        accel_later = (vz_history[-1] - vz_history[-4]) / (3 * DT)
        ratio = accel_first / accel_later if abs(accel_later) > 1e-6 else float("nan")

        self.log("3. モーター応答遅れの有無", [
            f"ステップ変化直後(1ステップ目)の加速度: {accel_first:.3f} m/s^2",
            f"数ステップ後の加速度: {accel_later:.3f} m/s^2",
            f"比率(1に近いほど遅れ無し、即座に全力が出ている): {ratio:.2f}",
            f"判定: {'遅れはほぼ無し(即時応答)' if 0.7 < ratio < 1.3 else '遅れ(立ち上がり)の可能性あり'}",
        ])
        return ratio

    # ------------------------------------------------------------------ #
    # テスト4・4b・5: 差動推力によるトルク伝達(フェーズ10の核心)
    # ------------------------------------------------------------------ #
    def _measure_angular_response(self, scene, rpms, n_steps=5):
        """短い時間(既定5ステップ=0.05秒)だけ差動RPMを与え、角速度と姿勢を測る。
        トルクが本当に伝わるなら角加速度は非常に大きくなる見込みなので、
        長く回すと機体が反転して測定が意味をなさなくなる。短時間で測る。"""
        self.reset_drone()
        for _ in range(n_steps):
            self.step(scene, rpms)
        _, _, ang, att = self.get_state()
        return ang, att, n_steps * DT

    def test_pitch_torque(self, scene):
        """4(核心): 差動推力だけ(手動トルク補正無し)でピッチ角速度が発生するか"""
        delta = 0.15 * BASE_RPM
        # quadcopter_controller.pyのミキサー規約: M1,M4がpitch+、M2,M3がpitch-
        rpms = [BASE_RPM + delta, BASE_RPM - delta, BASE_RPM - delta, BASE_RPM + delta]
        ang, att, t = self._measure_angular_response(scene, rpms)
        pitch_rate = ang[1]

        self.log("4. 差動推力によるピッチトルク伝達(フェーズ10の核心問題)", [
            f"RPM配分: {[f'{r:.0f}' for r in rpms]} (差分 ±{delta:.0f})",
            f"{t:.2f}秒後のピッチ角速度: {pitch_rate:.3f} rad/s (推定角加速度 {pitch_rate / t:.1f} rad/s^2)",
            f"{t:.2f}秒後の姿勢(roll,pitch,yaw): {att.round(2)} deg",
            "判定: " + (
                "手動補正無しでもトルクは伝わっている" if abs(pitch_rate) > 0.05
                else "手動補正無しではトルクがほぼ伝わっていない(ユーザーの修正が必要だった根拠)"
            ),
        ])
        return pitch_rate

    def test_roll_torque(self, scene):
        """4b: 同様にロール方向も確認(同じメカニズムなので念のため)"""
        delta = 0.15 * BASE_RPM
        # ミキサー規約: M1,M2がroll-、M3,M4がroll+
        rpms = [BASE_RPM - delta, BASE_RPM - delta, BASE_RPM + delta, BASE_RPM + delta]
        ang, att, t = self._measure_angular_response(scene, rpms)
        roll_rate = ang[0]

        self.log("4b. 差動推力によるロールトルク伝達", [
            f"RPM配分: {[f'{r:.0f}' for r in rpms]} (差分 ±{delta:.0f})",
            f"{t:.2f}秒後のロール角速度: {roll_rate:.3f} rad/s (推定角加速度 {roll_rate / t:.1f} rad/s^2)",
            f"{t:.2f}秒後の姿勢(roll,pitch,yaw): {att.round(2)} deg",
            "判定: " + (
                "手動補正無しでもトルクは伝わっている" if abs(roll_rate) > 0.05
                else "手動補正無しではトルクがほぼ伝わっていない"
            ),
        ])
        return roll_rate

    def test_yaw_reaction_torque(self, scene):
        """5: プロペラ回転方向の差によるヨー反トルクが発生するか"""
        delta = 0.15 * BASE_RPM
        # ミキサー規約: M1,M3がyaw-、M2,M4がyaw+
        rpms = [BASE_RPM - delta, BASE_RPM + delta, BASE_RPM - delta, BASE_RPM + delta]
        ang, att, t = self._measure_angular_response(scene, rpms)
        yaw_rate = ang[2]

        self.log("5. ヨー反トルク(プロペラ回転方向の差)", [
            f"RPM配分: {[f'{r:.0f}' for r in rpms]} (差分 ±{delta:.0f})",
            f"{t:.2f}秒後のヨー角速度: {yaw_rate:.3f} rad/s (推定角加速度 {yaw_rate / t:.1f} rad/s^2)",
            f"判定: {'反トルクは発生している' if abs(yaw_rate) > 0.02 else '反トルクがほぼ発生していない'}",
        ])
        return yaw_rate

    # ------------------------------------------------------------------ #
    # テスト6: 空力抵抗
    # ------------------------------------------------------------------ #
    def test_aerodynamic_drag(self, scene):
        """6: 水平方向の速度が、空気抵抗で減衰するか

        注意: インパルス印加後に外力が持続する仕様(クリアされない)だと、速度は
        減衰ではなく増加し続ける。その場合は残存比率が1.0を超えるので、
        テスト9(外力の持続性)の結果と合わせて解釈すること。"""
        self.reset_drone()
        for _ in range(5):
            self.step(scene, BASE_RPM)

        # 1ステップだけ外力を加え、水平方向に弾く
        self.step(scene, BASE_RPM, ext_force=(1.0, 0.0, 0.0))
        _, vel0, _, _ = self.get_state()
        vx0 = vel0[0]

        n_steps = 30
        for _ in range(n_steps):
            self.step(scene, BASE_RPM)  # 鉛直方向だけホバリング維持、外力は加えない
        _, vel1, _, _ = self.get_state()
        vx1 = vel1[0]

        decay_ratio = vx1 / vx0 if abs(vx0) > 1e-6 else float("nan")
        if decay_ratio > 1.1:
            verdict = "速度が増加している(外力が持続している疑い。テスト9を参照)"
        elif decay_ratio > 0.9:
            verdict = "空力抵抗はほぼ無い(速度保存)"
        else:
            verdict = "何らかの減衰あり(抵抗が存在する)"
        self.log("6. 空力抵抗(水平方向の速度減衰)の有無", [
            f"インパルス直後の水平速度: {vx0:.3f} m/s",
            f"{n_steps}ステップ({n_steps * DT:.2f}秒)後の水平速度: {vx1:.3f} m/s",
            f"残存比率: {decay_ratio:.3f} (1.0に近いほど抵抗が無い)",
            f"判定: {verdict}",
        ])
        return decay_ratio

    # ------------------------------------------------------------------ #
    # テスト7: 地面との衝突
    # ------------------------------------------------------------------ #
    def test_ground_collision(self, scene):
        """7: 地面(Plane)との衝突判定があるか"""
        # 地面よりかなり下にワープさせ、プロペラ停止で数ステップ様子を見る
        self.reset_drone(pos=(2.0, 2.0, -1.0))
        z_positions = []
        for _ in range(10):
            self.step(scene, 0.0)
            pos, _, _, _ = self.get_state()
            z_positions.append(pos[2])

        still_sinking = z_positions[-1] < z_positions[0] - 0.01
        self.log("7. 地面との衝突判定の有無", [
            f"Z座標の推移(地面より下からスタート): {[round(float(z), 3) for z in z_positions]}",
            f"判定: {'衝突判定は無い(地面下に沈み続ける)' if still_sinking else '何らかの反力で止まっている(衝突判定がある)'}",
        ])
        return z_positions

    # ------------------------------------------------------------------ #
    # テスト8: 風力(外力)印加の純粋性
    # ------------------------------------------------------------------ #
    def test_wind_force_purity(self, scene):
        """8: apply_links_external_wrench(ref='link_com')が、並進力のみでトルクを誘発しないか"""
        self.reset_drone()
        for _ in range(5):
            self.step(scene, BASE_RPM)  # 鉛直は打ち消しておく

        force = (0.05, 0.03, 0.0)
        n_steps = 20
        _, vel0, ang0, _ = self.get_state()
        for _ in range(n_steps):
            self.step(scene, BASE_RPM, ext_force=force)
        _, vel1, ang1, _ = self.get_state()

        accel_measured = (vel1[:2] - vel0[:2]) / (n_steps * DT)
        ang_vel_change = float(np.linalg.norm(ang1 - ang0))

        self.log("8. 風力(apply_links_external_wrench)の純粋性(トルク誘発の有無)", [
            f"与えた水平外力: {force[:2]} N",
            f"測定された水平加速度: {accel_measured.round(4)} m/s^2 (向きが外力と一致するか確認)",
            f"角速度の変化量ノルム: {ang_vel_change:.4f} rad/s",
            f"判定: {'並進力のみ、余計なトルク誘発は無い' if ang_vel_change < 0.05 else '意図しないトルクが誘発されている可能性'}",
        ])
        return ang_vel_change

    # ------------------------------------------------------------------ #
    # テスト9: 外力が毎ステップ自動クリアされるか
    # ------------------------------------------------------------------ #
    def test_external_force_persistence(self, scene):
        """9: apply_links_external_force で加えた外力が、次のステップに持ち越されるか

        WindHoverEnv は毎ステップ風を加えている。もし外力が自動でクリアされず
        蓄積される仕様なら、風が際限なく膨らんでしまう。それを確認する。
        1ステップだけ外力を加え、その後は外力ゼロで数ステップ進めて、
        速度が増え続けるか(=持続)、止まるか(=クリアされる)を見る。"""
        self.reset_drone()
        for _ in range(5):
            self.step(scene, BASE_RPM)

        _, vel_before, _, _ = self.get_state()
        self.step(scene, BASE_RPM, ext_force=(0.05, 0.0, 0.0))  # 1ステップだけ加える
        _, vel_after_force, _, _ = self.get_state()
        dv_force_step = vel_after_force[0] - vel_before[0]

        later_dvs = []
        prev_vx = vel_after_force[0]
        for _ in range(5):
            self.step(scene, BASE_RPM)  # 外力は加えない
            _, vel, _, _ = self.get_state()
            later_dvs.append(vel[0] - prev_vx)
            prev_vx = vel[0]
        avg_later_dv = float(np.mean(later_dvs))
        persistence_ratio = avg_later_dv / dv_force_step if abs(dv_force_step) > 1e-9 else float("nan")

        if abs(persistence_ratio) < 0.2:
            verdict = "外力は毎ステップ自動でクリアされる(1ステップ分のみ作用)"
        elif abs(persistence_ratio) > 0.8:
            verdict = "外力が次ステップ以降も持続している(蓄積の危険あり。WindHoverEnvの設計を要確認)"
        else:
            verdict = "部分的に持続している可能性(要追加確認)"
        self.log("9. 外力の持続性(毎ステップ自動クリアされるか)", [
            f"外力を加えたステップでの速度変化 Δvx: {dv_force_step:.5f} m/s",
            f"その後(外力ゼロ)の1ステップあたり平均 Δvx: {avg_later_dv:.5f} m/s",
            f"持続比率(0なら自動クリア、1なら毎ステップ同じ力が持続): {persistence_ratio:.3f}",
            f"判定: {verdict}",
        ])
        return persistence_ratio

    # ------------------------------------------------------------------ #
    def write_report(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Genesis/CF2X 物理検証レポート\n\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for title, lines in self.results:
                f.write(f"## {title}\n\n")
                for line in lines:
                    f.write(f"- {line}\n")
                f.write("\n")
        print(f"\n=== レポートを保存しました: {path} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action="store_true", help="ビューアを表示する(遅くなる)")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 5.0)),
    )
    scene.build()
    com_link_idx = drone.get_link("base_link").idx

    verifier = PhysicsVerifier(drone, com_link_idx)

    print("=" * 70)
    print(" Genesis / CF2X 物理モデル検証")
    print("=" * 70)

    verifier.test_gravity(scene)
    verifier.test_hover_calibration(scene)
    verifier.test_thrust_law(scene)
    verifier.test_motor_lag(scene)
    verifier.test_pitch_torque(scene)
    verifier.test_roll_torque(scene)
    verifier.test_yaw_reaction_torque(scene)
    verifier.test_aerodynamic_drag(scene)
    verifier.test_ground_collision(scene)
    verifier.test_wind_force_purity(scene)
    verifier.test_external_force_persistence(scene)

    verifier.write_report("logs/physics_verification_report.md")


if __name__ == "__main__":
    main()