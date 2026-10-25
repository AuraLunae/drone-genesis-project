"""
arena_scoring.py

AI・PID・人間を同じ課題で比較するための採点ロジック。Genesisには依存しない
(位置・姿勢・時刻だけを受け取る)ので、シミュレーション無しで単体テストできる。

課題: 順番に出てくる複数の目標地点(ウェイポイント)へ、できるだけ速く正確に近づく。

採点(各ドローン独立に集計し、合計点で順位をつける):
  到達ボーナス   : 目標の半径(capture_radius)以内に入ったら +waypoint_points
  スピードボーナス: その目標が出てから早く着くほど加点。waypoint_time_limit秒で0になる
                   (最大 +speed_bonus_max)
  接近ポイント   : 目標に近いほど毎秒加点。near_points_per_s * exp(-距離/near_scale)
                   (目標のすぐ近くに居続けるほど稼げる。遠くをさまよっても稼げない)
  完走ボーナス   : 全ての目標に到達したら、残り時間に応じて最大 +finish_bonus_max
  失敗ペナルティ : 墜落・範囲外・転倒・回転暴走で失敗した時に fail_penalty(1回だけ)
                   失敗した時点で、そのドローンの競技は終了する

失敗条件は、AIの学習時の終了条件(WindHoverEnv / train.py の env_cfg)と同じ値にしてあり、
AI・PID・人間の全員に同じ基準を適用する。

注意: 到達の判定は「半径内に入った瞬間」で、速度は問わない(通過でも到達になる)。
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class ScoringConfig:
    capture_radius: float = 0.25        # [m] この距離以内に入ったら到達
    waypoint_points: float = 100.0      # 到達ボーナス
    speed_bonus_max: float = 50.0       # スピードボーナスの最大値
    waypoint_time_limit: float = 15.0   # [s] スピードボーナスが0になる時間
    near_points_per_s: float = 10.0     # 接近ポイントの最大値 [点/秒]
    near_scale: float = 0.5             # [m] 接近ポイントの減衰距離
    finish_bonus_max: float = 100.0     # 完走ボーナスの最大値
    fail_penalty: float = -150.0        # 失敗ペナルティ


@dataclass(frozen=True)
class FailureRules:
    """失敗条件。既定値は train.py の env_cfg(AIの学習時の終了条件)と同じ"""
    min_height: float = 0.1       # [m] これより低いと墜落(termination_if_close_to_ground)
    max_abs_dx: float = 3.0       # [m] 目標からの水平ずれ(termination_if_x_greater_than)
    max_abs_dy: float = 3.0       # [m] (termination_if_y_greater_than)
    max_abs_dz: float = 2.0       # [m] 目標からの高さずれ(termination_if_z_greater_than)
    max_tilt_deg: float = 60.0    # [deg] ロール/ピッチの傾き(termination_if_roll/pitch_greater_than)
    max_ang_speed: float = 30.0   # [rad/s] 角速度のノルム(termination_if_ang_vel_greater_than)

    @classmethod
    def from_env_cfg(cls, env_cfg: dict) -> "FailureRules":
        """学習時の env_cfg(cfgs.pklから読み込んだもの)から、同じ失敗条件を作る"""
        defaults = cls()
        return cls(
            min_height=env_cfg.get("termination_if_close_to_ground", defaults.min_height),
            max_abs_dx=env_cfg.get("termination_if_x_greater_than", defaults.max_abs_dx),
            max_abs_dy=env_cfg.get("termination_if_y_greater_than", defaults.max_abs_dy),
            max_abs_dz=env_cfg.get("termination_if_z_greater_than", defaults.max_abs_dz),
            max_tilt_deg=min(
                env_cfg.get("termination_if_roll_greater_than", defaults.max_tilt_deg),
                env_cfg.get("termination_if_pitch_greater_than", defaults.max_tilt_deg),
            ),
            max_ang_speed=env_cfg.get("termination_if_ang_vel_greater_than") or defaults.max_ang_speed,
        )

    def check(self, pos, target, roll_deg: float, pitch_deg: float, ang_speed: float) -> Optional[str]:
        """失敗していればその理由(日本語)、していなければNoneを返す"""
        if pos[2] < self.min_height:
            return "墜落(高度が低すぎる)"
        rel = np.asarray(pos, dtype=float) - np.asarray(target, dtype=float)
        if abs(rel[0]) > self.max_abs_dx or abs(rel[1]) > self.max_abs_dy or abs(rel[2]) > self.max_abs_dz:
            return "範囲外(目標から離れすぎ)"
        if abs(roll_deg) > self.max_tilt_deg or abs(pitch_deg) > self.max_tilt_deg:
            return "転倒(傾きすぎ)"
        if ang_speed > self.max_ang_speed:
            return "回転暴走"
        return None


class ContestantScore:
    """1機分の採点状態"""

    RUNNING = "走行中"
    FINISHED = "完走"
    FAILED = "失敗"
    TIMEOUT = "時間切れ"

    def __init__(self, waypoints_abs, time_limit: float, cfg: ScoringConfig = ScoringConfig(),
                 rules: FailureRules = FailureRules()):
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints_abs]
        self.time_limit = float(time_limit)
        self.cfg = cfg
        self.rules = rules

        self.status = self.RUNNING
        self.fail_reason = ""
        self.wp_index = 0
        self.wp_started_at = 0.0
        self.captured_times: List[float] = []  # 各目標に到達した時刻

        self.pts_capture = 0.0
        self.pts_speed = 0.0
        self.pts_near = 0.0
        self.pts_finish = 0.0
        self.pts_penalty = 0.0

        self._err_sum = 0.0   # 距離の時間積分(平均誤差の計算用)
        self._err_time = 0.0
        self.end_time: Optional[float] = None  # 競技が終わった時刻(完走・失敗・時間切れ)

    # ---- 参照用 ---- #
    @property
    def running(self) -> bool:
        return self.status == self.RUNNING

    @property
    def current_target(self):
        """現在の目標(完走後は最後の目標)"""
        return self.waypoints[min(self.wp_index, len(self.waypoints) - 1)]

    @property
    def total(self) -> float:
        return self.pts_capture + self.pts_speed + self.pts_near + self.pts_finish + self.pts_penalty

    @property
    def captured(self) -> int:
        return len(self.captured_times)

    @property
    def mean_error(self) -> float:
        """走行中の目標までの平均距離[m]。小さいほど正確"""
        return self._err_sum / self._err_time if self._err_time > 0 else float("nan")

    # ---- 毎ステップの更新 ---- #
    def update(self, t: float, dt: float, pos, roll_deg: float, pitch_deg: float, ang_speed: float) -> List[str]:
        """1ステップ分の採点。t は「このステップ終了後」の経過時刻。発生したイベント(日本語)を返す"""
        if not self.running:
            return []
        events: List[str] = []
        target = self.current_target

        reason = self.rules.check(pos, target, roll_deg, pitch_deg, ang_speed)
        if reason is not None:
            self.status = self.FAILED
            self.fail_reason = reason
            self.pts_penalty += self.cfg.fail_penalty
            self.end_time = t
            return [f"失敗: {reason}"]

        dist = float(np.linalg.norm(np.asarray(pos, dtype=float) - target))
        self.pts_near += self.cfg.near_points_per_s * math.exp(-dist / self.cfg.near_scale) * dt
        self._err_sum += dist * dt
        self._err_time += dt

        if dist <= self.cfg.capture_radius:
            elapsed = t - self.wp_started_at
            speed_bonus = self.cfg.speed_bonus_max * max(0.0, 1.0 - elapsed / self.cfg.waypoint_time_limit)
            self.pts_capture += self.cfg.waypoint_points
            self.pts_speed += speed_bonus
            self.captured_times.append(t)
            events.append(f"目標{self.wp_index + 1}に到達({elapsed:.1f}秒、スピード+{speed_bonus:.0f})")
            self.wp_index += 1
            self.wp_started_at = t
            if self.wp_index >= len(self.waypoints):
                remaining = max(0.0, self.time_limit - t)
                self.pts_finish += self.cfg.finish_bonus_max * remaining / self.time_limit
                self.status = self.FINISHED
                self.end_time = t
                events.append(f"完走!(完走ボーナス+{self.pts_finish:.0f})")
        return events

    def time_up(self, t: float) -> None:
        """制限時間に達したら、走行中のものを時間切れにする"""
        if self.running and t >= self.time_limit:
            self.status = self.TIMEOUT
            self.end_time = t

    def summary(self) -> dict:
        return {
            "total": round(self.total, 1),
            "status": self.status,
            "fail_reason": self.fail_reason,
            "captured": self.captured,
            "waypoints": len(self.waypoints),
            "end_time": None if self.end_time is None else round(self.end_time, 2),
            "mean_error_m": None if math.isnan(self.mean_error) else round(self.mean_error, 3),
            "pts_capture": round(self.pts_capture, 1),
            "pts_speed": round(self.pts_speed, 1),
            "pts_near": round(self.pts_near, 1),
            "pts_finish": round(self.pts_finish, 1),
            "pts_penalty": round(self.pts_penalty, 1),
        }


def rank(scores: dict) -> list:
    """{名前: ContestantScore} を、順位順の [(名前, ContestantScore), ...] にする。
    合計点が高い順。同点なら、競技終了(完走)が早い方を上位にする。"""
    def sort_key(item):
        _, sc = item
        end = sc.end_time if sc.end_time is not None else float("inf")
        return (-sc.total, end)
    return sorted(scores.items(), key=sort_key)


def generate_waypoints(seed: int, n: int, min_separation: float = 0.8, extent: float = 1.0,
                       height: float = 1.0) -> np.ndarray:
    """レーン原点(開始位置の真下)からの相対座標で、n個の目標を作る。同じseedなら常に同じ。

    水平範囲は±extent[m](AIの学習時の command_cfg の pos_x/y_range と同じ ±1.0)、高さは
    学習時と同じ固定値(1.0 m)。連続する目標は min_separation[m] 以上離す
    (開始位置(0,0)からの最初の目標も同様)。"""
    rng = np.random.default_rng(seed)
    waypoints = []
    prev = np.zeros(2)
    while len(waypoints) < n:
        p = rng.uniform(-extent, extent, 2)
        if np.linalg.norm(p - prev) >= min_separation:
            waypoints.append([p[0], p[1], height])
            prev = p
    return np.array(waypoints)
