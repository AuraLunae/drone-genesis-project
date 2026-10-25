"""
test_arena_scoring.py

arena_scoring.py(AI・PID・人間の比較用の採点ロジック)の単体テスト。Genesisもtorchも不要。
採点の重みや失敗条件を変えた時に、意図通りに動くかを確認するために使う。

  uv run python test_arena_scoring.py
"""
import math, sys
import numpy as np
sys.path.insert(0, ".")
from arena_scoring import ContestantScore, ScoringConfig, FailureRules, rank, generate_waypoints

DT = 0.01
def fly(score, path_fn, duration, hz_dt=DT):
    """path_fn(t)->pos。duration秒分、1ステップずつ採点する"""
    events = []
    t = 0.0
    while t < duration and score.running:
        t += hz_dt
        pos = path_fn(t)
        events += score.update(t, hz_dt, pos, 0.0, 0.0, 0.0)
        score.time_up(t)
    return events, t

# --- 1. 目標生成: 再現性・距離・範囲 ---
a = generate_waypoints(7, 4); b = generate_waypoints(7, 4); c = generate_waypoints(8, 4)
assert np.array_equal(a, b) and not np.array_equal(a, c), "seed再現性"
assert np.all(np.abs(a[:, :2]) <= 1.0) and np.all(a[:, 2] == 1.0)
prev = np.zeros(2)
for w in a:
    assert np.linalg.norm(w[:2] - prev) >= 0.8 - 1e-9; prev = w[:2]
print("1. 目標生成 OK")

# --- 2. 1つの目標へ一定速度で近づく(3秒で到達) ---
wps = [[0.5, 0.0, 1.0]]
sc = ContestantScore(wps, time_limit=30)
start = np.array([0.0, 0.0, 1.0]); goal = np.array(wps[0])
events, t = fly(sc, lambda t: start + (goal - start) * min(t / 3.0, 1.0), 10)
assert sc.status == sc.FINISHED, sc.status
# 半径0.25に入るのは、距離0.5 -> 0.25 になる t=1.5s
assert 1.4 < sc.captured_times[0] < 1.6, sc.captured_times
cfg = ScoringConfig()
exp_speed = cfg.speed_bonus_max * (1 - sc.captured_times[0] / cfg.waypoint_time_limit)
assert abs(sc.pts_speed - exp_speed) < 0.5, (sc.pts_speed, exp_speed)
assert sc.pts_capture == 100.0
assert abs(sc.pts_finish - 100.0 * (30 - sc.end_time) / 30) < 0.01
print(f"2. 到達 OK  到達時刻={sc.captured_times[0]:.2f}s 合計={sc.total:.1f} {sc.summary()}")

# --- 3. 近いほど接近ポイントが高い ---
near = ContestantScore([[0, 0, 1.0], [5, 5, 1.0]], 30)   # 到達しても次の目標は遠い
far  = ContestantScore([[2.0, 0, 1.0]], 30)
fly(near, lambda t: np.array([0.3, 0, 1.0]), 1.0)   # 半径0.25の外、距離0.3で1秒
fly(far,  lambda t: np.array([0.0, 0, 1.0]), 1.0)   # 距離2.0で1秒
assert near.pts_near > 5 * far.pts_near, (near.pts_near, far.pts_near)
print(f"3. 接近ポイント OK  近い={near.pts_near:.2f} 遠い={far.pts_near:.2f}")

# --- 4. 失敗条件(各理由)と、失敗後は採点が止まる ---
rules = FailureRules()
cases = {
    "墜落": (np.array([0, 0, 0.05]), 0, 0, 0),
    "範囲外": (np.array([4.0, 0, 1.0]), 0, 0, 0),
    "範囲外z": (np.array([0, 0, 3.5]), 0, 0, 0),
    "転倒": (np.array([0, 0, 1.0]), 70, 0, 0),
    "回転": (np.array([0, 0, 1.0]), 0, 0, 40.0),
}
for name, (pos, r, p, w) in cases.items():
    s = ContestantScore([[0, 0, 1.0]], 30, rules=rules)
    ev = s.update(0.5, DT, pos, r, p, w)
    assert s.status == s.FAILED and s.pts_penalty == -150.0 and ev and s.fail_reason, name
    before = s.total
    s.update(0.51, DT, np.array([0, 0, 1.0]), 0, 0, 0)  # 失敗後は何も変わらない
    assert s.total == before and s.end_time == 0.5
print("4. 失敗条件 OK")

# --- 5. 時間切れ ---
s = ContestantScore([[3, 3, 1.0]], time_limit=2.0)
fly(s, lambda t: np.array([0, 0, 1.0]), 5.0)
assert s.status == s.TIMEOUT and s.captured == 0 and s.pts_penalty == 0
print("5. 時間切れ OK")

# --- 6. env_cfgからの失敗条件(train.pyの値)と、キー欠落への耐性 ---
cfg_new = {"termination_if_close_to_ground": 0.1, "termination_if_x_greater_than": 3.0,
           "termination_if_y_greater_than": 3.0, "termination_if_z_greater_than": 2.0,
           "termination_if_roll_greater_than": 60, "termination_if_pitch_greater_than": 60,
           "termination_if_ang_vel_greater_than": 30.0}
assert FailureRules.from_env_cfg(cfg_new) == FailureRules()
old_cfg = {k: v for k, v in cfg_new.items() if k != "termination_if_ang_vel_greater_than"}
assert FailureRules.from_env_cfg(old_cfg).max_ang_speed == 30.0     # 古いcfgs.pkl
assert FailureRules.from_env_cfg({**cfg_new, "termination_if_ang_vel_greater_than": None}).max_ang_speed == 30.0
print("6. env_cfg変換 OK")

# --- 7. 順位: 点数順、同点は完走が早い方 ---
def done(total_extra, end):
    s = ContestantScore([[0, 0, 1.0]], 30); s.pts_capture = total_extra; s.end_time = end; return s
r = rank({"A": done(100, 10), "B": done(150, 20), "C": done(100, 5)})
assert [n for n, _ in r] == ["B", "C", "A"], [n for n, _ in r]
print("7. 順位 OK")

# --- 8. 複数目標を順に到達し、完走ボーナスは残り時間比例 ---
wps = [[0.4, 0, 1.0], [0.4, 0.8, 1.0], [-0.4, 0.8, 1.0]]
def tour(t):
    pts = [np.array([0, 0, 1.0])] + [np.array(w) for w in wps]
    seg = min(int(t // 2.0), 2); frac = min((t - seg * 2.0) / 2.0, 1.0)
    return pts[seg] + (pts[seg + 1] - pts[seg]) * frac
s = ContestantScore(wps, 30)
ev, t = fly(s, tour, 10)
assert s.status == s.FINISHED and s.captured == 3, (s.status, s.captured)
assert abs(s.pts_finish - 100 * (30 - s.end_time) / 30) < 0.01
print(f"8. 複数目標 OK  到達時刻={[round(x,2) for x in s.captured_times]} 合計={s.total:.1f}")
print("\n全テスト合格")
