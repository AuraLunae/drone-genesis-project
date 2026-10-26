# この会話で行った変更のまとめ

対象: アップロードされた `9.md`(ドローン自律飛行プロジェクト完全マニュアル)に載っていたコード。
基準: 最初にアップロードされた `9.md` のコード → この会話でお渡しした最終版のコード。

**重要: 実際のGenesis上では、どの変更も動かしていません。** この作業環境にはGenesisもtorchも無く、確認できたのは構文チェックと、偽のGenesis/torchを使った単体テスト(旗)、および数値の試算(報酬)だけです。

---

## 1. 変更の一覧(依頼の順)

### 1-1. 旗をGenesisの粒子ソルバー(PBD布)にする / `--time`

| 項目 | 内容 |
|---|---|
| `eval.py --time` | 変更なし。アップロード時点で既に `-t/--time`(秒、既定60)だった |
| 旗の実体 | 剛体の板(蝶番、毎フレーム姿勢を書き込む)から、Genesisの粒子ソルバーPBDの布(`gs.materials.PBD.Cloth`)に変更。SPHは液体用、MPMは形を保つ薄い布に向かないため、布を表すPBDを選んだ(依頼は「SPHなど」だったが、PBDも粒子法) |
| 風の反映 | 布の粒子の速度に、風下向きの加速度を毎ステップ足す(風の強さに比例、減衰と速度上限つき、強風時は波打ち)。`genesis`の粒子API名は版で違う可能性があるため、候補名を順に試すアダプタ(`_Cloth`)に集約。見つからなければ旗への風の反映だけを止めて警告 |
| 衝突判定なし | 支柱は `collision=False`。布は `gs.Scene` に `LegacyCouplerOptions(rigid_pbd=False)` を渡して剛体との連成を切る |
| 呼び出し順 | `gs.Scene(..., **WindFlag.scene_kwargs())` → `WindFlag(...)` → `scene.build()` → `flag.post_build()` → 毎ステップ `flag.update(wind, dt)` |
| テスト | `test_wind_flag.py` を新規追加(偽のGenesis/torchで実行。Genesis不要) |
| 変更ファイル | `wind_flag.py`(全面書き換え)、`wind_hover_env.py`、`compare_arena.py`、`test_wind_flag.py`(新規)、`9.md` |

### 1-2. ドローンを大きく映す / 床を若干反射させる

| 項目 | 内容 |
|---|---|
| eval.py の視点 | `(4.5, 0, 2.2)`→`(-1, 0, 1.3)`、視野50度 から、`(2.2, 0, 1.5)`→`(-1, 0, 1.5)`、視野45度 へ。機体の見かけの大きさは約2.5倍(計算値) |
| アリーナの視点 | `(0, -8, 2.5)`→`(0, 0, 1)` から、`(0, -6, 2)`→`(0, 0, 1.2)` へ |
| 床の反射 | `genesis_compat.py` に `reflective_vis_options()`(`VisOptions(plane_reflection=True)`)、`add_floor()`(床の `roughness=FLOOR_ROUGHNESS=0.55`)、定数 `FLOOR_ROUGHNESS` を追加。引数が通らない版では警告して通常の床にする。見た目だけで物理には影響しない |
| 適用範囲 | アリーナと、`WindHoverEnv` でビューアを出す時(旗を出す時)。学習時は反射なし |
| 変更ファイル | `genesis_compat.py`、`wind_hover_env.py`、`compare_arena.py` |

### 1-3. 旗を奥へ / `eval.py --touch`(風なしのタッチチャレンジ)

| 項目 | 内容 |
|---|---|
| 旗の位置 | eval.pyでは `(-4.5, 0)`・`scale 1.2`(機体から約6.7 m奥。遠近法で小さく見える)。`WindHoverEnv` の既定値も同じ。アリーナは元のまま `(0, 3.5)`・`scale 1.6` |
| `--touch` | 風を0(`strength_range`・`sigma_range` を0)にし、ゴールに触れたら次のゴールが出るルールにする。タッチ回数・前のタッチからの間隔・墜落回数・終了時の集計を表示する `TouchCounter` クラスを `eval.py` に追加 |
| タッチの数え方 | `env.step()` の前後でゴール(`env.commands`)が変わったらタッチ。墜落などのリセット(`dones`)で変わった場合は数えず、墜落として別に数える |
| 診断 | 「ゴールの0.2 m以内で、触れずにいた時間」を表示。実行時間の30%を超えたら警告 |
| 変更ファイル | `eval.py`、`wind_hover_env.py` |

### 1-4. 「ゴール手前で待つ方が得では?」の指摘への対応(途中案は撤回)

指摘を受けて報酬を計算したところ、旧環境(触れた瞬間にゴールが飛ぶ)では、ゴールの0.1 m手前で待つ方が報酬上は得になる計算だった。

| 事実 | 内容 |
|---|---|
| 旧環境のゴールの動き | 機体が0.1 m以内に入ると、同じステップでゴールが飛ぶ。`resampling_time_s`(3秒)は環境のコードで使われていなかった |
| `at_target_bonus` | 「ゴール内に留まる」加点のつもりが、触れた瞬間にゴールが飛ぶので1ステップ限りの加点にしかならなかった |
| 試算(1ステップあたり) | 手前で待つ≒0.009、触って次へ向かう≒0.006(次のゴールまで平均約1.04 m、1回1〜2秒と仮定、角速度報酬を1と仮定した概算)。1回あたり約0.3〜0.6点、待機が得 |

途中で `touch_bonus` 報酬と `train.py --touch` を追加したが、次の1-5でコンセプトに合わせて**撤回**した(最終版には残っていない)。

### 1-5. 既定を「ゴールの地点で滞空し続ける」に戻し、タッチは共通モデルの評価オプションにする

| 項目 | 内容 |
|---|---|
| `env_cfg["goal_mode"]` | `"hover"`(既定): ゴールは `resampling_time_s`(3秒)ごとに変わる。触れても変わらない。 `"touch"`: 機体が `at_target_threshold`(0.1 m)以内に入った瞬間に変わる(`eval.py --touch` が指定)。`train.py` の `env_cfg` に `"goal_mode": "hover"` を追加 |
| モデルの共通化 | 観測は「ゴールまでの相対位置」などで、モードを区別しない。学習は `python train.py` の1本、モデルは1つ。`--touch` は評価時のルールの切り替えだけ |
| ゴールが飛んだ直後の処理 | ゴールを引き直した環境は、相対位置(`rel_pos`)と前ステップの相対位置(`last_rel_pos`)を新しいゴールで更新する。以前は観測が1ステップ古く、ゴールが飛んだ分が進歩報酬に混ざっていた。**これは学習の挙動を(わずかに)変える変更** |
| 診断メッセージ | `--touch` の待機警告を、「位置の精度が足りず、タッチの半径(0.1 m)に入れていない可能性」に書き換え |
| 変更ファイル | `wind_hover_env.py`、`train.py`、`eval.py` |

### 1-6. 質問への回答(コード変更なし)

- 「学習は共通化してあるか」→ はい(学習は1本、モデルは1つ)。ただし、学習(`hover`)と `--touch` 評価では、ゴールが飛ぶ時の機体の状態(滞空中か飛行中か)と風(あり/なし)が違うため、`--touch` でうまく動くかは未確認。悪ければ、学習の一部の環境だけ `touch` ルールにして混ぜる案がある。

---

## 2. 私の誤りと訂正

| 場面 | 誤り・訂正 |
|---|---|
| 1-3 | 「学習時から『タッチしたら次のゴール』を解いているので `eval.py` だけで足りる」「常に次のゴールへ向かって飛んでいたはず」と述べたが、コードを読んだだけの楽観的な判断だった。報酬を計算すると、旧環境ではゴール手前で待つ方が得になり得るため、撤回した |
| 1-4 | 最初の試算(触って次へ向かう≒0.0043/ステップ)は概算が雑だった。数値を実際に計算し直して≒0.006/ステップに訂正した(待機が得という結論は変わらない) |

---

## 3. 統合の試み(お渡しなし・中止)

別系統で作られた新しいマニュアル(旗の種類を `--flag-style` で選べる `surface`/`cloth-mesh`/`cloth`/`plate`、`--pbd`、`--no-wind-flag` など)との差分を取って統合する作業を、作業環境の中で途中まで進めましたが、中止しました。**お渡ししたファイルには含まれていません。**

- お渡し済みのコードの旗は、1-1の「PBD布+粒子速度への風の注入」方式で、`WindFlag` クラス、`scene_kwargs()`、`post_build()`、`along` 引数を使います。
- 新しい版の旗は `create_wind_flag()`、`wind_flag_scene_kwargs()`、`after_build()`、`--flag-style` を使います。**両者はAPIが違うため、ファイルを混ぜると `ImportError` などで動きません**。どちらかの系統にそろえてください。
- 新しい版の旗に、1-2・1-3・1-5の変更(視点・床の反射・旗の位置・`--touch`・`goal_mode`)を載せ直すには、`wind_hover_env.py`・`eval.py`・`compare_arena.py` の旗の呼び出し部分を、新しいAPIに書き換える必要があります。

---

## 4. 未確認事項(実際のGenesisで確認が必要)

1. PBD布: `gs.materials.PBD.Cloth()`、`gs.options.PBDOptions(lower_bound, upper_bound, particle_size, max_stretch_solver_iterations)`、`gs.options.LegacyCouplerOptions(rigid_pbd=False)` の引数が1.4.xで通るか(通らなければ警告して既定値で続行)。
2. 布の粒子API: `find_closest_particle`、`fix_particle`、`get_particles_pos`、`get_particles_vel`、`set_particles_vel` が実際にその名前か(見つからなければ旗は垂れ下がったままで、警告が1回出る)。
3. バッチ環境(`scene.build(n_envs=1)`)での粒子テンソルの形状。
4. PBDソルバーを足したことで、ドローンの物理や学習の結果が変わらないか。
5. 風に対する布の動きの強さ(`WIND_ACCEL=25`、`DAMPING_PER_S=1.5`、`FLUTTER_ACCEL=14`)は暫定値。起動直後に布が垂れ下がるまで1〜2秒揺れる。
6. 床の反射の強さ(`FLOOR_ROUGHNESS=0.55`)。`plane_reflection` や `roughness` を、Genesisがどう扱うか。
7. 視点の数値は計算上の値。ターゲットが画面外に出ないか、旗が画面に収まるか。
8. `goal_mode="hover"` の学習が安定するか、`resampling_time_s=3.0` が滞空の学習として適切か。
9. 共通モデルが `--touch` で0.1 mの半径に入れるか。
10. 今あるモデルは旧環境(触れるとゴールが飛ぶ)で学習したもの。コンセプト(滞空)どおりのモデルにするには `python train.py` で学習し直す。`--resume` はルールが変わるので注意。

---

## 5. ファイル別の変更量

| ファイル | 追加行 | 削除行 | 内容 |
|---|---|---|---|
| `wind_flag.py` | 231 | 73 | 剛体板 → PBD布(粒子速度への風の注入) |
| `wind_hover_env.py` | 30 | 10 | 旗のAPI、カメラ、床の反射、`goal_mode`、`rel_pos` の更新 |
| `compare_arena.py` | 9 | 6 | 旗のAPI、カメラ、床の反射 |
| `genesis_compat.py` | 26 | 0 | 床の反射の関数を追加 |
| `train.py` | 4 | 0 | `"goal_mode": "hover"` を追加 |
| `eval.py` | 77 | 4 | `--touch`、`TouchCounter`、旗の位置 |
| `test_wind_flag.py` | (新規) | | 旗の単体テスト |

---

## 6. 差分(元のコード → 変更後)

### 6-1 `wind_hover_env.py`

```diff
--- 元(アップロードされた9.md)/wind_hover_env.py
+++ 変更後/wind_hover_env.py
@@ -35,7 +35,7 @@
     transform_quat_by_quat,
 )
 
-from genesis_compat import PropellerVisual, apply_com_force
+from genesis_compat import PropellerVisual, apply_com_force, add_floor, reflective_vis_options
 from wind_flag import WindFlag
 
 
@@ -69,7 +69,7 @@
         # 旗を出す時だけ、低めに引いた視点に切り替える。
         show_flag = bool(env_cfg.get("visualize_wind_flag", False))
         viewer_camera = (
-            dict(camera_pos=(4.5, 0.0, 2.2), camera_lookat=(-1.0, 0.0, 1.3), camera_fov=50)
+            dict(camera_pos=(2.2, 0.0, 1.5), camera_lookat=(-1.0, 0.0, 1.5), camera_fov=45)
             if show_flag
             else dict(camera_pos=(3.0, 0.0, 3.0), camera_lookat=(0.0, 0.0, 1.0), camera_fov=40)
         )
@@ -82,14 +82,17 @@
                 enable_joint_limit=True,
                 constraint_solver=gs.constraint_solver.Newton,
             ),
-            vis_options=gs.options.VisOptions(
+            # 旗を出す時(評価)だけ、床を若干反射させる(見た目だけ。物理には影響しない)
+            vis_options=(reflective_vis_options if show_flag else gs.options.VisOptions)(
                 rendered_envs_idx=list(range(self.rendered_env_num)),
             ),
             viewer_options=gs.options.ViewerOptions(**viewer_camera),
             show_viewer=show_viewer,
+            # 旗(PBDの布)を出す時だけ、PBDソルバーと「布は何とも衝突しない」設定を追加する
+            **(WindFlag.scene_kwargs() if show_flag else {}),
         )
 
-        self.scene.add_entity(gs.morphs.Plane())
+        add_floor(self.scene, reflective=show_flag)
 
         if self.env_cfg["visualize_target"]:
             self.target = self.scene.add_entity(
@@ -108,7 +111,7 @@
 
         if self.env_cfg["visualize_camera"]:
             record_camera = (
-                dict(pos=(4.5, 0.0, 2.2), lookat=(-1.0, 0.0, 1.3), fov=50)
+                dict(pos=(2.2, 0.0, 1.5), lookat=(-1.0, 0.0, 1.5), fov=45)
                 if show_flag
                 else dict(pos=(3.5, 0.0, 2.5), lookat=(0, 0, 0.5), fov=30)
             )
@@ -122,19 +125,22 @@
             morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf")
         )
 
-        # 風の強さと向きを映す白い旗(衝突判定なし)。ドローンの後方(カメラの反対側)に立てる。
+        # 風の強さと向きを映す白い旗(PBDの布・衝突判定なし)。ドローンの後方(カメラの反対側)に立てる。
         self.wind_flag = (
             WindFlag(
                 self.scene,
-                base_xy=env_cfg.get("wind_flag_xy", (-2.8, 0.0)),
-                scale=env_cfg.get("wind_flag_scale", 1.3),
+                base_xy=env_cfg.get("wind_flag_xy", (-4.5, 0.0)),
+                scale=env_cfg.get("wind_flag_scale", 1.2),
                 n_envs=num_envs,
+                along=(0.0, 1.0),   # 視点は+x側から-x方向を見る。旗の広い面が見えるよう、最初はy方向へ伸ばす
             )
             if show_flag
             else None
         )
 
         self.scene.build(n_envs=num_envs)
+        if self.wind_flag is not None:
+            self.wind_flag.post_build()  # 布の根元を支柱に固定する(build後にしかできない)
 
         # ドローンのCOMリンク(風・外乱力を加える対象)
         # 注: Genesis 1.2.3 では DroneEntity.COM_link_idx が未初期化(内部属性
@@ -263,8 +269,22 @@
         self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
         self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)
 
-        envs_idx = self._at_target()
-        self._resample_commands(envs_idx)
+        # ゴールの引き直し。goal_mode で2通り(同じモデルで両方動かせる):
+        #   "hover"(既定): ゴールは一定時間(resampling_time_s)ごとに変わる。触れても変わらない
+        #                   → ゴールの位置で滞空し続けるほど報酬が増える(本来のコンセプト)
+        #   "touch"       : ゴールに触れた(at_target_threshold以内)瞬間に変わる
+        #                   → 触れて次のゴールへ、を繰り返す(eval.py --touch)
+        if self.env_cfg.get("goal_mode", "hover") == "touch":
+            envs_idx = self._at_target()
+        else:
+            interval = max(1, int(round(self.env_cfg["resampling_time_s"] / self.dt)))
+            envs_idx = (self.episode_length_buf % interval == 0).nonzero(as_tuple=False).reshape((-1,))
+        if len(envs_idx) > 0:
+            self._resample_commands(envs_idx)
+            # 新しいゴールに対する相対位置に更新する(古いゴールのままだと、観測が1ステップ古く、
+            # ゴールが飛んだ分が進歩報酬に混ざる)。ゴールが飛んだステップの進歩報酬は0にする
+            self.rel_pos[envs_idx] = self.commands[envs_idx] - self.base_pos[envs_idx]
+            self.last_rel_pos[envs_idx] = self.rel_pos[envs_idx]
 
         self.crash_condition = (
             (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
```

### 6-2 `compare_arena.py`

```diff
--- 元(アップロードされた9.md)/compare_arena.py
+++ 変更後/compare_arena.py
@@ -55,7 +55,7 @@
 from genesis.vis.keybindings import Key, KeyAction, Keybind
 
 from arena_scoring import ContestantScore, FailureRules, ScoringConfig, generate_waypoints, rank
-from genesis_compat import PropellerVisual, apply_com_force
+from genesis_compat import PropellerVisual, apply_com_force, add_floor, reflective_vis_options
 from manual_pid_fly import (
     BASE_RPM,
     DT,
@@ -345,17 +345,18 @@
             enable_joint_limit=True,
             constraint_solver=gs.constraint_solver.Newton,
         ),
-        vis_options=gs.options.VisOptions(show_world_frame=False),
+        vis_options=reflective_vis_options(show_world_frame=False),   # 床を若干反射させる
         viewer_options=gs.options.ViewerOptions(
-            camera_pos=(0.0, -8.0, 2.5),
-            camera_lookat=(0.0, 0.0, 1.0),
+            camera_pos=(0.0, -6.0, 2.0),
+            camera_lookat=(0.0, 0.0, 1.2),
             camera_fov=50,
             realtime_factor=1.0,
             run_in_thread=False,
         ),
         show_viewer=True,
+        **WindFlag.scene_kwargs(),   # 旗(PBDの布)用のソルバーと、「布は何とも衝突しない」設定
     )
-    scene.add_entity(gs.morphs.Plane())
+    add_floor(scene)
 
     keys = (["AI"] if ai else []) + ["PID", "HUMAN"]
     drones, markers = {}, {}
@@ -367,8 +368,10 @@
             morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.06, fixed=False, collision=False),
             surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=MARKER_COLORS[key])),
         )
-    wind_flag = WindFlag(scene, base_xy=WIND_FLAG_XY, scale=WIND_FLAG_SCALE)
+    wind_flag = WindFlag(scene, base_xy=WIND_FLAG_XY, scale=WIND_FLAG_SCALE,
+                         along=(1.0, 0.0))   # カメラは-y側から+y方向を見る。旗の広い面が見えるよう、最初はx方向へ伸ばす
     scene.build()
+    wind_flag.post_build()   # 布の根元を支柱に固定する(build後にしかできない)
 
     contestants = [Contestant(k, drones[k], markers[k], not args.no_prop_fix) for k in keys]
     by_key = {c.key: c for c in contestants}
```

### 6-3 `genesis_compat.py`

```diff
--- 元(アップロードされた9.md)/genesis_compat.py
+++ 変更後/genesis_compat.py
@@ -93,3 +93,29 @@
         rpm_tensor = rpm_tensor.reshape(-1, 1) if rpm_tensor.ndim == 1 else rpm_tensor.T
         self._revs = (self._revs + rpm_tensor * self.VISUAL_SCALE) % self._wrap
         revs.copy_(self._revs)
+
+
+# ---------------------------------------------------------------------- #
+# 床の反射(評価・アリーナ用の見た目だけ。物理には影響しない)
+# ---------------------------------------------------------------------- #
+FLOOR_ROUGHNESS = 0.55  # 小さいほど鏡に近い。若干の反射にするため、やや大きめ(暫定値。実機で見て調整)
+
+
+def reflective_vis_options(**kwargs):
+    """床の反射を有効にした VisOptions を作る(plane_reflection が無い版では、無しで作る)。"""
+    try:
+        return gs.options.VisOptions(plane_reflection=True, **kwargs)
+    except TypeError:
+        print("[注意] この版の VisOptions には plane_reflection がないため、床の反射なしで続けます")
+        return gs.options.VisOptions(**kwargs)
+
+
+def add_floor(scene, reflective=True):
+    """床(Plane)を足す。reflective=True なら、若干つやのある床にする(通らなければ通常の床)。"""
+    if reflective:
+        try:
+            surface = gs.surfaces.Default(color=(0.92, 0.92, 0.94, 1.0), roughness=FLOOR_ROUGHNESS)
+            return scene.add_entity(gs.morphs.Plane(), surface=surface)
+        except (TypeError, AttributeError) as e:
+            print(f"[注意] 床の表面設定が通らないため、通常の床にします: {e}")
+    return scene.add_entity(gs.morphs.Plane())
```

### 6-4 `train.py`

```diff
--- 元(アップロードされた9.md)/train.py
+++ 変更後/train.py
@@ -93,6 +93,10 @@
         "base_init_quat": [1.0, 0.0, 0.0, 0.0],
         "episode_length_s": 15.0,
         "at_target_threshold": 0.1,
+        # ゴールを引き直す間隔[s]。ゴールの位置で滞空し続ける学習なので、触れてもゴールは変わらず、
+        # この時間ごとに変わる("hover")。"touch" にすると、触れた瞬間に変わる(eval.py --touch が指定する)。
+        # 同じモデルが両方で使える: どちらでも観測は「ゴールまでの相対位置」だけだから。
+        "goal_mode": "hover",
         "resampling_time_s": 3.0,
         "simulate_action_latency": True,
         "clip_actions": 1.0,
```

### 6-5 `eval.py`

```diff
--- 元(アップロードされた9.md)/eval.py
+++ 変更後/eval.py
@@ -9,6 +9,9 @@
   # 実行時間を変える(秒)
   python eval.py --time 120
 
+  # 風なしで、ゴールを次々にタッチしていく(タッチするたびに次のゴールが出る)
+  python eval.py --touch
+
   # 特定のランの特定チェックポイントを評価
   python eval.py -e wind-hovering --run 20261004_120000 --ckpt 300
 """
@@ -41,6 +44,9 @@
     parser.add_argument("-r", "--record", action="store_true", help="Record the rollout to video")
     parser.add_argument("-t", "--time", dest="duration", type=float, default=60.0,
                         help="実行時間[秒]。既定60秒(旧版は約9秒だった)。録画時もこの長さになる")
+    parser.add_argument("--touch", action="store_true",
+                        help="タッチチャレンジ(同じモデルで動く): 風なしで、ゴール(赤い球)にタッチすると次のゴールが出る。"
+                             "タッチ回数と間隔を表示する")
     args = parser.parse_args()
 
     gs.init(backend=gs.cpu)
@@ -54,10 +60,18 @@
         env_cfg, obs_cfg, reward_cfg, command_cfg, wind_cfg, train_cfg = pickle.load(f)
 
     reward_cfg["reward_scales"] = {}
+    if args.touch:
+        # 風なし: 風の強さ・乱れを0にする(旗は垂れ下がったままになる)
+        wind_cfg = dict(wind_cfg, strength_range=[0.0, 0.0], sigma_range=[0.0, 0.0])
+        # 触れた瞬間にゴールが変わるルールにする(学習時の既定は、時間でゴールが変わる"hover")。
+        # 観測は「ゴールまでの相対位置」だけなので、同じモデルがそのまま使える
+        env_cfg["goal_mode"] = "touch"
     env_cfg["visualize_target"] = True
     env_cfg["visualize_camera"] = args.record
     # 風の強さと向きが誰にでも分かるよう、後方に衝突判定のない白い旗を立てる(視点も旗が入る位置になる)
     env_cfg["visualize_wind_flag"] = True
+    env_cfg["wind_flag_xy"] = (-4.5, 0.0)   # 旗は奥に置く(遠近法で小さく見える)
+    env_cfg["wind_flag_scale"] = 1.2
     env_cfg["max_visualize_FPS"] = 60
     # 実行時間が学習時のエピソード長(15秒)より長い場合、途中でタイムアウトして初期位置に
     # 戻されないよう、エピソード長も実行時間に合わせて延ばす(墜落・範囲外では従来通りリセットされる)
@@ -83,17 +97,76 @@
     # 実行時間[秒]をシミュレーションのステップ数に変換する(1ステップ = env.dt = 0.01秒)
     max_sim_step = int(args.duration / env.dt)
 
+    touch = TouchCounter(env.dt) if args.touch else None
+    if touch:
+        print("=== タッチチャレンジ(風なし)。ゴールにタッチすると次のゴールが出ます ===")
+
+    def step_once(obs_dict):
+        before = env.commands[0].clone() if touch else None
+        actions = policy(obs_dict)
+        obs_dict, rews, dones, infos = env.step(actions)
+        if touch:
+            touch.observe(before, env.commands[0], bool(dones[0]), float(env.rel_pos[0].norm()))
+        return obs_dict
+
     with torch.no_grad():
         if args.record:
             env.cam.start_recording(save_to_filename="out/wind_hover.mp4", fps=env_cfg["max_visualize_FPS"])
             for _ in range(max_sim_step):
-                actions = policy(obs_dict)
-                obs_dict, rews, dones, infos = env.step(actions)
+                obs_dict = step_once(obs_dict)
             env.cam.stop_recording()
         else:
             for _ in range(max_sim_step):
-                actions = policy(obs_dict)
-                obs_dict, rews, dones, infos = env.step(actions)
+                obs_dict = step_once(obs_dict)
+    if touch:
+        touch.summary()
+
+
+class TouchCounter:
+    """タッチ回数の集計。env.step() の前後でゴールが変わったら「タッチした」とみなす。
+
+    WindHoverEnv は、機体がゴールの at_target_threshold(既定0.1 m)以内に入ると、同じステップで
+    ゴールを新しい位置に引き直す。そのため、ゴールが変わった = タッチ。
+    ただし、墜落などのリセット(dones)でもゴールは引き直されるので、その場合は数えない。
+    """
+
+    NEAR_M = 0.2   # この距離[m]以内で、タッチ(0.1 m)せずにいた時間を「待機」として数える
+
+    def __init__(self, dt):
+        self.dt = dt
+        self.near_steps = 0
+        self.steps = 0
+        self.last_touch_step = 0
+        self.count = 0
+        self.crashes = 0
+        self.intervals = []
+
+    def observe(self, before, after, done, dist=None):
+        self.steps += 1
+        if done:
+            self.crashes += 1
+            self.last_touch_step = self.steps
+            print(f"  [{self.steps * self.dt:6.1f}s] 墜落・範囲外でリセットされました")
+            return
+        if dist is not None and 0.1 <= dist < self.NEAR_M:
+            self.near_steps += 1
+        if bool((before != after).any()):
+            self.count += 1
+            interval = (self.steps - self.last_touch_step) * self.dt
+            self.last_touch_step = self.steps
+            self.intervals.append(interval)
+            print(f"  [{self.steps * self.dt:6.1f}s] タッチ {self.count} 回目 (前のタッチから {interval:.2f} 秒)")
+
+    def summary(self):
+        print("=== 結果 ===")
+        print(f"実行時間 {self.steps * self.dt:.1f} 秒 / タッチ {self.count} 回 / 墜落 {self.crashes} 回")
+        if self.intervals:
+            print(f"タッチの間隔: 平均 {sum(self.intervals) / len(self.intervals):.2f} 秒 / "
+                  f"最短 {min(self.intervals):.2f} 秒 / 最長 {max(self.intervals):.2f} 秒")
+        near = self.near_steps * self.dt
+        print(f"ゴールの{self.NEAR_M:.1f} m以内で、タッチせずにいた時間: {near:.1f} 秒")
+        if self.steps > 0 and near > 0.3 * self.steps * self.dt:
+            print("[注意] ゴール手前(0.1〜0.2 m)で待っている時間が長い。位置の精度が足りず、タッチの半径(0.1 m)に入れていない可能性がある")
 
 
 if __name__ == "__main__":
```

### 6-6 `wind_flag.py`

```diff
--- 元(アップロードされた9.md)/wind_flag.py
+++ 変更後/wind_flag.py
@@ -4,22 +4,41 @@
 風の強さと向きが、誰にでも一目で分かるようにするための、白い旗(衝突判定なし)。
 eval.py(wind_hover_env.py経由)と compare_arena.py の、後方(背景側)に立てる。
 
+旗の実体は、Genesisの粒子ベースのソルバーの1つ「PBD(Position Based Dynamics)」の布
+(gs.materials.PBD.Cloth)。布の頂点が粒子になっていて、伸び・曲げの拘束で布らしく動く。
+  * SPH は液体用、MPM は液体・砂・弾性体用の粒子ソルバーで、どちらも「布の形を保つ」ことが
+    できない(旗が液体のように崩れる)。Genesisで布を表すのは PBD なので、旗には PBD を使う。
+
 見方(実際の旗と同じ):
   - 旗が垂れ下がっている  : ほぼ無風
-  - 旗が斜めに持ち上がる  : 風がある。持ち上がるほど強い
-  - 旗がほぼ水平になる    : 強風(機体の重量の約半分以上の力)
+  - 旗が風下へ持ち上がる  : 風がある。持ち上がるほど強い
+  - 旗がほぼ横に流れる    : 強風(機体の重量の約半分以上の力)
   - 旗が向く方向          : 風下(風が吹いていく方向 = 機体が流される方向)
-  - 強い時は旗が細かくはためく
+  - 風が強い時は旗が波打つ
 
-旗はシミュレーションに加えている「実際の風の力」(OU過程の風)をそのまま映すので、
+旗はシミュレーションに加えている「実際の風の力」(OU過程の風)の強さと向きをそのまま映す。
 風が時間とともに強まったり弱まったりする様子(突風)も、そのまま旗の動きになる。
-旗も支柱も衝突判定が無く、ドローンが通り抜けても何も起きない。
 
-構造: 支柱の先端に、上端を蝶番で留めた長方形の板(旗)がぶら下がっている。
-      風の力に応じて、板が風下側へ持ち上がる(0度=垂れ下がり、90度=水平)。
+衝突判定なし:
+  - 支柱は Box(fixed=True, collision=False)(剛体側の衝突判定なし)
+  - 布は、剛体との連成(rigid_pbd)を scene_kwargs() で切っている。ドローンも地面も支柱も
+    すり抜ける。布が動かすのは布の粒子だけで、ドローンの物理には一切影響しない
+
+使い方(scene.build() の前後で2か所必要):
+    scene = gs.Scene(..., **WindFlag.scene_kwargs())   # 1. PBDソルバーと連成設定
+    flag = WindFlag(scene, base_xy=(x, y), scale=1.3)    # 2. build の前に生成
+    scene.build()
+    flag.post_build()                                    # 3. build の後(布の根元を支柱に固定する)
+    flag.update(wind_force, dt)                          # 4. 毎ステップ、scene.step() の前に呼ぶ
+
+Genesisの版による名前の違い(布の粒子の取得・設定のメソッド名など)は、_Cloth アダプタに
+まとめてある。見つからない時は、シミュレーション全体は止めずに、旗への風の反映だけを
+無効にして理由を1度だけ表示する。
 """
 
 import math
+import os
+import tempfile
 
 import numpy as np
 import torch
@@ -27,73 +46,143 @@
 import genesis as gs
 
 
-def matrix_to_quat(rotation):
-    """3x3の回転行列を、Genesisの並び(w, x, y, z)のクォータニオンに変換する"""
-    m = rotation
-    trace = m[0, 0] + m[1, 1] + m[2, 2]
-    if trace > 0.0:
-        s = math.sqrt(trace + 1.0) * 2.0
-        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
-    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
-        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
-        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
-    elif m[1, 1] > m[2, 2]:
-        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
-        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
-    else:
-        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
-        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
-    quat = np.array([w, x, y, z], dtype=np.float64)
-    return quat / np.linalg.norm(quat)
-
-
-def flag_pose(wind_vec, azimuth, level, hinge, flag_length, flutter_rad=0.0):
-    """旗の姿勢を計算する(Genesis非依存。単体テストできる)。
-
-    azimuth     : 風下の水平方向の単位ベクトル(長さ2)
-    level       : 風の強さ(0=無風 〜 1=旗が水平になる強さ)
-    hinge       : 旗の蝶番(支柱の先端)の位置(長さ3)
-    戻り値      : (旗の中心位置(長さ3), クォータニオン(w,x,y,z), 旗の長さ方向の単位ベクトル d)
+def write_cloth_obj(path, origin, along, height, length, nx, nz):
+    """鉛直な長方形の布(格子状の三角形メッシュ)をOBJファイルに書き出す(Genesis非依存)。
+
+    origin : 布の左上の角(支柱に留める縦の辺の上端)のワールド座標(長さ3)
+    along  : 支柱から布の先端へ向かう水平の単位ベクトル(長さ2)
+    戻り値 : 頂点座標 (nz+1, nx+1, 3) の配列(行 = 上から下、列 = 支柱側から先端側)
+    """
+    grid = np.zeros((nz + 1, nx + 1, 3))
+    for i in range(nz + 1):
+        for j in range(nx + 1):
+            s = length * j / nx
+            grid[i, j] = (origin[0] + along[0] * s, origin[1] + along[1] * s, origin[2] - height * i / nz)
+    with open(path, "w", encoding="utf-8") as f:
+        for i in range(nz + 1):
+            for j in range(nx + 1):
+                f.write("v {:.6f} {:.6f} {:.6f}\n".format(*grid[i, j]))
+        for i in range(nz):
+            for j in range(nx):
+                a = i * (nx + 1) + j + 1
+                b = a + 1
+                c = a + (nx + 1)
+                d = c + 1
+                f.write(f"f {a} {c} {b}\n")
+                f.write(f"f {b} {c} {d}\n")
+    return grid
+
+
+class _Cloth:
+    """PBD布エンティティの粒子操作(版による名前の違いを吸収する薄いアダプタ)。
+
+    Genesisの版によって、粒子の位置・速度を読み書きするメソッド名が違うことがある。
+    候補を順に試し、使えるものを1つ選ぶ。全部だめな時は ok=False にして理由を保持する。
     """
-    alpha = min(max(level * (math.pi / 2.0) + flutter_rad, 0.0), math.pi / 2.0)  # 0=垂れ下がり, 90度=水平
-    d = np.array([math.sin(alpha) * azimuth[0], math.sin(alpha) * azimuth[1], -math.cos(alpha)])
-    b = np.array([-azimuth[1], azimuth[0], 0.0])    # 旗の幅方向(水平で、風向きに垂直)
-    n = np.cross(b, d)                               # 旗の面の法線
-    rotation = np.column_stack([d, n, b])            # 旗の局所軸 x=長さ, y=厚み, z=幅 → ワールド軸
-    center = np.asarray(hinge, dtype=float) + d * (flag_length / 2.0)
-    return center, matrix_to_quat(rotation), d
+
+    POS_GETTERS = ("get_particles_pos", "get_particles_position", "get_particle_pos")
+    VEL_GETTERS = ("get_particles_vel", "get_particles_velocity", "get_particle_vel")
+    VEL_SETTERS = ("set_particles_vel", "set_particles_velocity", "set_particle_vel")
+
+    def __init__(self, entity):
+        self.entity = entity
+        self.get_pos = self._pick(self.POS_GETTERS)
+        self.get_vel = self._pick(self.VEL_GETTERS)
+        self.set_vel = self._pick(self.VEL_SETTERS)
+        missing = [n for n, f in (("粒子位置の取得", self.get_pos), ("粒子速度の取得", self.get_vel),
+                                  ("粒子速度の設定", self.set_vel)) if f is None]
+        self.ok = not missing
+        self.reason = "、".join(missing) + "のメソッドが見つからない" if missing else ""
+
+    def _pick(self, names):
+        for name in names:
+            fn = getattr(self.entity, name, None)
+            if callable(fn):
+                return fn
+        return None
+
+    @staticmethod
+    def to_numpy(x):
+        if hasattr(x, "detach"):
+            x = x.detach().cpu().numpy()
+        return np.asarray(x, dtype=np.float64)
 
 
 class WindFlag:
-    """風の強さと向きを映す白い旗。scene.build() の前に生成し、毎ステップ update() を呼ぶ。"""
+    """風の強さと向きを映す白い旗(PBDの布、衝突判定なし)。"""
 
-    FULL_SCALE_N = 0.13     # [N] この強さで旗が水平になる(機体重量0.265 Nの約半分)
-    SMOOTHING_S = 0.15      # [s] 旗の動きをなめらかにする時定数(風の細かい揺らぎで震えすぎないように)
-    FLUTTER_RAD = 0.10      # 強風時のはためきの振れ幅 [rad]
-    FLUTTER_HZ = 3.0        # はためきの周波数
+    # ---- 見た目と動きの調整値(実機で見ながら調整する暫定値) ----
+    FULL_SCALE_N = 0.13     # [N] この強さで旗が大きく横に流れる(機体重量0.265 Nの約半分)
+    SMOOTHING_S = 0.15      # [s] 風への反応をなめらかにする時定数(風の細かい揺らぎで震えすぎないように)
+    WIND_ACCEL = 25.0       # [m/s^2] 風が FULL_SCALE_N の時に、布の粒子へ加える風下向きの加速度
+                            #   重力(9.8)との釣り合いで、風の強さ0.5で約50度、1.0で約68度持ち上がる
+    DAMPING_PER_S = 1.5     # [1/s] 粒子速度の減衰(空気抵抗の代わり。ばたつきすぎを抑える)
+    MAX_SPEED = 8.0         # [m/s] 粒子速度の上限(物理の発散を防ぐ安全弁)
+    FLUTTER_ACCEL = 14.0    # [m/s^2] 強風時に、布を横方向へ波打たせる加速度の大きさ
+    FLUTTER_HZ = 2.5        # はためきの周波数
+    FLUTTER_WAVE = 9.0      # [rad/m] 波が支柱から先端へ進む時の、距離あたりの位相の進み
     CALM_N = 0.003          # [N] これ未満では、風の向きを更新しない(ほぼ無風で向きがばらつくのを防ぐ)
 
-    def __init__(self, scene, base_xy, scale=1.0, n_envs=0):
+    GRID_X = 20             # 布の分割数(支柱から先端方向)
+    GRID_Z = 12             # 布の分割数(縦方向)。粒子数は (GRID_X+1)×(GRID_Z+1) = 273
+
+    @staticmethod
+    def scene_kwargs():
+        """gs.Scene(...) に展開して渡す追加オプション(PBDソルバーと、布の衝突の無効化)。
+
+        * pbd_options : 布を動かす粒子ソルバー。範囲(lower/upper_bound)を、実際に使う範囲に絞る
+        * coupler_options : 剛体とPBDの連成(rigid_pbd)を切る = 布がドローン・地面・支柱と衝突しない
+        どちらも版によって引数が違う可能性があるため、通らなければ警告して既定値で続ける。
         """
-        scene   : 旗を置くシーン(build前)
+        kwargs = {}
+        try:
+            kwargs["pbd_options"] = gs.options.PBDOptions(
+                lower_bound=(-8.0, -8.0, 0.0),
+                upper_bound=(8.0, 8.0, 6.0),
+                particle_size=0.05,
+                max_stretch_solver_iterations=8,
+            )
+        except (AttributeError, TypeError) as e:
+            print(f"[注意] PBDOptions を指定できなかったため既定値で続けます: {e}")
+            try:
+                kwargs["pbd_options"] = gs.options.PBDOptions()
+            except Exception as e2:  # noqa: BLE001
+                print(f"[注意] PBDOptions が使えません(旗の布は出ない可能性があります): {e2}")
+        try:
+            kwargs["coupler_options"] = gs.options.LegacyCouplerOptions(rigid_pbd=False)
+        except (AttributeError, TypeError) as e:
+            print(f"[注意] 剛体と布の連成を切れませんでした。旗はドローンから十分離してあるため\n"
+                  f"       通常は触れませんが、衝突判定は完全には無効化できていません: {e}")
+        return kwargs
+
+    def __init__(self, scene, base_xy, scale=1.0, n_envs=0, along=(0.0, 1.0)):
+        """
+        scene   : 旗を置くシーン(build前。scene_kwargs() を渡して作ったもの)
         base_xy : 支柱の根元の水平位置 (x, y)
         scale   : 大きさの倍率(カメラから遠いほど大きくする)
-        n_envs  : scene.build(n_envs=...) に渡す環境数。単一環境(n_envsを渡さない)なら0
+        n_envs  : scene.build(n_envs=...) に渡す環境数(0なら単一環境)。粒子は全環境に同じものが入る
+        along   : 旗が最初に伸びている水平方向(カメラに旗の広い面が見えるよう、視線に垂直にする)
         """
         self.n_envs = n_envs
+        self.base_xy = np.asarray(base_xy, dtype=float)
         self.pole_height = 2.6 * scale
-        self.flag_length = 0.70 * scale
-        self.flag_width = 0.45 * scale
-        self.hinge = np.array([base_xy[0], base_xy[1], self.pole_height], dtype=float)
+        self.flag_length = 0.80 * scale
+        self.flag_height = 0.50 * scale
+
+        a = np.asarray(along, dtype=float)
+        self._along = a / np.linalg.norm(a)
 
         self._azimuth = np.array([1.0, 0.0])
         self._level = 0.0
         self._time = 0.0
-        self.strength = 0.0   # 直近の風の強さ [N](表示用)
+        self.strength = 0.0     # 直近の風の強さ [N](表示用)
+
+        self._cloth = None
+        self._ready = False
+        self._warned = False
 
         gray = gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.72, 0.72, 0.72)))
-        white = gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(1.0, 1.0, 1.0)))
-        # 衝突判定なし(collision=False)。支柱は固定、旗は毎ステップ姿勢を書き込む
+        # 支柱: 固定。剛体としても衝突判定なし
         scene.add_entity(
             morph=gs.morphs.Box(
                 pos=(base_xy[0], base_xy[1], self.pole_height / 2.0),
@@ -103,26 +192,67 @@
             ),
             surface=gray,
         )
-        self.flag = scene.add_entity(
-            morph=gs.morphs.Box(
-                pos=(base_xy[0], base_xy[1], self.pole_height - self.flag_length / 2.0),
-                size=(self.flag_length, 0.015 * scale, self.flag_width),
-                fixed=False,
-                collision=False,
-            ),
-            surface=white,
+
+        # 布: 支柱の上端から少し下を上辺に、支柱から少しだけ離れた縦の辺を根元にする
+        offset = 0.04 * scale
+        top_z = self.pole_height - 0.05 * scale
+        origin = (base_xy[0] + self._along[0] * offset, base_xy[1] + self._along[1] * offset, top_z)
+        obj_path = os.path.join(tempfile.gettempdir(), f"wind_flag_cloth_{os.getpid()}_{scale:.3f}.obj")
+        self._grid = write_cloth_obj(obj_path, origin, self._along, self.flag_height, self.flag_length,
+                                     self.GRID_X, self.GRID_Z)
+
+        try:
+            material = gs.materials.PBD.Cloth()
+        except Exception as e:  # noqa: BLE001
+            raise RuntimeError(f"PBD.Cloth を作れません(Genesisの版を確認してください): {e}") from e
+
+        try:
+            surface = gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0), vis_mode="visual", double_sided=True)
+        except TypeError:  # 古い版には double_sided が無い
+            surface = gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0), vis_mode="visual")
+
+        self.cloth_entity = scene.add_entity(
+            material=material,
+            morph=gs.morphs.Mesh(file=obj_path, scale=1.0, pos=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0)),
+            surface=surface,
         )
 
     @staticmethod
     def legend():
         """画面の見方(起動時に表示する)"""
         return (
-            "白い旗 = 風の見える化(衝突判定なし): 垂れ下がる=ほぼ無風 / 斜めに持ち上がる=風あり(高いほど強い) / "
-            f"水平=強風(約{WindFlag.FULL_SCALE_N:.2f} N以上。機体重量の約半分)。旗が向く方向 = 風下(機体が流される方向)"
+            "白い旗 = 風の見える化(衝突判定なし): 垂れ下がる=ほぼ無風 / 風下へ持ち上がる=風あり(高いほど強い) / \n"
+            f"  ほぼ横に流れて波打つ=強風(約{WindFlag.FULL_SCALE_N:.2f} N以上。機体重量の約半分)。旗が向く方向 = 風下(機体が流される方向)"
         )
 
+    # ------------------------------------------------------------------ #
+    def post_build(self):
+        """scene.build() の後に1度呼ぶ。布の根元(支柱側の縦の辺)の粒子を固定する。"""
+        entity = self.cloth_entity
+        self._cloth = _Cloth(entity)
+        # 根元の縦の辺(各行の先頭の頂点)に一番近い粒子を探して固定する
+        fixed = set()
+        for i in range(self.GRID_Z + 1):
+            p = tuple(float(v) for v in self._grid[i, 0])
+            try:
+                idx = entity.find_closest_particle(p)
+            except TypeError:
+                idx = entity.find_closest_particle(p, 0)
+            idx = int(idx.item()) if hasattr(idx, "item") else int(idx)
+            fixed.add(idx)
+        for idx in sorted(fixed):
+            entity.fix_particle(idx)
+        if not self._cloth.ok:
+            self._warn(f"風を旗に伝えられません({self._cloth.reason})。旗は垂れ下がったままになります")
+        self._ready = True
+
+    def _warn(self, message):
+        if not self._warned:
+            print(f"[注意] 風の旗: {message}")
+            self._warned = True
+
     def update(self, wind_vec, dt):
-        """現在の風(力ベクトル[N]、長さ3)を旗に反映する。毎ステップ呼ぶ。"""
+        """現在の風(力ベクトル[N]、長さ3)を旗に反映する。毎ステップ、scene.step() の前に呼ぶ。"""
         wind = np.asarray(wind_vec, dtype=float).reshape(3)
         self.strength = float(np.linalg.norm(wind))
         self._time += dt
@@ -135,11 +265,39 @@
         target_level = min(self.strength / self.FULL_SCALE_N, 1.0)
         self._level += (target_level - self._level) * min(1.0, dt / self.SMOOTHING_S)
 
-        flutter = self.FLUTTER_RAD * self._level * math.sin(2.0 * math.pi * self.FLUTTER_HZ * self._time)
-        center, quat, _ = flag_pose(wind, self._azimuth, self._level, self.hinge, self.flag_length, flutter)
+        if self._ready and self._cloth is not None and self._cloth.ok:
+            self._push_wind_to_particles(dt)
 
-        if self.n_envs > 0:  # バッチ環境では、全環境に同じ姿勢を書き込む
-            center = np.tile(center, (self.n_envs, 1))
-            quat = np.tile(quat, (self.n_envs, 1))
-        self.flag.set_pos(torch.as_tensor(center, dtype=torch.float32, device=gs.device), zero_velocity=True)
-        self.flag.set_quat(torch.as_tensor(quat, dtype=torch.float32, device=gs.device), zero_velocity=True)
+    def wind_accel(self, positions, velocities, dt):
+        """布の粒子に加える速度の変化 dv (N,3) を返す(Genesis非依存。単体テストできる)。
+
+        positions  : (N,3) 粒子の位置(はためきの位相を、支柱からの距離で決めるのに使う)
+        velocities : (N,3) 粒子の速度(減衰に使う)
+        dv = ( 風下向きの加速度 - 減衰 × 速度 + 波打ち ) × dt
+        """
+        az = np.array([self._azimuth[0], self._azimuth[1], 0.0])
+        side = np.array([-self._azimuth[1], self._azimuth[0], 0.0])   # 水平で風向きに垂直
+        push = self.WIND_ACCEL * self._level * az
+        dist = np.linalg.norm(positions[:, :2] - self.base_xy[None, :], axis=1)
+        phase = 2.0 * math.pi * self.FLUTTER_HZ * self._time - self.FLUTTER_WAVE * dist
+        flutter = (self.FLUTTER_ACCEL * self._level * np.sin(phase))[:, None] * side[None, :]
+        return (push[None, :] - self.DAMPING_PER_S * velocities + flutter) * dt
+
+    def _push_wind_to_particles(self, dt):
+        cloth = self._cloth
+        try:
+            vel_raw = cloth.get_vel()
+            pos_raw = cloth.get_pos()
+            vel = cloth.to_numpy(vel_raw)
+            pos = cloth.to_numpy(pos_raw)
+            n = pos.shape[-2]
+            v = vel.reshape(-1, n, 3)
+            p = pos.reshape(-1, n, 3)[0]            # はためきの位相は環境0の位置で決める(全環境に同じ旗)
+            new_v = v + self.wind_accel(p, v[0], dt)[None, :, :]
+            speed = np.linalg.norm(new_v, axis=-1, keepdims=True)
+            new_v = new_v * np.minimum(1.0, self.MAX_SPEED / (speed + 1e-9))
+            out = torch.as_tensor(new_v.reshape(vel.shape), dtype=torch.float32, device=gs.device)
+            cloth.set_vel(out)
+        except Exception as e:  # noqa: BLE001  版差で形状・引数が合わない場合は、旗だけ止めて続行する
+            cloth.ok = False
+            self._warn(f"粒子の速度を更新できませんでした({type(e).__name__}: {e})。旗への風の反映を止めます")
```

### 6-7 `test_wind_flag.py`(新規ファイル全体)

```python
"""wind_flag.py の単体テスト(偽のGenesis/torchで実行。実機のGenesisは不要)"""
import sys, types, os
import numpy as np

# --- 偽torch ---
torch = types.ModuleType("torch")
torch.float32 = "f32"
torch.as_tensor = lambda x, dtype=None, device=None: np.asarray(x, dtype=np.float32)
sys.modules["torch"] = torch

# --- 偽genesis ---
class Rec:
    def __init__(self, **kw): self.__dict__.update(kw)
gs = types.ModuleType("genesis")
gs.device = "cpu"
gs.options = types.SimpleNamespace(
    PBDOptions=lambda **kw: Rec(kind="pbd", **kw),
    LegacyCouplerOptions=lambda **kw: Rec(kind="coupler", **kw))
gs.textures = types.SimpleNamespace(ColorTexture=lambda **kw: Rec(**kw))
gs.surfaces = types.SimpleNamespace(Rough=lambda **kw: Rec(**kw), Default=lambda **kw: Rec(**kw))
gs.materials = types.SimpleNamespace(PBD=types.SimpleNamespace(Cloth=lambda **kw: Rec(cloth=True, **kw)))
gs.morphs = types.SimpleNamespace(Box=lambda **kw: Rec(box=True, **kw), Mesh=lambda **kw: Rec(mesh=True, **kw))
sys.modules["genesis"] = gs

class FakeCloth:
    def __init__(self, obj, n_envs): 
        verts = [list(map(float, l.split()[1:])) for l in open(obj) if l.startswith("v ")]
        self.p = np.array(verts); self.v = np.zeros_like(self.p); self.fixed=set(); self.n_envs=n_envs
        self.set_calls=0
    def find_closest_particle(self, pos):
        return int(np.argmin(np.linalg.norm(self.p-np.array(pos),axis=1)))
    def fix_particle(self, i): self.fixed.add(i)
    def _b(self,a): return a[None].copy() if self.n_envs else a.copy()
    def get_particles_pos(self): return self._b(self.p)
    def get_particles_vel(self): return self._b(self.v)
    def set_particles_vel(self, v):
        v=np.asarray(v); assert v.shape==((1,)+self.p.shape if self.n_envs else self.p.shape), v.shape
        self.v = v[0].copy() if self.n_envs else v.copy(); self.set_calls+=1

class FakeScene:
    def __init__(self, n_envs=0): self.ents=[]; self.n_envs=n_envs
    def add_entity(self, **kw):
        self.ents.append(kw)
        if "material" in kw: 
            self.cloth=FakeCloth(kw["morph"].file, self.n_envs); return self.cloth
        return Rec()

import wind_flag as wf
from wind_flag import WindFlag

# 1. scene_kwargs
kw = WindFlag.scene_kwargs()
assert kw["coupler_options"].rigid_pbd is False and kw["pbd_options"].kind=="pbd"
print("1 scene_kwargs OK")

for n_envs in (0, 1):
    sc = FakeScene(n_envs)
    f = WindFlag(sc, base_xy=(-2.8,0.0), scale=1.3, n_envs=n_envs, along=(0.0,1.0))
    # 2. 支柱は衝突なし・固定、布はPBD
    pole = sc.ents[0]["morph"]; assert pole.collision is False and pole.fixed is True
    assert "material" in sc.ents[1]
    # 3. OBJ: 頂点数・面数、上辺の高さ・根元の位置
    n=(f.GRID_X+1)*(f.GRID_Z+1); assert sc.cloth.p.shape==(n,3)
    assert abs(sc.cloth.p[:,2].max()-(f.pole_height-0.05*1.3))<1e-5
    # 4. post_build: 根元の縦の辺(GRID_Z+1個)を固定
    f.post_build(); assert len(sc.cloth.fixed)==f.GRID_Z+1, len(sc.cloth.fixed)
    # 5. 無風: 速度は増えない(向きも保持)
    for _ in range(100): f.update(np.zeros(3), 0.01)
    assert np.abs(sc.cloth.v).max()<1e-9 and f._level<1e-9
    # 6. 風: +x方向0.13N → 全粒子が+xに加速、level→1
    for _ in range(100): f.update(np.array([0.13,0,0]), 0.01)
    assert f._level>0.99
    free=[i for i in range(n) if i not in sc.cloth.fixed]
    assert sc.cloth.v[free,0].mean()>0.2, sc.cloth.v[free,0].mean()
    # 7. 向きが追従(-y方向の風)
    for _ in range(100): f.update(np.array([0,-0.13,0]), 0.01)
    assert f._azimuth[1]<-0.99
    print(f"2-7 n_envs={n_envs} OK (set_vel calls={sc.cloth.set_calls})")

# 8. 弱い風では向きを更新しない(CALM)
sc=FakeScene(0); f=WindFlag(sc,(0,0)); f.post_build()
f.update(np.array([0,0.05,0]),0.01); az=f._azimuth.copy()
f.update(np.array([-0.002,0.001,0]),0.01); assert np.allclose(az,f._azimuth); print("8 calm OK")

# 9. 強さと「持ち上がり」の単調性(1粒子が重力+風+減衰で釣り合う角度)
def angle(level):
    g=9.8; a=WindFlag.WIND_ACCEL*level; return np.degrees(np.arctan2(a,g))
vals=[angle(l) for l in (0,0.1,0.3,0.5,1.0)]; assert vals==sorted(vals); print("9 angles(deg)",[round(v) for v in vals])

# 10. 速度上限
sc=FakeScene(0); f=WindFlag(sc,(0,0)); f.post_build(); sc.cloth.v[:]=100.0
f.update(np.array([0.13,0,0]),0.01); assert np.linalg.norm(sc.cloth.v,axis=1).max()<=WindFlag.MAX_SPEED+1e-6; print("10 clamp OK")

# 11. 版差: 速度設定メソッドが無い → 止めずに警告1回だけ
sc=FakeScene(0); f=WindFlag(sc,(0,0)); del FakeCloth.set_particles_vel
f.post_build(); 
for _ in range(5): f.update(np.array([0.1,0,0]),0.01)
assert f._cloth.ok is False and f._warned; print("11 missing-API fallback OK")

# 12. 実行時エラー(形状不一致)でも例外を出さず無効化
FakeCloth.set_particles_vel=lambda self,v: (_ for _ in ()).throw(ValueError("shape"))
sc=FakeScene(0); f=WindFlag(sc,(0,0)); f.post_build(); f.update(np.array([0.1,0,0]),0.01); f.update(np.array([0.1,0,0]),0.01)
assert f._cloth.ok is False; print("12 runtime-error fallback OK")
print("ALL PASS")
```