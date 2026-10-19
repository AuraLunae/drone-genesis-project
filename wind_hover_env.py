"""Wind hover environment with OU wind disturbances."""

import torch
import math
import copy
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

from genesis_compat import PropellerVisual, apply_com_force
from wind_flag import create_wind_flag, wind_flag_scene_kwargs


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class WindHoverEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, wind_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # 100Hz (公式サンプルと同じ)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.wind_cfg = wind_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # 風の見える化(白い旗)。評価(eval.py)で env_cfg["visualize_wind_flag"]=True にした時だけ出す。
        # 既定の視点(上から見下ろす位置)では、後方に立てた旗が画面の外に出てしまうため、
        # 旗を出す時だけ、低めに引いた視点に切り替える。
        show_flag = bool(env_cfg.get("visualize_wind_flag", False))
        # デフォルトを procedural に切り替え
        flag_style = env_cfg.get("wind_flag_style", "procedural")  # procedural が既定
        flag_scale = env_cfg.get("wind_flag_scale", 1.3)
        # 布(PBD)の旗には、布とドローン・地面の連成を切る設定などが要る(板の旗なら空)
        flag_scene_kwargs = wind_flag_scene_kwargs(flag_style, flag_scale) if show_flag else {}
        viewer_camera = (
            dict(camera_pos=(4.5, 0.0, 2.2), camera_lookat=(-1.0, 0.0, 1.3), camera_fov=50)
            if show_flag
            else dict(camera_pos=(3.0, 0.0, 3.0), camera_lookat=(0.0, 0.0, 1.0), camera_fov=40)
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt / 2,
                enable_collision=True,
                enable_joint_limit=True,
                constraint_solver=gs.constraint_solver.Newton,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.rendered_env_num)),
            ),
            viewer_options=gs.options.ViewerOptions(**viewer_camera),
            show_viewer=show_viewer,
            **flag_scene_kwargs,
        )

        self.scene.add_entity(gs.morphs.Plane())

        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5)),
                ),
            )
        else:
            self.target = None

        if self.env_cfg["visualize_camera"]:
            record_camera = (
                dict(pos=(4.5, 0.0, 2.2), lookat=(-1.0, 0.0, 1.3), fov=50)
                if show_flag
                else dict(pos=(3.5, 0.0, 2.5), lookat=(0, 0, 0.5), fov=30)
            )
            self.cam = self.scene.add_camera(res=(640, 480), GUI=True, **record_camera)

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf")
        )

        # 風の強さと向きを映す白い旗(衝突判定なし)。ドローンの後方(カメラの反対側)に立てる。
        self.wind_flag = (
            create_wind_flag(
                self.scene,
                base_xy=env_cfg.get("wind_flag_xy", (-2.8, 0.0)),
                scale=flag_scale,
                n_envs=num_envs,
                style=flag_style,
            )
            if show_flag
            else None
        )

        self.scene.build(n_envs=num_envs)
        if self.wind_flag is not None:
            self.wind_flag.after_build()   # 布の旗は、build の後に支柱へ留める

        # ドローンのCOMリンク(風・外乱力を加える対象)
        # 注: Genesis 1.2.3 では DroneEntity.COM_link_idx が未初期化(内部属性
        # _COM_link_idx が設定されずAttributeErrorになる)ため、COM_link_idx経由では
        # なく、CF2XのURDFに実在するリンク名"base_link"を直接指定して取得する。
        self.com_link = self.drone.get_link("base_link")
        self.com_link_idx = self.com_link.idx

        # 風を加えるAPI(版によって名前・置き場所が違う)は genesis_compat.apply_com_force
        # に集約してある。重心(link_com)に並進力だけを加え、トルクは誘発しない
        # (verify_physics.py のテスト8で確認済み)。

        # 学習中はビューアが無いので何もしない。評価(eval.py)でビューアを出す時だけ、
        # プロペラが加速で止まって見える現象(ワゴンホイール効果)を避ける表示補正を入れる。
        self.prop_visual = PropellerVisual(self.drone, self.dt) if show_viewer else None

        # 事前学習時と同じく、報酬スケールにdtを乗じる
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        # --- 風(OU過程)用バッファ ---
        self.wind_vec = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.wind_mean = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.wind_theta = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.wind_sigma = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.extras = dict()
        self.reset()

    # ------------------------------------------------------------------ #
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), gs.device)

    def _resample_wind(self, envs_idx):
        """風のベース方向・強さ・収束速度・乱れ幅をエピソードごとに再抽選する"""
        n = len(envs_idx)
        if n == 0:
            return
        direction = torch.randn((n, 3), device=gs.device)
        direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)
        strength = gs_rand_float(*self.wind_cfg["strength_range"], (n,), gs.device)
        self.wind_mean[envs_idx] = direction * strength.unsqueeze(1)
        self.wind_theta[envs_idx] = gs_rand_float(*self.wind_cfg["theta_range"], (n,), gs.device)
        self.wind_sigma[envs_idx] = gs_rand_float(*self.wind_cfg["sigma_range"], (n,), gs.device)
        self.wind_vec[envs_idx] = 0.0

    def _update_and_apply_wind(self):
        """OU過程で風を更新し、ドローンのCOMリンクへ外力として加える。

        注: OU過程はガウスノイズを含むため理論上は無限大の値も取り得る。
        8192並列×毎ステップでガウス乱数を引き続けると、試行回数が膨大なため
        「滅多に起きない極端な外れ値」がいつかは必ず発生し、軽量な機体(約27g)に
        一瞬で過大な力が加わって物理シミュレーションがNaNで発散する原因になり得る。
        これを防ぐため、風の大きさ(ノルム)を strength_range の上限の数倍で
        クリップする(方向は変えず大きさだけ制限する)。
        """
        noise = torch.randn((self.num_envs, 3), device=gs.device)
        self.wind_vec += (
            self.wind_theta.unsqueeze(1) * (self.wind_mean - self.wind_vec) * self.dt
            + self.wind_sigma.unsqueeze(1) * math.sqrt(self.dt) * noise
        )

        max_wind_norm = self.wind_cfg["strength_range"][1] * self.wind_cfg.get("max_norm_multiplier", 3.0)
        wind_norm = torch.norm(self.wind_vec, dim=1, keepdim=True)
        clip_scale = torch.clamp(max_wind_norm / (wind_norm + 1e-8), max=1.0)
        self.wind_vec = self.wind_vec * clip_scale

        apply_com_force(self.drone, self.com_link_idx, self.wind_vec, n_envs=self.num_envs)

        if self.wind_flag is not None:  # 旗は環境0の風を映す(評価時は環境数1)
            self.wind_flag.update(self.wind_vec[0].detach().cpu().numpy(), self.dt)

    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]) 
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    # ------------------------------------------------------------------ #
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # 14468.429... はCF2Xモデルのホバリング基準RPM
        propeller_rpms = (1 + exec_actions * 0.8) * 14468.429183500699
        self.drone.set_propellers_rpm(propeller_rpms)
        if self.prop_visual is not None:
            self.prop_visual.apply(propeller_rpms)  # 表示用のプロペラ角度だけを上書き(物理には影響しない)

        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)

        # 風を物理ステップの直前に加える(1 RLステップ=1 scene.step()、dt=0.01)
        self._update_and_apply_wind()

        self.scene.step()

        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]) 
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]) 
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"]) 
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"]) 
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"]) 
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )

        # 角速度による終了条件。この物理モデルには空力抵抗が無く(verify_physics.py
        # のテスト6)、ヨー反トルクで回り始めると減衰せず際限なく角速度が積み上がる。
        # 学習初期のランダムな方策がヨーを空転させ続けるのを、ここで打ち切る。
        # 設定が無い(古いcfgs.pklを読み込んだ)場合は何もしない。
        max_ang_vel = self.env_cfg.get("termination_if_ang_vel_greater_than")
        if max_ang_vel is not None:
            self.crash_condition = self.crash_condition | (torch.norm(self.base_ang_vel, dim=1) > max_ang_vel)

        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._update_observation()
        self.last_actions[:] = self.actions[:]

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _update_observation(self):
        # 公式HoverEnvと同じ観測構成。風そのものは観測に含めない(非観測の外乱として扱う)。
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # 初期位置をbase_init_pos周辺でランダム化(ドメインランダム化)
        n = len(envs_idx)
        init_offset = gs_rand_float(-0.3, 0.3, (n, 3), gs.device)
        init_offset[:, 2] = torch.abs(init_offset[:, 2])  # 地面にめり込まないようZは正方向のみ

        self.base_pos[envs_idx] = self.base_init_pos + init_offset
        self.last_base_pos[envs_idx] = self.base_pos[envs_idx]
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self._resample_wind(envs_idx)
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    # ---- 報酬関数 ---- #
    def _reward_target(self):
        """進歩報酬(公式HoverEnvと同一、potential-basedなので最適方策を変えずに学習を加速する)"""
        return torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)

    def _reward_alive(self):
        """生存ボーナス: クラッシュしない限り毎ステップ一定値。
        「遠くにいる減点」の累積が「生き残る価値」を上回って、わざと早く
        墜落した方が得、という誤学習を防ぐための土台。"""
        return torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _reward_distance_penalty(self):
        """「遠くにいること」自体へのジワジワ減点。tanhで頭打ちにすることで
        無限に悪化しないようにする(生存ボーナスを食いつぶさないため)。
        distance_penalty_scale_m 付近の距離でペナルティが概ね最大(-1)に近づく。"""
        dist = torch.norm(self.rel_pos, dim=1)
        d0 = self.reward_cfg["distance_penalty_scale_m"]
        return -torch.tanh(dist / d0)

    def _reward_at_target_bonus(self):
        """目標半径内に実際にいる間、毎ステップ明示的に加点する。
        _reward_target は「近づいた差分」にしか報酬を与えないため、
        「目標付近に留まり続ける」ことそのものへの加点をここで補う。"""
        dist = torch.norm(self.rel_pos, dim=1)
        return (dist < self.env_cfg["at_target_threshold"]).float()

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159
        return torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))

    def _reward_angular(self):
        """角速度の安定性報酬。rsl-rl系(legged_gym)の標準設計に合わせ、
        生のノルム/二乗和ではなく exp(-k * ||ang_vel||^2) という指数カーネルで
        [0, 1] に有界化する(0に近いほど悪い、1が最良=静止)。

        理由: 生のノルムだと、何らかの理由で角速度が一瞬跳ね上がった際に
        この項だけが青天井に悪化し、他の報酬項目(せいぜい0.1〜数程度)を
        桁違いに圧倒して学習を不安定化させる(実際に rew_angular が -2551
        まで悪化し、総報酬のほぼ全てを占めてしまう現象が発生した)。
        指数カーネルなら最悪でも0に漸近するだけなので、この暴走を防げる。"""
        ang_vel_sq = torch.sum(torch.square(self.base_ang_vel), dim=1)
        return torch.exp(-self.reward_cfg["angular_penalty_sigma"] * ang_vel_sq)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
