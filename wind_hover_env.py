"""
WindHoverEnv: Genesis World公式の examples/drone/hover_env.py (HoverEnv) をベースに、
OU過程(Ornstein-Uhlenbeck process)による時間変化する風外乱を追加した環境。

ベースにした公式コード:
  https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/drone/hover_env.py

変更点(公式HoverEnvとの差分):
  1. 風パラメータ(wind_mean/theta/sigma/vec)をバッチテンソルとして追加
  2. reset_idx()で風パラメータをエピソードごとに再抽選(コマンドの再抽選と同じタイミング)
  3. step()で scene.step() の直前に、ドローンのCOMリンクへOU過程の風力を加える
  4. reset_idx()で初期位置を base_init_pos 周辺でランダム化(ドメインランダム化)

観測・報酬・終了条件は公式HoverEnvと同一(風の情報は観測に含めない。
風を直接観測させず、姿勢の乱れから間接的に対処させることで、ロバスト性を狙う)。

このファイル中の `apply_external_force` の呼び出しは Genesis World の
`RigidLink.apply_external_force(force, envs_idx=None, *, pos=None, ref=..., local=False)`
API に基づく想定です。お使いのGenesisのバージョンで `drone.links[drone.COM_link_idx]`
が期待通りのRigidLinkを返すか、最初に短いスクリプトで一度確認してから
本番学習に進んでください(バージョンによりCOMリンクへのアクセス方法が
変わっている可能性があります)。
"""

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
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
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
            self.cam = self.scene.add_camera(
                res=(640, 480), pos=(3.5, 0.0, 2.5), lookat=(0, 0, 0.5), fov=30, GUI=True,
            )

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf")
        )

        self.scene.build(n_envs=num_envs)

        # ドローンのCOMリンク(風・外乱力を加える対象)
        self.com_link = self.drone.links[self.drone.COM_link_idx]

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
        """OU過程で風を更新し、ドローンのCOMリンクへ外力として加える"""
        noise = torch.randn((self.num_envs, 3), device=gs.device)
        self.wind_vec += (
            self.wind_theta.unsqueeze(1) * (self.wind_mean - self.wind_vec) * self.dt
            + self.wind_sigma.unsqueeze(1) * math.sqrt(self.dt) * noise
        )
        self.com_link.apply_external_force(self.wind_vec, local=False)

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
        self.drone.set_propellers_rpm((1 + exec_actions * 0.8) * 14468.429183500699)

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

    # ---- 報酬関数(公式HoverEnvと同一) ---- #
    def _reward_target(self):
        return torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159
        return torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))

    def _reward_angular(self):
        return torch.norm(self.base_ang_vel / 3.14159, dim=1)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew