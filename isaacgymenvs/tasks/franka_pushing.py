# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import math

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array   
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaPushing(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.bin_radius = 0.15
        # self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_position_noise = self.bin_radius - 0.03
        self.goal_position_noise = self.start_position_noise
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.n_cubes_train = self.cfg["env"]["nCubes"]
        self.n_cubes_test = 6
        self.n_observed_cubes = self.cfg["env"].get("nObservedCubes", self.n_cubes_train)
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.dist_reward_dropoff  = self.cfg["env"]["distRewardDropoff"]
        self.dist_reward_threshold  = self.cfg["env"]["distRewardThreshold"]
        self.test_task = self.cfg["env"].get("testTask", -1)
        self.friction = self.cfg["env"].get("friction", 1)

        # modes 
        self.mode = self.cfg["env"].get("mode", '')
        assert self.mode in ['', 'easy', 'grasping']
        self.distance_from_block = self.cfg["env"].get("distanceFromBlock", 0.0)

        self.target_idx = [14,15,16]
        self.target_name = 'cube0_pos'

        # Create dicts to pass to reward function
        self.reward_settings = {}

        # Controller type
        self.observe_velocities = self.cfg["env"]["observeVelocities"]
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 12 + 10 * self.n_observed_cubes if self.control_type == "osc" else 19 + 10 * self.n_observed_cubes
        if self.observe_velocities:
            self.cfg["env"]["numObservations"] += 6 + 3 * self.n_observed_cubes
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                                # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                              # will be dict mapping names to relevant sim handles
        self.num_dofs = None                            # Total number of DOFs per env
        self.actions = None                             # Current actions to be deployed
        self._init_cube_states = [None] * self.n_cubes_test  # Initial state of cubes for the current env
        self._cube_states = [None] * self.n_cubes_test       # Current state of cubes for the current env
        self._cube_ids = [None] * self.n_cubes_test          # Actor ID corresponding to cubes for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
        self._goal_state = None

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx()

        # Refresh tensors
        self._refresh()
                

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        self.define_tasks()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.dynamic_friction = self.friction
        plane_params.static_friction = self.friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        wall_opts = gymapi.AssetOptions()
        wall_opts.fix_base_link = True
        wall_assets = []
        thick = 0.01
        tall = 0.1
        apart = self.bin_radius
        wall_assets += [self.gym.create_box(self.sim, *[thick, apart * 2 + thick, tall], wall_opts)]
        wall_assets += [self.gym.create_box(self.sim, *[thick, apart * 2 + thick, tall], wall_opts)]
        wall_assets += [self.gym.create_box(self.sim, *[apart * 2 + thick, thick, tall], wall_opts)]
        wall_assets += [self.gym.create_box(self.sim, *[apart * 2 + thick, thick, tall], wall_opts)]
        wall_poses = []
        wall_poses += [[apart, 0.0, 1 + tall / 2]]
        wall_poses += [[-apart, 0.0, 1 + tall / 2]]
        wall_poses += [[0.0, apart, 1 + tall / 2]]
        wall_poses += [[0.0, -apart, 1 + tall / 2]]

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.cube_sizes = [0.07] + [0.07] * self.n_cubes_test

        # Create cubeA asset
        cube_assets = []
        self.cube_colors = cube_colors = []
        cube_options = gymapi.AssetOptions()
        cube_options.density = 1000
        for j in range(self.n_cubes_test):
            cube_assets += [self.gym.create_box(self.sim, *([self.cube_sizes[j]] * 3), cube_options)]
            cube_colors += [gymapi.Vec3(0.0, np.random.rand(), np.random.rand())]
        cube_colors[0] = gymapi.Vec3(0.6, 0.1, 0.0)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []

        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cube_start_poses = []
        for j in range(self.n_cubes_test):
            cube_start_poses += [gymapi.Transform()]
            cube_start_poses[j].p = gymapi.Vec3(-1.0, 0.0, 0.0)
            cube_start_poses[j].r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Assets for goal
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.02, asset_options)
        default_pose = gymapi.Transform()
        default_pose.p.z = 1.0

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 7 + self.n_cubes_test     # 1 for table, table stand, n cubes, goal, 4 walls
        max_agg_shapes = num_franka_shapes + 7 + self.n_cubes_test     # 1 for table, table stand, n cubes, goal, 4 walls

        self.frankas = []
        self.envs = []
        if self.enable_camera_sensors:
            self.cams = []
            self.cam_tensors = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            # Create bin
            wall_actors = []
            for j in range(4):
                wall_start_pose = gymapi.Transform()
                wall_start_pose.p = gymapi.Vec3(*wall_poses[j])
                wall_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                wall_actors += [self.gym.create_actor(env_ptr, wall_assets[j], wall_start_pose, "table", i, 1, 0)]

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create goal
            self._marker_id = self.gym.create_actor(env_ptr, marker_asset, default_pose, "marker", num_envs, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, self._marker_id, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

            # Create cubes
            for j in range(self.n_cubes_test):
                self._cube_ids[j] = self.gym.create_actor(env_ptr, cube_assets[j], cube_start_poses[j], f"cube{j}", i, 0, 0)
                # set colors
                self.gym.set_rigid_body_color(env_ptr, self._cube_ids[j], 0, gymapi.MESH_VISUAL, cube_colors[j])
                # Set friction
                cube_props = gymapi.RigidShapeProperties()
                cube_props.friction = self.friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, self._cube_ids[j], [cube_props])

            # Set up camera
            if self.enable_camera_sensors and i < self.max_pix:
                # Camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cam_w
                cam_props.height = self.cam_w
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                # Top view
                self.gym.set_camera_location(cam_handle, env_ptr, gymapi.Vec3(0.12, 0, 1.25), gymapi.Vec3(0, 0, 1))
                # Side view
                # self.gym.set_camera_location(cam_handle, env_ptr, gymapi.Vec3(0.22, 0, 1.25), gymapi.Vec3(-0.3, 0, 1))
                self.cams.append(cam_handle)
                # Camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
                self.cam_tensors.append(cam_tensor_th)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        for j in range(self.n_cubes_test):
                self._init_cube_states[j] = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            # Cubes
            # "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            # "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        for j in range(self.n_cubes_test):
            self._cube_states[j] = self._root_state[:, self._cube_ids[j], :]
        self._goal_state = self._root_state[:, self._marker_id, :]

        # Initialize states
        for j in range(self.n_cubes_test):
            self.states.update({f"cube{j}_size": torch.ones_like(self._eef_state[:, 0]) * self.cube_sizes[j]})

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * (8 + self.n_cubes_test), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],

            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            # "cube0_to_cube1_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
            "goal_pos": self._goal_state[:, :3],
        })
        # Cubes
        for j in range(self.n_cubes_test):
            self.states.update({
                f"cube{j}_angvel": self._cube_states[j][:, 10:],
                f"cube{j}_vel": self._cube_states[j][:, 7:10],
                f"cube{j}_quat": self._cube_states[j][:, 3:7],
                f"cube{j}_pos": self._cube_states[j][:, :3],
            })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_observations(self):
        self._refresh()
        obs = ["eef_pos", "eef_quat", "goal_pos"]
        if self.observe_velocities:
            obs += ["eef_vel"]
        for j in range(self.n_observed_cubes):
            obs = obs + [f"cube{j}_quat", f"cube{j}_pos", f"cube{j}_vel"]
            if self.observe_velocities:
                obs += [f"cube{j}_angvel"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf
    
    def _compute_pixel_obs_save(self):
        """Save images for debugging"""
        img = []
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(min(self.num_envs, self.max_pix)):
            crop_l = (self.cam_w - self.im_size) // 2
            crop_r = crop_l + self.im_size
            img.append(self.cam_tensors[i][:, crop_l:crop_r, :3].cpu().numpy())
        self.gym.end_access_image_tensors(self.sim)

        # save images
        import imageio
        img = np.array(img)
        assert img.shape == (16, self.im_size, self.im_size, 3)
        img = img.reshape(4, 4, self.im_size, self.im_size, 3)
        img = img.transpose(0, 2, 1, 3, 4).reshape(4*self.im_size, 4*self.im_size, 3)
        imageio.imwrite("render.png", img)

    def compute_pixel_obs(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(min(self.num_envs, self.max_pix)):
            crop_l = (self.cam_w - self.im_size) // 2
            crop_r = crop_l + self.im_size
            self.pix_buf[i] = self.cam_tensors[i][:, crop_l:crop_r, :3].permute(2, 0, 1).float() / 255.
            # self.obs_buf[i] = (self.obs_buf[i] - self.im_mean) / self.im_std
        self.gym.end_access_image_tensors(self.sim)

    def reset(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.freeze_cubes()
        self.gym.simulate(self.sim)
        self.reset_idx(env_ids)
        self.progress_buf += 1

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict
    
    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        for j in range(self.n_cubes_test):
            self._reset_init_cube_state(cube=j, env_ids=env_ids, check_valid=j>0)
            # Write these new init states to the sim states
            self._cube_states[j][env_ids] = self._init_cube_states[j][env_ids]
        self._reset_goal_state(env_ids=env_ids)

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        
        if self.mode == 'easy':
            # pos[:, :] = torch.tensor([ 0.0871,  0.5839, -0.0948, -2.5892, -0.1235,  3.4638,  0.9550,  0.0400, 0.0400], device=self.device)[None]
            x1 = torch.tensor([ 0.0871,  0.5839, -0.0948, -2.5892, -0.1235,  3.4638,  0.9550,  0.0400, 0.0400], device=self.device)
            x2 = torch.tensor([ 0.2171,  0.0121, -0.2010, -2.7235,  0.0092,  3.0235,  0.8217,  0.0400, 0.0400], device=self.device)
            # a = torch.rand(len(env_ids), device=self.device)[:, None]
            a = torch.ones(len(env_ids), device=self.device)[:, None] * self.distance_from_block
            # height ~ 1.11 + 0.17 * a
            pos = x1 * (1 - a) + x2 * a

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -(1 + self.n_cubes_test):].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def freeze_cubes(self):
        # TODO this doesn't work with self.test_task
        if self.test:
            # Freeze
            cube_colors = [gymapi.Vec3(0.5, 0.5, 0.5) for _ in range(len(self._cube_ids) - 1)]
            mass_multiplier = 100
        else:
            # Restore
            cube_colors = self.cube_colors[1:]
            mass_multiplier = 1
        
        for t, task in enumerate(self.tasks):
            if task.get('rigid', False):
                for i, id in enumerate(self._cube_ids[1:]):
                    properties = self.gym.get_actor_rigid_body_properties(self.envs[t], id)[0]
                    if self.cube_masses[t] is None: self.cube_masses[t] = properties.mass
                    properties.mass = self.cube_masses[t] * mass_multiplier
                    self.gym.set_actor_rigid_body_properties(self.envs[t], id, [properties], True)

                    num_bodies = self.gym.get_actor_rigid_body_count(self.envs[t], id)
                    for n in range(num_bodies):
                        self.gym.set_rigid_body_color(self.envs[t], id, n, gymapi.MESH_VISUAL, cube_colors[i])


    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        this_cube_state_all = self._init_cube_states[cube]

        # Sampling is "centered" around middle of table
        table_center = torch.tensor(self._table_surface_pos[:3], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        table_center[2] = table_center[2] + 0.15
        if self.mode == 'easy':
            table_center[2] = table_center[2] - 0.1

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid: 
            # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
            min_dists = (self.states[f"cube0_size"] + self.states[f"cube1_size"])[env_ids] * np.sqrt(2) / 2.0

            # We scale the min dist by 2 so that the cubes aren't too close together
            min_dists = min_dists * 2.0

            success = False
            previous_cube_states = torch.stack(self._init_cube_states[:cube], 0)[:, env_ids, :]
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, :3] = table_center + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :3]) - 0.5)
                # Check if sampled values are valid

                cube_dist = torch.linalg.norm(sampled_cube_state[None, :, :3] - previous_cube_states[:, :, :3], dim=-1)
                active_idx = torch.nonzero((cube_dist < min_dists).sum(0), as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            # assert success, "Sampling cube locations was unsuccessful! ):"
            
        else:
            # We just directly sample
            sampled_cube_state[:, :3] = table_center.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 3, device=self.device) - 0.5)
    
        if not self.test:
            # remove extra cubes
            if cube >= self.n_cubes_train:
                off = np.array([0.0, 0.0, -0.2])
                for i in range(3):
                    sampled_cube_state[:, i] = table_center[i] + off[i]

        if self.test:
            for t in range(min(len(self.tasks), sampled_cube_state.shape[0])):
                for i in range(3):
                    sampled_cube_state[t, i] = table_center[i] + self.tasks[t]['cubes'][cube][i]

        # Test specific task
        if self.test_task >=0 and self.test:
            for i in range(3):
                sampled_cube_state[:, i] = table_center[i] + self.tasks[self.test_task]['cubes'][cube][i]

        # Sample rotation value
        if not self.test and self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_cube_state_all[env_ids, :] = sampled_cube_state


    def define_tasks(self):
        n_tasks = 16
        self.tasks = tasks = [None] * n_tasks 
        height = -0.115 # height at ground (1.06)
        center = np.array([.0, .0, height])
        left = np.array([.0, -.08, height])
        right = np.array([.0, .08, height])
        bottom = np.array([.08, .0, height])
        top = np.array([-.08, .0, height])
        bottom_right = np.array([.08, .08, height])
        bottom_left = np.array([.08, -.08, height])
        top_right = np.array([-.08, .08, height])
        top_left = np.array([-.08, -.08, height])
        off = np.array([0.3, 0.3, height])

        tasks[0]  = dict(cubes=[top_left, off, off, off, off, off],                                             name='push')
        tasks[1]  = dict(cubes=[top_left, left, off, off, off, off], rigid=True,                                name='1 cube blocking way')
        tasks[2]  = dict(cubes=[top_left, left, center, off, off, off],                                         name='2 cubes blocking way')
        tasks[3]  = dict(cubes=[top_left, left, center, right, off, off],                                       name='3 cubes blocking way')
        tasks[4]  = dict(cubes=[top_left, bottom_left, off, off, off, off],                                     name='1 cube at the goal')
        tasks[5]  = dict(cubes=[top_left, top, center, left, bottom, right],                                    name='star')
        tasks[6]  = dict(cubes=[top_left, bottom_right, left, bottom_left, bottom, center],                     name='6 cubes blocking way')
        tasks[7]  = dict(cubes=[top_left, left, bottom_left, [-0.08, -0.08, 0.0], bottom, center],              name='star + cube on top')
        tasks[8]  = dict(cubes=[[-0.08, -0.08, 0.0], left, bottom_left, top_left, bottom, center],              name='star + red is on top')
        tasks[9]  = dict(cubes=[[.03, -.11, height], [.03, -.04, height], [-.05, -.11, height], off, off, off], name='tool use')
        tasks[10] = dict(cubes=[top_left, [.04, -.11, height], [.11, -.04, height], off, off, off], rigid=True, name='insertion')
        tasks[11] = dict(cubes=[center, [.07, -.11, height], [-.07, -.11, height], [.07, -.11, -0.045], [-.07, -.11, -0.045], [.0, -.11, -0.045]], 
                         goal=[.0, -.11, 0], rigid=True, name='insertion 2')
        tasks[12] = dict(cubes=[center, [-.11, .07, height], [-.11, -.07, height], [-.11, .07, -0.045], [-.11, -.07, -0.045], [-.11, .0, -0.045]], 
                         goal=[-.11, .0, 0], rigid=True, name='insertion 3')
        tasks[13] = dict(cubes=[center, [.07, -.11, height], [-.07, -.11, height], off, off, off], 
                         goal=[.0, -.11, 0], rigid=True, name='insertion 4')
        tolerance = 0.005; 
        tasks[14] = dict(cubes=[center, [.07 + tolerance, -.11, height], [-.07 - tolerance, -.11, height], off, off, off], 
                         goal=[.0, -.11, 0], rigid=True, name='tolerance')
        tolerance = 0.01; 
        tasks[15] = dict(cubes=[center, [.07 + tolerance, -.11, height], [-.07 - tolerance, -.11, height], off, off, off], 
                         goal=[.0, -.11, 0], rigid=True, name='tolerance')
        
        self.cube_masses = [None] * n_tasks


    def _reset_goal_state(self, env_ids):
        """

        Args:
            env_ids (tensor or None): Specific environments to reset cube for
        """
        # If env_ids is None, we reset all the envs
        # TODO randomize Z
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_goal_state = torch.zeros(num_resets, 13, device=self.device)

        # Sampling is "centered" around middle of table
        center = torch.tensor(self._table_surface_pos[:3], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        center[2] = center[2] + 0.05

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_goal_state[:, 6] = 1.0

        sampled_goal_state[:, :2] = center[:2].unsqueeze(0) + \
                                            2.0 * self.goal_position_noise * (
                                                    torch.rand(num_resets, 2, device=self.device) - 0.5)
        sampled_goal_state[:, 2] = center[2]
        if self.mode == 'grasping':
            # sampled_goal_state[:, 2] = center[2] + 0.14 * torch.rand(num_resets, device=self.device)
            sampled_goal_state[::2, 2] = center[2] + 0.14 * torch.rand(num_resets // 2, device=self.device)

        if self.test:
            sampled_goal_state[:, 0] = center[0] + 0.11
            sampled_goal_state[:, 1] = center[1] - 0.11
            sampled_goal_state[:, 2] = center[2] 
            for t in range(len(self.tasks)):
                if 'goal' in self.tasks[t]:
                    for i in range(3):
                        sampled_goal_state[t, i] = center[i] + self.tasks[t]['goal'][i]

        # Test specific task
        if self.test_task >=0 and self.test:
            if 'goal' in self.tasks[self.test_task]:
                    for i in range(3):
                        sampled_goal_state[:, i] = center[i] + self.tasks[self.test_task]['goal'][i]

        # Lastly, set these sampled values as the new init state
        self._goal_state[env_ids, :] = sampled_goal_state


    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        # Increment counter
        self.progress_buf += 1
        self.reset_buf[:] = torch.where((self.progress_buf >= self.max_episode_length), torch.ones_like(self.reset_buf), self.reset_buf)
        self.done = self.reset_buf.clone()
        
        # Produce observation
        self.compute_observations()
        self.rew_buf[:] = self.compute_franka_reward(self.states)

        # Extra logging
        if 'images' in self.extras:
            self.extras.pop("images")
        if self._do_override_render:
            self.override_render = True
        if self.render_this_step():
            self.compute_pixel_obs()
            self.extras["images"] = self.pix_buf

            # Add success marker
            marker_size = self.im_size // 32
            success = (torch.norm(self.states["goal_pos"] - self.states[self.target_name], dim=-1) < 0.02)[:self.max_pix]
            self.extras["images"][:,:,:marker_size,:marker_size] = torch.tensor([0., 0., 0.], device=self.device)[None, :, None,None]
            self.extras["images"][success,:,:marker_size,:marker_size] = torch.tensor([0., 0.75, 0.], device=self.device)[None, :, None,None]
            # import imageio
            # imageio.imwrite("render.png", (self.extras["images"].cpu().numpy().transpose(0, 2, 3, 1).reshape(4, 4, self.im_size, self.im_size, 3).transpose(0, 2, 1, 3, 4).reshape(4*self.im_size, 4*self.im_size, 3) * 255).astype(np.uint8))

        metrics = dict()
        metrics["goal_dist"] = torch.norm(self.states["goal_pos"] - self.states[self.target_name], dim=-1)
        metrics["success_4"] = torch.norm(self.states["goal_pos"] - self.states[self.target_name], dim=-1) < 0.04
        metrics["success_2"] = torch.norm(self.states["goal_pos"] - self.states[self.target_name], dim=-1) < 0.02
        metrics["failure_2"] = torch.norm(self.states["goal_pos"] - self.states[self.target_name], dim=-1) > 0.02
        if self.test:
            for i in range(min(self.max_pix, self.states['goal_pos'].shape[0])):
                metrics[f"goal_dist_{i}"] = torch.norm(self.states["goal_pos"] - self.states[self.target_name], dim=-1)[i]
                metrics[f"success_4_{i}"] = torch.norm(self.states["goal_pos"] - self.states[self.target_name], dim=-1)[i] < 0.04
                metrics[f"success_2_{i}"] = torch.norm(self.states["goal_pos"] - self.states[self.target_name], dim=-1)[i] < 0.02
                metrics[f"failure_2_{i}"] = torch.norm(self.states["goal_pos"] - self.states[self.target_name], dim=-1)[i] > 0.02
        self.extras["episodic"] = metrics
        # self.extras["episode_cumulative"]["cubeA_vel"] = torch.norm(self.states["cubeA_vel"], dim=-1)
        # self.extras["episode_cumulative"]["cubeA_vel"] = torch.norm(self.states["cubeA_vel"], dim=-1)
        # self.extras["episode_cumulative"]["cubeB_vel"] = torch.norm(self.states["cubeB_vel"], dim=-1)
        
        # Reset if needed
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        # debug viz
        # if self.viewer and self.debug_viz:
        #     self.gym.clear_lines(self.viewer)
        #     self.gym.refresh_rigid_body_state_tensor(self.sim)

        #     # Grab relevant states to visualize
        #     eef_pos = self.states["eef_pos"]
        #     eef_rot = self.states["eef_quat"]
        #     cubeA_pos = self.states["cubeA_pos"]
        #     cubeA_rot = self.states["cubeA_quat"]
        #     cubeB_pos = self.states["cubeB_pos"]
        #     cubeB_rot = self.states["cubeB_quat"]

        #     # Plot visualizations
        #     for i in range(self.num_envs):
        #         for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
        #             px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #             py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #             pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #             p0 = pos[i].cpu().numpy()
        #             self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
        #             self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
        #             self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

    def compute_franka_reward(self, states):
        ## type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float, float, float, float) -> Tuple[Tensor, Tensor]

        # distance from cube to the goal
        d = torch.norm(states["goal_pos"] - states["cube0_pos"], dim=-1)
        if self.dist_reward_threshold:
            dist_reward = torch.where(d < self.dist_reward_threshold, torch.ones_like(d), torch.zeros_like(d))
        else:
            dist_reward = 1 - torch.tanh(self.dist_reward_dropoff * d)

        return self.dist_reward_scale * dist_reward

