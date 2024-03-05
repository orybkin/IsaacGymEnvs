import imageio
from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from .franka_pushing import axisangle2quat


class CubesDebug:
    """Simple environment for debugging cube resets"""
    
    def __init__(self, n_cubes):
        self.n_cubes = n_cubes   
        self.num_envs = 1
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.gym = gymapi.acquire_gym()
        
        self._init_cube_states = [None] * self.n_cubes  # Initial state of cubes for the current env
        self._cube_states = [None] * self.n_cubes       # Current state of cubes for the current env
        self._cube_ids = [None] * self.n_cubes          # Actor ID corresponding to cubes for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._rigid_body_state = None
        self.states = {}

        self.envs = []
        self.bin_radius = 0.15
        self.start_position_noise = self.bin_radius - 0.03
        self.start_rotation_noise = 0.785
        self.cam_tensors = []

        self._init_cube_states = [None] * self.n_cubes  # Initial state of cubes for the current env
        self._cube_states = [None] * self.n_cubes
        self.cube_mass = None
        self.frozen = False

        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.init_data()

        self.reset_idx()
        self.refresh()
        
        
    def create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.physx.solver_type = 1
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        self._create_ground_plane()
        self._create_envs()
        

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self):
        spacing = 1.5
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # table pos
        # table_pos = [0.0, 0.0, 1.0]
        # table_thickness = 0.05
        # table_opts = gymapi.AssetOptions()
        # table_opts.fix_base_link = True
        # table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # table_start_pose = gymapi.Transform()
        # table_start_pose.p = gymapi.Vec3(*table_pos)
        # table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

        # table_stand_height = 0.1
        # table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        # table_stand_opts = gymapi.AssetOptions()
        # table_stand_opts.fix_base_link = True
        # table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # table_stand_start_pose = gymapi.Transform()
        # table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        # table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # cube pos + properties
        cube_size = 0.07
        cube_options = gymapi.AssetOptions()
        cube_options.density = 1000
        cube_assets = [self.gym.create_box(self.sim, *([cube_size] * 3), cube_options) for _ in range(self.n_cubes)]
        self.cube_colors = [gymapi.Vec3(0.0, np.random.rand(), np.random.rand()) for _ in range(self.n_cubes)]

        # defining start poses, which will get reset later
        cube_start_pose = gymapi.Transform()
        cube_init_pos = np.array([0, 0, cube_size]) # + self._table_surface_pos
        cube_start_pose.p = gymapi.Vec3(*cube_init_pos)
        cube_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self._cube_ids = [None] * self.n_cubes
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            
            # add camera
            cam_props = gymapi.CameraProperties()
            cam_props.enable_tensors = True
            cam_handle = self.gym.create_camera_sensor(env, cam_props)
            self.gym.set_camera_location(cam_handle, env, gymapi.Vec3(0.12, 0, 2), gymapi.Vec3(0, 0, 1))# gymapi.Vec3(5, 1, 0), gymapi.Vec3(0, 1, 0))
            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
            self.cam_tensors.append(gymtorch.wrap_tensor(cam_tensor))

            # table_actor = self.gym.create_actor(env, table_asset, table_start_pose, "table", i, 1, 0)
            # table_stand_actor = self.gym.create_actor(env, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)
            for j in range(self.n_cubes):
                self._cube_ids[j] = self.gym.create_actor(env, cube_assets[j], cube_start_pose, "cube", i, 0, 0)
                self.gym.set_rigid_body_color(env, self._cube_ids[j], 0, gymapi.MESH_VISUAL, self.cube_colors[j])
            
            self.envs.append(env)
        
    
    def init_data(self):
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        for j in range(self.n_cubes):
            self._init_cube_states[j] = torch.zeros(self.num_envs, 13, device=self.device)
            self._cube_states[j] = self._root_state[:, self._cube_ids[j], :]


    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        for j in range(self.n_cubes):
            self._reset_init_cube_state(cube=j, env_ids=env_ids, check_valid=j>0)
            # Write these new init states to the sim states
            self._cube_states[j][env_ids] = self._init_cube_states[j][env_ids]
        

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
        # table_center = torch.tensor(self._table_surface_pos[:3], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        # table_center[2] = table_center[2] + 0.15

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
                sampled_cube_state[active_idx, :3] = torch.tensor([0, 0, 0.15], device=self.device) + \
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
            #table_center.unsqueeze(0) + \
            sampled_cube_state[:, :3] = torch.tensor([[0, 0, 0.15]], device=self.device) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 3, device=self.device) - 0.5)
        sampled_cube_state[:,2] += 0.25
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])
        
        this_cube_state_all[env_ids, :] = sampled_cube_state

        breakpoint()

    
    def refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self._update_states()

    
    def _update_states(self):
        for j in range(self.n_cubes):
            self.states.update({
                f"cube{j}_vel": self._cube_states[j][:, 7:10],
                f"cube{j}_quat": self._cube_states[j][:, 3:7],
                f"cube{j}_pos": self._cube_states[j][:, :3],
            })

    def step(self):
        self.gym.simulate(self.sim)
        
        
    def render(self, filename=None):
        self.gym.fetch_results(self.sim, True)

        # refresh state data in the tensor
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.step_graphics(self.sim)

        # render sensors and refresh camera tensors
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # TODO: fix this
        img = self.cam_tensors[0].cpu().numpy()
        if filename is not None:
            imageio.imwrite(filename, img)
        
        self.gym.end_access_image_tensors(self.sim)

        return img


    def freeze_cubes(self):
        if not self.frozen:
            # Freeze
            for i, env in enumerate(self.envs):
                for id in self._cube_ids:
                    # Set mass x100
                    properties = self.gym.get_actor_rigid_body_properties(env, id)[0]
                    if self.cube_mass is None: self.cube_mass = properties.mass
                    properties.mass = self.cube_mass * 100
                    self.gym.set_actor_rigid_body_properties(env, id, [properties], True)

                    # Set color to gray
                    num_bodies = self.gym.get_actor_rigid_body_count(env, id)
                    for n in range(num_bodies):
                        self.gym.set_rigid_body_color(env, id, n, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0.5, 0.5))
        else:
            # Restore
            for i, env in enumerate(self.envs):
                for j, id in enumerate(self._cube_ids):
                    # Set mass x100
                    properties = self.gym.get_actor_rigid_body_properties(env, id)[0]
                    if self.cube_mass is None: self.cube_mass = properties.mass
                    properties.mass = self.cube_mass
                    self.gym.set_actor_rigid_body_properties(env, id, [properties], True)

                    # Set color to gray
                    num_bodies = self.gym.get_actor_rigid_body_count(env, id)
                    for n in range(num_bodies):
                        self.gym.set_rigid_body_color(env, id, n, gymapi.MESH_VISUAL, self.cube_colors[i])

        self.frozen = not self.frozen