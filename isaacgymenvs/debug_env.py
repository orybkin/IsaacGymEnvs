""" 
Debug headless camera rendering
Author: Oleg 
"""


import imageio
from isaacgym import gymapi
from isaacgym import gymtorch
import torch


gym = gymapi.acquire_gym()

# Config    
sim_params = gymapi.SimParams()
sim_params.physx.solver_type = 1
sim_params.physx.num_threads = 0
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True

cam_props = gymapi.CameraProperties()
cam_props.enable_tensors = True

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)


for i in range(2):
    # create env
    env = gym.create_env(sim, gymapi.Vec3(-2.0, 0.0, -2.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)

    # add camera
    # cam_handle = gym.create_camera_sensor(env, cam_props)
    # gym.set_camera_location(cam_handle, env, gymapi.Vec3(5, 1, 0), gymapi.Vec3(0, 1, 0))

    # obtain camera tensor
    # cam_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)

    # wrap camera tensor in a pytorch tensor
    # torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)

    # prepare tensor access
    gym.prepare_sim(sim)

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh state data in the tensor
    gym.refresh_rigid_body_state_tensor(sim)
    # gym.step_graphics(sim)

    # render sensors and refresh camera tensors
    # gym.render_all_camera_sensors(sim)
    # gym.start_access_image_tensors(sim)

    # imageio.imwrite("render.png" , torch_cam_tensor.cpu().numpy())

    # import pdb; pdb.set_trace()

    # gym.end_access_image_tensors(sim)

