import numpy as np

from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

def cube_projection(size, center, quat):
    """
    Args:
        center: (3,) nparray
        size: float
        quat: (4,) nparray
    Returns:
        x, y coordinates of convex hull of the projection onto xy plane
    """
    r = Rotation.from_quat(quat).as_matrix()
    vertices = [(x, y, z) for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
    vertices = np.array(vertices).reshape(8, 3).T
    vertices = (size / 2) * r @ vertices
    vertices = vertices + center.reshape(3, 1)
    proj = vertices[:2].T
    hull = ConvexHull(proj)
    return proj[hull.vertices]

def add_cubes_to_viz(obs, env, ax: Axes):
    for j in range(env.n_cubes_train):
        pos_idx = env.obs_buf_idx[f'cube{j}_pos']
        quat_idx = env.obs_buf_idx[f'cube{j}_quat']
        size = env.cube_sizes[j]
        pos = obs[0, pos_idx[0] : pos_idx[1]].cpu().numpy()
        quat = obs[0, quat_idx[0] : quat_idx[1]].cpu().numpy()
        hull_vertices = cube_projection(size, pos, quat)
        poly = Polygon(hull_vertices)
        ax.add_patch(poly)
        poly.set_facecolor('gray')
        poly.set_edgecolor('black')
        poly.set_alpha(0.3)