from isaacgym.torch_utils import *
import torch
import pybullet as p
from scipy.spatial.transform import Rotation as R
from skeleton.skeleton3d import SkeletonState, SkeletonTree, SkeletonMotion
import time
import sys


def load_sk_motion():
    retarget_motion_path = 'data/retarget_npy/walking_smpl_out.npy'
    motion = SkeletonMotion.from_file(retarget_motion_path)
    return motion

# Connect to PyBullet with GUI
p.connect(p.GUI)

skeletonstate = load_sk_motion()
global_translation_list = skeletonstate.global_translation
global_rotation_list = skeletonstate.global_rotation
parent_indices = skeletonstate.skeleton_tree.parent_indices.numpy()

# for frame_index in range(len(global_translation_list)):
for frame_index in range(len(global_translation_list)):

    # Remove previous frame's debug items
    p.removeAllUserDebugItems()
    
    # Disable rendering so all debug items are added together invisibly
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    
    global_translation = global_translation_list[frame_index]
    global_rotation = global_rotation_list[frame_index]
    
    
    # --- Batch add all local axes (debug lines) ---
    for position, rotation in zip(global_translation, global_rotation):
        pos_list = position.tolist()
        rot_matrix = R.from_quat(rotation).as_matrix()
        
        # Compute line endpoints for each local axis, scaled to 0.1
        x_line_end = (position + rot_matrix[:, 0] * 0.1).tolist()
        y_line_end = (position + rot_matrix[:, 1] * 0.1).tolist()
        z_line_end = (position + rot_matrix[:, 2] * 0.1).tolist()
        
        # Draw X-axis (red)
        p.addUserDebugLine(lineFromXYZ=pos_list,
                           lineToXYZ=x_line_end,
                           lineColorRGB=[1, 0, 0],
                           lineWidth=2.0,
                           lifeTime=0)
        # Draw Y-axis (green)
        p.addUserDebugLine(lineFromXYZ=pos_list,
                           lineToXYZ=y_line_end,
                           lineColorRGB=[0, 1, 0],
                           lineWidth=2.0,
                           lifeTime=0)
        # Draw Z-axis (blue)
        p.addUserDebugLine(lineFromXYZ=pos_list,
                           lineToXYZ=z_line_end,
                           lineColorRGB=[0, 0, 1],
                           lineWidth=2.0,
                           lifeTime=0)
    
    # --- Add parent-child connection lines ---
    for idx, parent_idx in enumerate(parent_indices):
        if parent_idx != -1:  # Skip joints with no parent
            parent_pos = global_translation[parent_idx].tolist()
            child_pos = global_translation[idx].tolist()
            p.addUserDebugLine(lineFromXYZ=parent_pos,
                               lineToXYZ=child_pos,
                               lineColorRGB=[0, 1, 0],
                               lineWidth=2.0,
                               lifeTime=0)
    
    # Re-enable rendering so the entire frame is drawn at once
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    time.sleep(0.05)

# Keep the simulation running
while True:
    pass
