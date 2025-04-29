import pybullet as p
from visual_tools.load_one_state import create_cmu_tpose
from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
import os


# Start PyBullet
p.connect(p.GUI)

# Load two skeleton states
skeleton1 = create_cmu_tpose('tpose/smpl_dart_tpose.npy')
skeleton2 = create_cmu_tpose('tpose/g1_tpose.npy')

# Extract data from skeleton 1
global_translation_1 = skeleton1.global_translation
global_rotation_1 = skeleton1.global_rotation
parent_indices_1 = skeleton1.skeleton_tree.parent_indices.numpy()

# Extract data from skeleton 2
global_translation_2 = skeleton2.global_translation
global_rotation_2 = skeleton2.global_rotation
parent_indices_2 = skeleton2.skeleton_tree.parent_indices.numpy()

# Apply offset to the second skeleton (to separate the visualization)
offset = np.array([0.0, 0.0, 0.0])  # shift 0.5m on x-axis

# === Visualize skeleton 1 ===
for pos, rot in zip(global_translation_1, global_rotation_1):
    pos = pos.tolist()
    rot_matrix = R.from_quat(rot).as_matrix()
    
    # Joint point
    p.addUserDebugPoints([pos], [[1, 0, 0]], 8.0, 0)
    
    # Local axes
    for axis, color in zip(rot_matrix.T, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        p.addUserDebugLine(pos, (np.array(pos) + axis * 0.1).tolist(), color, 1.0, 0)

# Skeleton 1 parent-child links
for i in range(len(parent_indices_1)):
    if parent_indices_1[i] != -1:
        p.addUserDebugLine(
            global_translation_1[parent_indices_1[i]].tolist(),
            global_translation_1[i].tolist(),
            [0, 0, 0], 2.0, 0  # black lines
        )

# === Visualize skeleton 2 (with offset) ===
for pos, rot in zip(global_translation_2, global_rotation_2):
    pos_offset = (np.array(pos) + offset).tolist()
    rot_matrix = R.from_quat(rot).as_matrix()
    
    # Joint point
    p.addUserDebugPoints([pos_offset], [[0, 0, 1]], 8.0, 0)  # blue points
    
    # Local axes
    for axis, color in zip(rot_matrix.T, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        p.addUserDebugLine(pos_offset, (np.array(pos_offset) + axis * 0.1).tolist(), color, 1.0, 0)

# Skeleton 2 parent-child links (with offset)
for i in range(len(parent_indices_2)):
    if parent_indices_2[i] != -1:
        parent_pos = global_translation_2[parent_indices_2[i]] + offset
        child_pos = global_translation_2[i] + offset
        p.addUserDebugLine(
            parent_pos.tolist(),
            child_pos.tolist(),
            [0.5, 0, 0.5], 2.0, 0  # purple lines
        )

# Keep the simulation running
while True:
    pass
