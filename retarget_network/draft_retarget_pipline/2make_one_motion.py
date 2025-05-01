
import torch
import numpy as np
from skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
 
def build_skeleton_tree():
    joint_names = [
        'pelvis', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_link', 
        'right_hip_yaw_link', 'right_knee_link', 'right_ankle_link', 'torso_link', 
        'left_shoulder_pitch_link', 'left_elbow_link', 'left_hand_link', 
        'right_shoulder_pitch_link', 'right_elbow_link', 'right_hand_link'
    ]
    parent_indices = torch.tensor([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  7, 11, 12])
    
    local_translation = torch.tensor([[ 0.0000e+00,  0.0000e+00,  1.1000e+00],
       [ 0.0000e+00,  8.7500e-02, -1.7420e-01],
       [ 0.0000e+00,  0.0000e+00, -4.0000e-01],
       [ 0.0000e+00,  0.0000e+00, -4.0000e-01],
       [ 0.0000e+00, -8.7500e-02, -1.7420e-01],
       [ 0.0000e+00,  0.0000e+00, -4.0000e-01],
       [ 0.0000e+00,  0.0000e+00, -4.0000e-01],
       [ 4.8900e-04,  2.7970e-03,  2.0484e-01],
       [ 5.5000e-03,  1.5535e-01,  4.2999e-01],
       [ 1.8500e-02,  0.0000e+00, -1.9800e-01],
       [ 3.0000e-01,  0.0000e+00,  0.0000e+00],
       [ 5.5000e-03, -1.5535e-01,  4.2999e-01],
       [ 1.8500e-02,  0.0000e+00, -1.9800e-01],
       [ 3.0000e-01,  0.0000e+00,  0.0000e+00]], dtype=torch.float32)
    
    return SkeletonTree(joint_names, parent_indices, local_translation)
 
def create_motion_from_npy(npy_path, fps=30, save_path='data/retarget_npy/predicted_data1.npy'):
    data = np.load(npy_path, allow_pickle=True).item()
    quat = data['local_rotation'].cpu()
    root = data['root_pos'].cpu()
 
    skeleton_tree = build_skeleton_tree()
    skeleton_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton_tree,
        r=quat,
        t=root,
        is_local=True
    )
 
    skeleton_motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=fps)
    skeleton_motion.to_file(save_path)
    print(f"Saved to {save_path}")
 
if __name__ == "__main__":
    create_motion_from_npy("data/retarget_npy/predicted_retarget.npy")
 
 