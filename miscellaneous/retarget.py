from skeleton.skeleton3d import SkeletonState, SkeletonTree, SkeletonMotion
import torch
import numpy as np
from core import rotation3d
import time


#TODO:
#Step1
# make one source tpose (first check initial pose of dart(mdm)) ok
joint_names = [
    'Pelvis',                    # 0
    'L_Hip',        # 1
    'R_Hip',        # 2
    'Spine1',         # 3
    'L_Knee',           # 4
    'R_Knee',          # 5
    'Spine2',         # 6
    'L_Ankle', # 7
    'R_Ankle', # 8
    'Spine3', #9
    'L_Foot', #10
    'R_Foot', #11
    'Neck', #12
    'L_Collar', #13
    'R_Collar', #14
    'Head', #15
    'L_Shoulder', #16
    'R_Shoulder', #17
    'L_Elbow', #18
    'R_Elbow', #19
    'L_Wrist', #20
    'R_Wrist', #21
]

parent_indices = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])

local_translation = torch.tensor([
        [-0.00217368, -0.24078919,  0.02858379], #'Pelvis'
        [ 0.05858135, -0.08228007, -0.01766408], # 'L_Hip'
        [-0.06030973, -0.09051332, -0.01354253],#'R_Hip'
        [ 0.00443945,  0.12440356, -0.03838522],#'Spine1'
        [ 0.04345143, -0.3864696 ,  0.008037  ], # 'L_Knee'
        [-0.04325663, -0.3836879 , -0.00484304],#'R_Knee'
        [ 0.00448844,  0.1379564 ,  0.02682032],#'Spine2'
        [-0.01479033, -0.42687446, -0.03742799], # 'L_Ankle'
        [ 0.01905555, -0.42004555, -0.03456167],#'R_Ankle'
        [-0.00226459,  0.0560324 ,  0.00285505],#'Spine3'
        [ 0.04105436, -0.06028578,  0.12204242],#'L_Foot'
        [-0.03483988, -0.06210563,  0.1303233 ],#'R_Foot'
        [-0.01339018,  0.21163554, -0.03346758],#'#Neck'
        [ 0.07170247,  0.11399969, -0.01889817],#'L_Collar'
        [-0.08295365,  0.11247235, -0.02370739],#'R_Collar'
        [ 0.01011321,  0.08893741,  0.05040986],#'Head'
        [ 0.12292139,  0.04520511, -0.019046  ],#'L_Shoulder'
        [-0.11322831,  0.04685327, -0.00847207],#'R_Shoulder'
        [ 0.25533187, -0.01564904, -0.02294649],#'L_Elbow'
        [-0.26012748, -0.0143693 , -0.03126873],#'R_Elbow'
        [ 0.26570928,  0.01269813, -0.00737473],#'L_Wrist'
        [-0.2691084 ,  0.00679374, -0.00602677],#'R_Wrist'
        ], dtype=torch.float32)

skeleton_tree = SkeletonTree(parent_indices=parent_indices,
                            local_translation=local_translation,
                            node_names=joint_names)

# [0.5,0.5,0.5,0.5]
local_rotation = torch.tensor([[ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1],
        [ 0,  0,  0,  1]],dtype=torch.float32)

root_translation = torch.tensor([0. , 0. , 0.9], dtype=torch.float32)

source_tpose = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree=skeleton_tree,
    r=local_rotation,
    t=root_translation,
    is_local=True
)


# source_tpose.to_file('retarget_npy/source_tpose.npy')

# make one target tpose (first check initial pose of h1)    ok
joint_names = ['pelvis', 
                'left_hip_yaw_link', 
                'left_knee_link', 
                'left_ankle_link', 
                'right_hip_yaw_link', 
                'right_knee_link', 
                'right_ankle_link', 
                'torso_link', 
                'left_shoulder_pitch_link', 
                'left_elbow_link', 
                'left_hand_link', 
                'right_shoulder_pitch_link', 
                'right_elbow_link', 
                'right_hand_link']

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

skeleton_tree = SkeletonTree(parent_indices=parent_indices,
                            local_translation=local_translation,
                            node_names=joint_names)

local_rotation = torch.tensor([[ 0.       ,  0.       ,  0.       ,  1.       ],
    [ 0.       ,  0.       ,  0.       ,  1.       ],
    [ 0.       ,  0.       ,  0.       ,  1.       ],
    [ 0.       ,  0.       ,  0.       ,  1.       ],
    [ 0.       ,  0.       ,  0.       ,  1.       ],
    [ 0.       ,  0.       ,  0.       ,  1.       ],
    [ 0.       ,  0.       ,  0.       ,  1.       ],
    [ 0.       ,  0.       ,  0.       ,  1.       ],
    [ 0.7071068,  0.       ,  0.       ,  0.7071068],
    [ 0.       ,  0.7071068,  0.       ,  0.7071068],
    [ 0.       ,  0.       ,  0.       ,  1.       ],
    [-0.7071068,  0.       ,  0.       ,  0.7071068],
    [ 0.       ,  0.7071068,  0.       ,  0.7071068],
    [ 0.       ,  0.       ,  0.       ,  1.       ]],dtype=torch.float32)

root_translation = torch.tensor([0. , 0. , 0.9], dtype=torch.float32)

target_tpose = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree=skeleton_tree,
    r=local_rotation,
    t=root_translation,
    is_local=True
)



# load raw data to make motion
mdm_data_path = 'data/processed_data/dart_walk.npy'
raw_data = np.load(mdm_data_path, allow_pickle=True).item()

source_local_rot = raw_data['local_rotation'].cpu()  # [F, 22, 4]
source_root_trans = raw_data['root_pos'].cpu()   # [F, 3]




#Step2
# make the source state class
source_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=source_tpose.skeleton_tree,
            r=source_local_rot,
            t=source_root_trans,
            is_local=True,
        )

global_rotation = source_state.global_rotation

# # Rdiff = R_g_source * inv(source tpose)    Rdiff is global
# global_rotation_diff = rotation3d.quat_mul_norm(
#             source_state.global_rotation, rotation3d.quat_inverse(source_tpose.global_rotation)
#         )


#Step3
# select Rdiff from source into target (remove some Rdiff we don't need) 
# rearrange the order of joint angles in the sequence of target skeleton tree
original_joint_names = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle',
    'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head',
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
]

desired_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee',
    'R_Ankle', 'Spine1', 'L_Shoulder', 'L_Elbow',
    'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'
]  


desired_indices = [original_joint_names.index(name) for name in desired_joint_names]
new_global_rotation = global_rotation[:, desired_indices, :]



# # implement Rdiff onto target tpose
# new_global_rotation = rotation3d.quat_mul_norm(
#             global_rotation_diff, target_tpose.global_rotation
#         )


# #Step4
# # make the target state class
target_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree = target_tpose.skeleton_tree,
            r=new_global_rotation,
            t= source_state.root_translation,
            is_local=False,
        )




# target_state = SkeletonState.from_rotation_and_root_translation(
#             skeleton_tree = target_tpose.skeleton_tree,
#             r= target_state.local_rotation,
#             t= source_state.root_translation,
#             is_local=True,
#         )



target_motion = SkeletonMotion.from_skeleton_state(target_state, fps=30)
target_motion.to_file('data/retarget_npy/MDM_motion_walk.npy')

