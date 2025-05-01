from isaacgym.torch_utils import *
import torch
import json
import numpy as np

from core.rotation3d import *
from skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

"""
This scripts shows how to retarget a motion clip from the source skeleton to a target skeleton.
Data required for retargeting are stored in a retarget config dictionary as a json file. This file contains:
  - source_motion: a SkeletonMotion npy format representation of a motion sequence. The motion clip should use the same skeleton as the source T-Pose skeleton.
  - target_motion_path: path to save the retargeted motion to
  - source_tpose: a SkeletonState npy format representation of the source skeleton in it's T-Pose state
  - target_tpose: a SkeletonState npy format representation of the target skeleton in it's T-Pose state (pose should match source T-Pose)
  - joint_mapping: mapping of joint names from source to target
  - rotation: root rotation offset from source to target skeleton (for transforming across different orientation axes), represented as a quaternion in XYZW order.
  - scale: scale offset from source to target skeleton
"""



def main():
    # load retarget config
    retarget_data_path = "configs/retarget_smpl_to_h1.json"
    with open(retarget_data_path) as f:
        retarget_data = json.load(f)
    # load and visualize t-pose files
    source_tpose = SkeletonState.from_file(retarget_data["source_tpose"])


    target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])

    # load and visualize source motion sequence
    source_motion = SkeletonMotion.from_file(retarget_data["source_motion"])


    # parse data from retarget config
    joint_mapping = retarget_data["joint_mapping"]
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])

    print("run retargeting")
    # run retargeting
    target_motion = source_motion.retarget_to_by_tpose(
      joint_mapping=retarget_data["joint_mapping"],
      source_tpose=source_tpose,
      target_tpose=target_tpose,
      rotation_to_target_skeleton=rotation_to_target_skeleton,
      scale_to_target_skeleton=retarget_data["scale"]
    )



    # keep frames between [trim_frame_beg, trim_frame_end - 1]
    frame_beg = retarget_data["trim_frame_beg"]
    frame_end = retarget_data["trim_frame_end"]
    if (frame_beg == -1):
        frame_beg = 0
        
    if (frame_end == -1):
        frame_end = target_motion.local_rotation.shape[0]
        
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]
      

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation
    min_h = torch.min(tar_global_pos[..., 2])
    # root_translation[:, 2] += -min_h
    



    rotation_to_target_skeleton = torch.tensor([0.0, -0.7071, 0.0, 0.7071], device='cpu')
    local_rotation[..., 0, :] = quat_mul_norm(
        rotation_to_target_skeleton, local_rotation[..., 0, :]
    )
    root_translation = quat_rotate(rotation_to_target_skeleton, root_translation)

    root_translation[:, 2] = root_translation[:, 2] + 1.85


    # src = local_rotation[:, 0, :]  # 取第0个关节，shape (num_frames, 4)
    # # 定义绕Y轴90°旋转的四元数，注意是 (x, y, z, w) 排列！
    # q_y90 = torch.tensor([0.0, -0.7071, 0.0, 0.7071], device=src.device).expand_as(src)  # (num_frames, 4)
    # # 批量四元数乘
    # new_src = quat_mul(q_y90, src)
    # # 更新回去
    # local_rotation[:, 0, :] = new_src


    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # save retargeted motion
    target_motion.to_file(retarget_data["target_motion_path"])


    
    return

if __name__ == '__main__':
    main()