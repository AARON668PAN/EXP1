from isaacgym import gymapi
from isaacgym import gymtorch
import os

import torch
import numpy as np

from MotionLib.motion_lib import MotionLib
from utils import torch_utils



_dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                    4, 5, 6,
                    7,       # Torso
                    8, 9, 10, # Shoulder, Elbow, Hand
                    11, 12, 13]  # 13

_dof_offsets = [0, 3, 4, 5, 8, 9, 10, 
                    11, 
                    14, 15, 16, 19, 20, 21]  # 14

_dof_offsets_29 = [0, 3, 4, 6, 9, 10, 12, 
                    15, 
                    18, 19, 22, 25, 26, 29]  # 14

_key_body_ids = [0, 1, 2, 3, # Hip, Knee, Ankle
                    4, 5, 6,
                    7,       # Torso
                    8, 9, 10, # Shoulder, Elbow, Hand
                    11, 12, 13]  # 13



file_path = 'data/retarget_npy/walking_smpl_out.npy'

motion_lib_class = MotionLib(motion_file=file_path,
                             dof_body_ids=_dof_body_ids,
                             dof_offsets=_dof_offsets_29,
                             key_body_ids=_key_body_ids, 
                             device='cuda', 
                             no_keybody=True)

motion_id = torch.tensor([0], device='cuda')

# 获取动作的帧数（总时长）
motion_length = motion_lib_class.get_motion_length(motion_id)[0].item()
fps = motion_lib_class.get_motion_fps(motion_id)[0].item()
dt = 1.0 / fps
num_frames = motion_lib_class.get_motion_num_frames(motion_id)[0].item()

# 生成等间距的时间戳
motion_times = torch.arange(0, motion_length, dt, device='cuda')  # (T,)
motion_ids = motion_id.repeat(len(motion_times))  # (T,)

print(motion_times.shape)
# 获取整段动作的状态（包括根位置、旋转、关节角度等）
root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = \
    motion_lib_class.get_motion_state(motion_ids, motion_times)


print(root_pos.shape)

output = {
        'root_trans_offset': root_pos,  
        'pose_aa': key_pos,             
        'dof': dof_pos,
        'root_rot': root_rot,
        'fps': 30,
        'contact_states': 0
    }


# 保存输出数据
save_path = 'data/processed_data/g1_dof29_data.npy'
np.save(save_path, output)
print("Saved to", save_path)


print(dof_pos[0])