
import os
import numpy as np
import torch
from skeleton.quaternion import cont6d_to_matrix
from skeleton import rotation_conversions as rc
import time

# 0 pelvis 3torso 6torso  9neck 


def cont6d_to_quat_perframe(cont6d_params: torch.Tensor) -> torch.Tensor:
    """
    将单帧的 cont6d 参数（形状 [22, 6]）转换为四元数表示（形状 [22, 4]）。
    假设 cont6d_to_matrix 将每个关节的 6D 参数转换成 3x3 旋转矩阵，
    而 rc.matrix_to_quaternion 接受 3x3 矩阵，返回四元数（格式假定为 [w, x, y, z]），
    然后重排成 [x, y, z, w]。
    """
    quats = []
    num_joints = cont6d_params.shape[0]
    for j in range(num_joints):
        # 将第 j 个关节的 6D 参数转换为 3x3 旋转矩阵
        R = cont6d_to_matrix(cont6d_params[j, :])
        # 转换成四元数（返回格式 [w, x, y, z]）
        q = rc.matrix_to_quaternion(R)
        # 重排列顺序为 [x, y, z, w]
        q_rearr = q[[1, 2, 3, 0]]
        quats.append(q_rearr)
    return torch.stack(quats, dim=0)  # 形状 [22, 4]

import torch

def invert_quaternion(quat):
    """
    quat: tensor of shape [N, 4], [x, y, z, w] format
    return: tensor of shape [N, 4], inverted quaternions
    """
    quat_inv = quat.clone()
    quat_inv[:, :3] = -quat_inv[:, :3]  # Negate x, y, z
    return quat_inv

def main():

    # Path to your .npy file
    file_path = 'data/raw_data/all_train_data/final_result7.npy'

    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    data = data.item()

    data = data['motion_cont6d']

    transl = data[:,0:3]

    poses_6d = data[:,3:135]
    poses_6d = poses_6d.reshape(poses_6d.shape[0], 22, 6)

    # transl_delta = data[:,135:138]

    # global_orient_delta_6d = data[:,138:144]

    # joints = data[:,144:210]

    # joints_delta = data[:,210:276]


    # 将每帧的 cont6d 转换成对应的四元数，得到形状 [T, 22, 4]
    quat_list = []
    n_frame = poses_6d.shape[0]
    

    for frame in range(n_frame):
        frame_cont6d = poses_6d[frame]  # [22, 6]
        
        quat_tensor = cont6d_to_quat_perframe(frame_cont6d)  # [22, 4]
        quat_tensor =  invert_quaternion(quat_tensor)
        quat_list.append(quat_tensor)
    processed_quat = torch.stack(quat_list, dim=0)  # [T, 22, 4]

    # print(processed_quat[0])

    root_pos = transl
    # # 构造输出字典
    output = {
        'local_rotation': processed_quat,  # [T, 22, 4] 的 torch.Tensor
        'root_pos': root_pos,              # [T, 3] 的 torch.Tensor
    }

 
    # 保存输出数据
    os.makedirs('processed_data', exist_ok=True)
    save_path = 'data/processed_data/all_train_data/final_result7.npy'
    np.save(save_path, output)
    print("Saved to", save_path)


if __name__ == "__main__":
    main()
