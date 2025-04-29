# import numpy as np

# # Path to your .npy file
# file_path = 'tpose/results_WholeData.npy'

# # Load the .npy file
# data = np.load(file_path, allow_pickle=True)
# data = data.item()

# data = data['motion_cont6d']

# # # Print shape and a preview
# # print("Shape:", data['motion_cont6d'].shape)

# transl = data[:,0:3]

# poses_6d = data[:,3:135]

# transl_delta = data[:,135:138]

# global_orient_delta_6d = data[:,138:144]

# joints = data[:,144:210]

# joints_delta = data[:,210:276]


# joint = joints[0]

# joint = joint.reshape(22, 3)

# joints_delta = joints_delta[0]

# joints_delta = joints_delta.reshape(22, 3)



# import torch

# def euler_to_quaternion(euler_angles):
#     """
#     euler_angles: tensor of shape [N, 3], angles in radians
#                   order: roll (X), pitch (Y), yaw (Z)
#     return: tensor of shape [N, 4], quaternion (x, y, z, w)
#     """
#     roll = euler_angles[:, 0]
#     pitch = euler_angles[:, 1]
#     yaw = euler_angles[:, 2]

#     cy = torch.cos(yaw * 0.5)
#     sy = torch.sin(yaw * 0.5)
#     cp = torch.cos(pitch * 0.5)
#     sp = torch.sin(pitch * 0.5)
#     cr = torch.cos(roll * 0.5)
#     sr = torch.sin(roll * 0.5)

#     w = cr * cp * cy + sr * sp * sy
#     x = sr * cp * cy - cr * sp * sy
#     y = cr * sp * cy + sr * cp * sy
#     z = cr * cp * sy - sr * sp * cy

#     quat = torch.stack([x, y, z, w], dim=1)  # shape [N, 4]
#     return quat

# quat = euler_to_quaternion(joints_delta)
# print(quat.shape)  # [22, 4]
# print(quat)

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# joints 数据
joints = torch.tensor([
    [ 5.3877e-04,  7.0095e-05,  1.0992e-04], # plevis
    [-5.6419e-02, -1.3627e-02, -9.6132e-02], # left hip
    [ 5.8232e-02, -1.3622e-02, -1.0532e-01], # right hip
    [-1.6256e-03, -4.4969e-02,  1.0772e-01], # spine 1
    [-1.0702e-01, -4.9822e-02, -4.9189e-01], # left knee
    [ 9.4867e-02, -6.4157e-02, -4.8343e-01], # right knee
    [-1.4513e-02, -5.3351e-02,  2.5937e-01], # spine 2
    [-4.9839e-02, -7.4783e-02, -9.1270e-01], # left ankle
    [ 6.8873e-02, -8.9361e-02, -9.2022e-01], # right ankle
    [-2.5985e-03, -3.4053e-02,  3.1895e-01], # spine 3
    [-1.0358e-01,  5.4216e-02, -9.7104e-01], # left foot
    [ 1.3230e-01,  3.7276e-02, -9.8018e-01], # right foot
    [ 5.3932e-03, -5.3974e-02,  4.8335e-01], # neck
    [-4.4332e-02, -3.0110e-02,  3.9455e-01], # left suo gu
    [ 4.2843e-02, -4.4547e-02,  3.9441e-01], # right suo gu
    [ 1.5006e-03, -1.2976e-02,  6.4351e-01], # head
    [-1.9763e-01, -3.4226e-02,  3.9745e-01], # left shoulder
    [ 1.8278e-01, -7.7988e-02,  3.9129e-01], # right shoulder
    [-2.0713e-01, -7.2914e-02,  1.3187e-01], # left elbow
    [ 2.1207e-01, -9.0011e-02,  1.1607e-01], # right elbow
    [-2.3361e-01,  5.8752e-02, -1.0424e-01], # left hand
    [ 2.4822e-01,  1.9096e-02, -1.2105e-01], # right hand
], device='cuda:0')

# 转为cpu numpy
joints = joints.cpu().numpy()

# 画图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='blue')

# 设置固定比例
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# 调用设置
set_axes_equal(ax)

# Label
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Joints (Fixed Aspect Ratio)')
ax.view_init(elev=20, azim=60)  # 设定观察角度
plt.show()
