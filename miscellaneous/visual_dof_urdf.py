import numpy as np
import pybullet as p
import pybullet_data
import time

# 加载 DOF 数据
dof_map = np.load('data/retarget_npy/NN_dof/dof_map.npy')   # (133,29)

# 启动 PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
robot_id = p.loadURDF("G1_mimic/resources/robots/g1/g1_29dof.urdf", useFixedBase=True)

# 扫描所有 joints，过滤出非 fixed（即真正的 DOF） joints
dof_joint_indices = []
dof_joint_names = []
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    joint_type = info[2]
    if joint_type != p.JOINT_FIXED:
        dof_joint_indices.append(i)
        dof_joint_names.append(info[1].decode())

print("Detected DOF joints (count={}):".format(len(dof_joint_indices)), dof_joint_names)
# 确认应该是 29 个：
assert len(dof_joint_indices) == 29

# 播放
fps = 30
for frame in dof_map:
    for idx, angle in zip(dof_joint_indices, frame):
        p.resetJointState(robot_id, idx, angle)
    p.stepSimulation()
    time.sleep(1.0/fps)
