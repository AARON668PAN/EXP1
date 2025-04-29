import pybullet as p
from visual_tools.load_one_state import create_cmu_tpose
from scipy.spatial.transform import Rotation as R



# 启动 PyBullet 仿真
p.connect(p.GUI)


skeletonstate = create_cmu_tpose('tpose/h1_tpose.npy')

# root_translation = skeletonstate.root_translation.numpy()
# root_quaternion = skeletonstate.rotation[0].numpy()


global_translation = skeletonstate.global_translation

global_rotation = skeletonstate.global_rotation

parent_indices = skeletonstate.skeleton_tree.parent_indices.numpy()




# 可视化点和方向
for position, rotation in zip(global_translation, global_rotation):
    # 可视化点
    p.addUserDebugPoints(
        pointPositions=[position.tolist()],
        pointColorsRGB=[[1, 0, 0]],  # 红色点
        pointSize=8.0,
        lifeTime=0
    )

    # 可视化方向 (使用 Z 轴方向)
    rot_matrix = R.from_quat(rotation).as_matrix()
    x_axis_world = rot_matrix[:, 0]  # X 轴方向
    y_axis_world = rot_matrix[:, 1]  # Y 轴方向
    z_axis_world = rot_matrix[:, 2]  # Z 轴方向


    line_end = position + z_axis_world * 0.1  # 缩放方向线长度
    p.addUserDebugLine(
        lineFromXYZ=position.tolist(),
        lineToXYZ=line_end.tolist(),
        lineColorRGB=[0, 0, 1],  # 蓝色方向线
        lineWidth=1.0,
        lifeTime=0
    )

    # 绘制 Y 轴 (绿色)
    y_end = position + y_axis_world * 0.1  # 放大 Y 轴方向线
    p.addUserDebugLine(
        lineFromXYZ=position.tolist(),
        lineToXYZ=y_end.tolist(),
        lineColorRGB=[0, 1, 0],  # 绿色
        lineWidth=1.0,
        lifeTime=0
    )

    x_end = position + x_axis_world * 0.1  # 放大 X 轴方向线
    p.addUserDebugLine(
        lineFromXYZ=position.tolist(),
        lineToXYZ=x_end.tolist(),
        lineColorRGB=[1, 0, 0],  # 红色
        lineWidth=1.0,
        lifeTime=0
    )

# 可视化父子节点的连线
for i in range(len(parent_indices)):
    if parent_indices[i] != -1:  # 根节点没有父节点
        p.addUserDebugLine(
            lineFromXYZ=global_translation[parent_indices[i]].tolist(),
            lineToXYZ=global_translation[i].tolist(),
            lineColorRGB=[0, 0, 0],  # 紫色连线
            lineWidth=2.0,
            lifeTime=0
        )


# 保持仿真运行
while True:
    pass