from skeleton.skeleton3d import SkeletonState, SkeletonTree
import torch
import numpy as np

def create_g1_tpose():
    # 1. 创建骨骼树结构
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


    # 2. 创建骨骼树
    skeleton_tree = SkeletonTree(parent_indices=parent_indices,
                               local_translation=local_translation,
                               node_names=joint_names)


    local_rotation = torch.tensor(
    [   [ 0,  0.,  0, 1],
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
        [ 0,  0,  0,  1],],dtype=torch.float32)

    # 4. 创建根节点平移（与H1相同）
    root_translation = torch.tensor([ 0,  0,  0.9], dtype=torch.float32)

    # 5. 创建SkeletonState
    skeleton_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton_tree,
        r=local_rotation,
        t=root_translation,
        is_local=True
    )

    # 6. 保存为npy文件
    skeleton_state.to_file('tpose/smpl_dart_tpose.npy')

if __name__ == '__main__':
    create_g1_tpose()




    #    [-0.00217368, -0.24078919,  0.02858379], #'Pelvis'
    #    [ 0.05858135, -0.08228007, -0.01766408], # 'L_Hip'
    #    [ 0.04345143, -0.3864696 ,  0.008037  ], # 'L_Knee'
    #    [-0.01479033, -0.42687446, -0.03742799], # 'L_Ankle'
    #    [ 0.04105436, -0.06028578,  0.12204242],'L_Foot'
    #    [-0.06030973, -0.09051332, -0.01354253],'R_Hip'
    #    [-0.04325663, -0.3836879 , -0.00484304],'R_Knee'
    #    [ 0.01905555, -0.42004555, -0.03456167],'R_Ankle'
    #    [-0.03483988, -0.06210563,  0.1303233 ],'R_Foot'
    #    [ 0.00443945,  0.12440356, -0.03838522],'Spine1'
    #    [ 0.00448844,  0.1379564 ,  0.02682032],'Spine2'
    #    [-0.00226459,  0.0560324 ,  0.00285505],'Spine3'
    #    [-0.01339018,  0.21163554, -0.03346758],'Neck'
    #    [ 0.01011321,  0.08893741,  0.05040986],'Head'
    #    [ 0.07170247,  0.11399969, -0.01889817],'L_Collar'
    #    [ 0.12292139,  0.04520511, -0.019046  ],'L_Shoulder'
    #    [ 0.25533187, -0.01564904, -0.02294649],'L_Elbow'
    #    [ 0.26570928,  0.01269813, -0.00737473],'L_Wrist'
    #    [-0.08295365,  0.11247235, -0.02370739],'R_Collar'
    #    [-0.11322831,  0.04685327, -0.00847207],'R_Shoulder'
    #    [-0.26012748, -0.0143693 , -0.03126873],'R_Elbow'
    #    [-0.2691084 ,  0.00679374, -0.00602677],'R_Wrist'


# [-0.00217368, -0.24078919,  0.02858379], #'Pelvis'
# [ 0.05858135, -0.08228007, -0.01766408], # 'L_Hip'
# [-0.06030973, -0.09051332, -0.01354253],'R_Hip'
# [ 0.00443945,  0.12440356, -0.03838522],'Spine1'
# [ 0.04345143, -0.3864696 ,  0.008037  ], # 'L_Knee'
# [-0.04325663, -0.3836879 , -0.00484304],'R_Knee'
# [ 0.00448844,  0.1379564 ,  0.02682032],'Spine2'
# [-0.01479033, -0.42687446, -0.03742799], # 'L_Ankle'
# [ 0.01905555, -0.42004555, -0.03456167],'R_Ankle'
# [-0.00226459,  0.0560324 ,  0.00285505],'Spine3'
# [ 0.04105436, -0.06028578,  0.12204242],'L_Foot'
# [-0.03483988, -0.06210563,  0.1303233 ],'R_Foot'
# [-0.01339018,  0.21163554, -0.03346758],'Neck'
# [ 0.07170247,  0.11399969, -0.01889817],'L_Collar'
# [-0.08295365,  0.11247235, -0.02370739],'R_Collar'
# [ 0.01011321,  0.08893741,  0.05040986],'Head'
# [ 0.12292139,  0.04520511, -0.019046  ],'L_Shoulder'
# [-0.11322831,  0.04685327, -0.00847207],'R_Shoulder'
# [ 0.25533187, -0.01564904, -0.02294649],'L_Elbow'
# [-0.26012748, -0.0143693 , -0.03126873],'R_Elbow'
# [ 0.26570928,  0.01269813, -0.00737473],'L_Wrist'
# [-0.2691084 ,  0.00679374, -0.00602677],'R_Wrist'














    # local_translation = torch.tensor([
    #    [-0.00217368, -0.24078919,  0.02858379], # Pelvis
    #    [ 0.05858135, -0.08228007, -0.01766408], # L_Hip
    #    [ 0.04345143, -0.3864696 ,  0.008037  ], #L_Knee
    #    [-0.01479033, -0.42687446, -0.03742799], #L_Ankle
    #    [ 0.04105436, -0.06028578,  0.12204242],#L_Foot
    #    [-0.06030973, -0.09051332, -0.01354253], #R_Hip
    #    [-0.04325663, -0.3836879 , -0.00484304],#R_Knee
    #    [ 0.01905555, -0.42004555, -0.03456167], #R_Ankle
    #    [-0.03483988, -0.06210563,  0.1303233 ],#R_Foot
    #    [ 0.00443945,  0.12440356, -0.03838522], #Spine1
    #    [ 0.00448844,  0.1379564 ,  0.02682032],#Spine2
    #    [-0.00226459,  0.0560324 ,  0.00285505], #Spine3
    #    [-0.01339018,  0.21163554, -0.03346758],#Neck
    #    [ 0.01011321,  0.08893741,  0.05040986], #Head
    #    [ 0.07170247,  0.11399969, -0.01889817],#L_Collar
    #    [ 0.12292139,  0.04520511, -0.019046  ], #L_Shoulder
    #    [ 0.25533187, -0.01564904, -0.02294649],#L_Elbow
    #    [ 0.26570928,  0.01269813, -0.00737473], #L_Wrist
    #    [-0.08295365,  0.11247235, -0.02370739], #R_Collar
    #    [-0.11322831,  0.04685327, -0.00847207],#R_Shoulder
    #    [-0.26012748, -0.0143693 , -0.03126873], #R_Elbow
    #    [-0.2691084 ,  0.00679374, -0.00602677] ], dtype=torch.float32) #R_Wrist



