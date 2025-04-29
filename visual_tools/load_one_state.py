
from skeleton.skeleton3d import SkeletonState, SkeletonTree
import torch
import numpy as np



def create_cmu_tpose(tpose_path):

    # 读取T-pose文件
    tpose_data = np.load(tpose_path, allow_pickle=True)
    tpose_data = tpose_data.item()

    # 构建一个skeleton state需要用到的所有要素
    node_names = tpose_data['skeleton_tree']['node_names']
    local_translation = tpose_data['skeleton_tree']['local_translation']['arr']
    local_translation = torch.tensor(local_translation, dtype=torch.float32)

    parent_indices = tpose_data['skeleton_tree']['parent_indices']['arr']
    parent_indices = torch.tensor(parent_indices, dtype=torch.int64)

    source_rotation = tpose_data['rotation']['arr']
    source_rotation = torch.tensor(source_rotation, dtype=torch.float32)

    source_root_translation = tpose_data['root_translation']['arr']
    source_root_translation = torch.tensor(source_root_translation, dtype=torch.float32)

    # 构建SkeletonTree
    skeleton_tree = SkeletonTree(
        parent_indices=parent_indices,
        local_translation=local_translation,
        node_names=node_names
    )


    # 创建CMU SkeletonState
    skeleton_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton_tree,
        r=source_rotation,
        t=source_root_translation,
        is_local=True
    )
    # CMU的是global rotation
    
    return skeleton_state

if __name__ == '__main__':
    create_cmu_tpose('ASE/ase/poselib/data/tpose/MDM_Tpose.npy')