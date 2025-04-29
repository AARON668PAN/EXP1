import numpy as np
import torch

base_path = 'data/raw_data/results.npy'
data_dict = np.load(base_path, allow_pickle=True).item()

# 将 GPU 上的 Tensor 移到 CPU，然后转成 numpy
motion_data = data_dict['motion_cont6d'][:, 0:135].cpu().numpy()

# 保存为 .npy 文件
np.save('input_data.npy', motion_data)
