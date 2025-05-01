import numpy as np
from pathlib import Path

folder = Path("data/retarget_npy/dof_data")        # 存放 *.npy 的目录
pattern = "retarget_final*.npy"                    # 文件名模式

# 把文件按编号 0,1,2,… 排好序
files = sorted(folder.glob(pattern),
               key=lambda p: int(p.stem.replace("retarget_final", "")))

arrays = [np.load(f, allow_pickle=True) for f in files]   # 读入为 list[np.ndarray]
big_data = np.concatenate(arrays, axis=0)                 # 在第 0 维拼接

out_path = folder / "retarget_final_all.npy"
np.save(out_path, big_data)

print(f"concatenated shape {big_data.shape}  →  {out_path}")
