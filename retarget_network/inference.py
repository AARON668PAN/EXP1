# inference_optimized.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RetargetNet(nn.Module):
    def __init__(self, input_dim=135, hidden_dim=512, num_joints=14):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.trans_head = nn.Linear(hidden_dim, 3)
        self.rot_head   = nn.Linear(hidden_dim, num_joints * 4)

    def forward(self, x):
        x = self.shared(x)
        pred_trans = self.trans_head(x)                   # (N, 3)
        pred_rot   = self.rot_head(x).reshape(-1, 14, 4)  # (N, 14, 4)
        pred_rot   = F.normalize(pred_rot, dim=-1)        # unit quaternions
        return pred_trans, pred_rot

def run_inference(
    model_checkpoint: str,
    input_npy: str,
    output_pt: str,
    device: str = None
):
    # 1. 设备设置
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 2. 加载输入 (N,135)
    X = np.load(input_npy)
    X_tensor = torch.from_numpy(X).float().to(device)
    N = X_tensor.shape[0]

    # 3. 构建模型并载入权重
    model = RetargetNet().to(device)
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4. 推理并计时
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        pred_trans, pred_rot = model(X_tensor)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t1 = time.time()

    print(f"Inference on {N} frames took {t1 - t0:.4f} seconds")

    # 5. 保存为 PyTorch tensor
    out_dict = {
        'root_pos':       pred_trans.cpu(),
        'local_rotation': pred_rot.cpu()
    }
    os.makedirs(os.path.dirname(output_pt), exist_ok=True)
    torch.save(out_dict, output_pt)
    print(f"Saved predictions (tensors) to {output_pt}")
    print(f"  root_pos shape:       {out_dict['root_pos'].shape}")
    print(f"  local_rotation shape: {out_dict['local_rotation'].shape}")

if __name__ == "__main__":
    MODEL_PATH  = 'retarget_network/Model_weight/retarget_model.pth'
    INPUT_PATH  = 'data/raw_data/input_data.npy'              # your (133,135) data
    OUTPUT_PATH = 'data/retarget_npy/predicted_retarget.pt'   # saves tensors

    run_inference(MODEL_PATH, INPUT_PATH, OUTPUT_PATH)
