import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class RetargetDataset(Dataset):
    def __init__(self, input_path, output_path):
        # 输入数据：shape (20253, 135)
        self.input_data = np.load(input_path)
        # 输出数据：dict with 'root_translation' (20253,3) and 'rotation' (20253,14,4)
        output_data = np.load(output_path, allow_pickle=True).item()
        self.root_translation = output_data['root_translation']
        self.rotation = output_data['rotation']

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.input_data[idx], dtype=torch.float32)
        trans = torch.tensor(self.root_translation[idx], dtype=torch.float32)
        rot = torch.tensor(self.rotation[idx], dtype=torch.float32)
        return x, trans, rot

class RetargetNet(nn.Module):
    def __init__(self, input_dim=135, hidden_dim=512, num_joints=14):
        super().__init__()
        # shared MLP backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # head for root_translation
        self.trans_head = nn.Linear(hidden_dim, 3)
        # head for rotation (14 joints × 4 dims)
        self.rot_head = nn.Linear(hidden_dim, num_joints * 4)

    def forward(self, x):
        x = self.shared(x)
        pred_trans = self.trans_head(x)                   # (B, 3)
        pred_rot = self.rot_head(x).reshape(-1, 14, 4)    # (B, 14, 4)
        pred_rot = F.normalize(pred_rot, dim=-1)          # ensure unit quaternions
        return pred_trans, pred_rot

def train():
    # --------- 配置 ---------
    input_path  = 'data/raw_data/all_train_data/raw_data_all.npy'
    output_path = 'data/retarget_npy/NN_retarget/retarget_data_all.npy'
    batch_size  = 64
    lr          = 1e-3
    epochs      = 30
    save_dir    = 'retarget_network/Model_weight'
    save_name   = 'retarget_model.pth'

    # --------- 准备数据 ---------
    dataset   = RetargetDataset(input_path, output_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --------- 构建模型 ---------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetargetNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --------- 训练循环 ---------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, gt_trans, gt_rot in dataloader:
            x        = x.to(device)
            gt_trans = gt_trans.to(device)
            gt_rot   = gt_rot.to(device)

            pred_trans, pred_rot = model(x)

            loss_trans = criterion(pred_trans, gt_trans)
            loss_rot   = criterion(pred_rot, gt_rot)
            loss = loss_trans + loss_rot

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    # --------- 保存模型权重 ---------
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to: {save_path}")

if __name__ == "__main__":
    train()
