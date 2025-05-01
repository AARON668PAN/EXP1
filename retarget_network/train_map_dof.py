import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class RetargetDataset(Dataset):
    """Dataset for mapping 135‑dim input 👉 29‑DoF output."""

    def __init__(self, input_path: str, output_path: str):
        # 输入数据 (N, 135)
        self.input_data = np.load(input_path)

        # 目标输出数据 (N, 29)
        self.target_data = np.load(output_path)
        if self.target_data.ndim == 1:
            # (N,) → (N, 1) 容错处理
            self.target_data = self.target_data[:, None]
        assert (
            self.target_data.shape[1] == 29
        ), f"Expected output dim 29, got {self.target_data.shape[1]}"

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.input_data[idx], dtype=torch.float32)
        y = torch.tensor(self.target_data[idx], dtype=torch.float32)
        return x, y


class RetargetNet(nn.Module):
    """Simple MLP: 135‑D → 29‑D."""

    def __init__(self, input_dim: int = 135, hidden_dim: int = 512, output_dim: int = 29):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def train():
    # ---------- 配置 ----------
    input_path = "data/raw_data/all_train_data/raw_data_all.npy"
    output_path = "data/retarget_npy/dof_data/retarget_final_all.npy"  # (N,29)
    batch_size = 64
    lr = 1e-3
    epochs = 30
    save_dir = "retarget_network/Model_weight"
    save_name = "retarget_29dof_model.pth"

    # ---------- 数据 ----------
    dataset = RetargetDataset(input_path, output_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------- 模型 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetargetNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ---------- 训练 ----------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    # ---------- 保存 ----------
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model weights saved to: {save_path}")


if __name__ == "__main__":
    train()
