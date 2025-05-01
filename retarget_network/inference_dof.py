import os
import argparse
import numpy as np
import torch
import torch.nn as nn

# If your training script is named train.py and is in the same folder,
# you can import RetargetNet directly. Otherwise, copy the class below.
try:
    from train import RetargetNet
except ImportError:
    # Fallback: redefine the network architecture
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


def inference(input_path: str, weights_path: str, output_path: str, batch_size: int = 256):
    """
    Run inference on a set of input features and save the predictions.

    Args:
        input_path: Path to a .npy file of shape (N, 135).
        weights_path: Path to the trained .pth model weights.
        output_path: Path where to save the predictions (.npy of shape (N, 29)).
        batch_size: Inference batch size (adjust to fit your GPU/CPU).
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = RetargetNet().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load inputs
    inputs = np.load(input_path)
    num_samples = inputs.shape[0]

    # Prepare output container
    all_preds = []

    # Inference loop
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = torch.tensor(inputs[start:end], dtype=torch.float32).to(device)
            preds = model(batch).cpu().numpy()
            all_preds.append(preds)

    # Concatenate and save
    outputs = np.vstack(all_preds)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, outputs)
    print(f"✅ Inference complete. Predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained RetargetNet model.")
    parser.add_argument("--input", type=str, default='data/raw_data/input_data.npy', help="Path to input .npy file (shape N x 135)")
    parser.add_argument("--weights", type=str, default='retarget_network/Model_weight/retarget_29dof_model.pth', help="Path to model weights .pth")
    parser.add_argument("--output", type=str, default='data/retarget_npy/NN_dof/dof_map', help="Where to save output .npy predictions")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    args = parser.parse_args()
    inference(args.input, args.weights, args.output, args.batch_size)



