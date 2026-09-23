import torch
from PINN.pinn import PINN, DEVICE
from TRAIN.train import train_pinn
from DIRS.dirs import MODEL_PATH

torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    # Recreate network architecture
    model = PINN(hidden_dim=64, n_layers=4).to(DEVICE)

    # Load Stage 1 weights
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )

    # Continue training
    model, history = train_pinn(model)
