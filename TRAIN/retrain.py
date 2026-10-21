import torch
from PINN.pinn import PINN, DEVICE
from TRAIN.train import train_pinn
from DIRS.dirs import MODEL_PATH

torch.set_default_dtype(torch.float32)


def load_model():
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    linear_weights = sorted(
        (
            key
            for key in state_dict
            if key.startswith("net.") and key.endswith(".weight")
        ),
        key=lambda key: int(key.split(".")[1]),
    )
    if len(linear_weights) < 2 or linear_weights[0] != "net.0.weight":
        raise ValueError(f"Unsupported PINN checkpoint at {MODEL_PATH}")

    first_layer = state_dict[linear_weights[0]]
    if first_layer.shape[1] == 4:
        state_dict[linear_weights[0]] = torch.cat(
            [
                first_layer,
                torch.zeros(
                    first_layer.shape[0],
                    1,
                    device=first_layer.device,
                    dtype=first_layer.dtype,
                ),
            ],
            dim=1,
        )
    elif first_layer.shape[1] != 5:
        raise ValueError(
            "Checkpoint input layer must have four legacy features or five "
            f"current features; got {first_layer.shape[1]}."
        )

    model = PINN(
        hidden_dim=first_layer.shape[0],
        n_layers=len(linear_weights) - 1,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    model = load_model()
    model, history = train_pinn(model)
