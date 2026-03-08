import torch
from model import TinySRNet


def main():
    device = torch.device("cpu")

    model = TinySRNet(channels=8).to(device)
    state = torch.load("checkpoints/tinysr_final.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("Parameter summary:\n")

    for name, tensor in model.state_dict().items():
        t = tensor.detach().cpu()
        print(
            f"{name:15s} "
            f"shape={tuple(t.shape)!s:18s} "
            f"min={t.min().item(): .6f} "
            f"max={t.max().item(): .6f} "
            f"mean={t.mean().item(): .6f}"
        )


if __name__ == "__main__":
    main()