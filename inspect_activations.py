import torch
from dataset import CIFARGrayscaleSRDataset
from model import TinySRNet


def print_stats(name, x):
    x = x.detach().cpu()
    print(
        f"{name:12s} "
        f"shape={tuple(x.shape)!s:18s} "
        f"min={x.min().item(): .6f} "
        f"max={x.max().item(): .6f} "
        f"mean={x.mean().item(): .6f}"
    )


def main():
    device = torch.device("cpu")

    ds = CIFARGrayscaleSRDataset(train=False)
    _, bicubic_64, _ = ds[0]

    x = bicubic_64.unsqueeze(0).to(device)

    model = TinySRNet(channels=8).to(device)
    state = torch.load("checkpoints/tinysr_final.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    print_stats("input", x)

    with torch.no_grad():
        x1 = model.net[0](x)
        print_stats("conv0", x1)

        x2 = model.net[1](x1)
        print_stats("relu0", x2)

        x3 = model.net[2](x2)
        print_stats("conv1", x3)

        x4 = model.net[3](x3)
        print_stats("relu1", x4)

        x5 = model.net[4](x4)
        print_stats("conv2", x5)

        x6 = model.net[5](x5)
        print_stats("relu2", x6)

        x7 = model.net[6](x6)
        print_stats("conv3", x7)

        out = torch.clamp(x + x7, 0.0, 1.0)
        print_stats("output", out)


if __name__ == "__main__":
    main()