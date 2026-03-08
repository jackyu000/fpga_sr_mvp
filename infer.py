import torch
import matplotlib.pyplot as plt
from dataset import CIFARGrayscaleSRDataset
from model import TinySRNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CIFARGrayscaleSRDataset(train=False)
    _, bicubic_64, hr_64 = ds[0]

    model = TinySRNet(channels=8).to(device)
    model.load_state_dict(torch.load("checkpoints/tinysr_final.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        inp = bicubic_64.unsqueeze(0).to(device)
        pred = model(inp).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(bicubic_64.squeeze(0), cmap="gray")
    axes[0].set_title("Bicubic")
    axes[0].axis("off")

    axes[1].imshow(pred.squeeze(0), cmap="gray")
    axes[1].set_title("Model Output")
    axes[1].axis("off")

    axes[2].imshow(hr_64.squeeze(0), cmap="gray")
    axes[2].set_title("Target HR")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()