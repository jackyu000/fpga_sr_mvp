import os
import torch
from model import TinySRNet


def quantize_symmetric_int8(t: torch.Tensor):
    max_abs = t.abs().max().item()
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def save_mem_int8(path, tensor):
    flat = tensor.reshape(-1).tolist()
    with open(path, "w") as f:
        for v in flat:
            if v < 0:
                v = (1 << 8) + v
            f.write(f"{v:02x}\n")


def main():
    os.makedirs("exported", exist_ok=True)

    model = TinySRNet(channels=8)
    state = torch.load("checkpoints/tinysr_final.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    scale_info = {}

    for name, tensor in model.state_dict().items():
        q, scale = quantize_symmetric_int8(tensor.float())
        scale_info[name] = scale

        safe_name = name.replace(".", "_")
        save_mem_int8(f"exported/{safe_name}.mem", q)

        print(
            f"{name:15s} shape={tuple(tensor.shape)!s:18s} "
            f"scale={scale:.8f}"
        )

    torch.save(scale_info, "exported/scales.pt")
    print("\nSaved quantized .mem files to exported/")


if __name__ == "__main__":
    main()