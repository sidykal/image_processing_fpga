import torch
import numpy as np
import re

PTH_FILE = "model.pth"
OUT_FILE = "parameters.h"

def clean_name(name):
    name = name.replace(".", "_")
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return name

def format_array(arr, indent="\t"):
    flat = arr.flatten()
    lines = []

    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        line = indent + ", ".join(f"{x:.8f}" for x in chunk) + ","
        lines.append(line)

    return "\n".join(lines)

def shape_to_c(shape):
    return "".join(f"[{dim}]" for dim in shape)

def main():
    checkpoint = torch.load(PTH_FILE, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    with open(OUT_FILE, "w") as f:
        f.write("#ifndef PARAMETERS_H\n")
        f.write("#define PARAMETERS_H\n\n")
        f.write("// Automatic Export from PyTorch\n")
        f.write("// Float32 weights for HLS\n\n")

        for name, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                continue

            clean = clean_name(name)
            arr = tensor.detach().cpu().numpy().astype(np.float32)

            f.write(f"// Layer: {clean} | Shape: {arr.shape}\n")
            f.write(f"const float {clean}{shape_to_c(arr.shape)} = {{\n")
            f.write(format_array(arr))
            f.write("\n};\n\n")

        f.write("#endif\n")

    print(f"Saved parameters to {OUT_FILE}")

if __name__ == "__main__":
    main()