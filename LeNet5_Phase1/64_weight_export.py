import torch
import numpy as np

PTH_FILE = "64_lenet5_traffic.pth"
OUT_FILE = "parameters_int8.h"
NUM_CLASSES = 43

def quantize(x, scale=127.0):
    x = x.detach().cpu().numpy()
    x = np.round(x * scale)
    x = np.clip(x, -128, 127)
    return x.astype(np.int8)

def c_shape(shape):
    return "".join([f"[{s}]" for s in shape])

def format_tensor(name, tensor):
    shape = tensor.shape
    flat = tensor.reshape(-1)

    s = f"// Layer: {name} | Shape: {shape}\n"
    s += f"const int8_t {name}{c_shape(shape)} = {{\n\t"

    for i, v in enumerate(flat):
        s += f"{int(v)}, "
        if (i + 1) % 10 == 0:
            s += "\n\t"

    s += "\n};\n\n"
    return s

def write_file(filename, blocks):
    with open(filename, "w") as f:
        f.write("#ifndef PARAMETERS_INT8_H\n")
        f.write("#define PARAMETERS_INT8_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Automatic Export from PyTorch\n")
        f.write("// Int8 quantized weights for HLS\n\n")

        for block in blocks:
            f.write(block)

        f.write("#endif\n")

def export(model):
    blocks = []

    blocks.append(format_tensor("conv1_weights", quantize(model.conv1.weight.data)))
    blocks.append(format_tensor("conv1_bias", quantize(model.conv1.bias.data)))

    blocks.append(format_tensor("conv2_weights", quantize(model.conv2.weight.data)))
    blocks.append(format_tensor("conv2_bias", quantize(model.conv2.bias.data)))

    blocks.append(format_tensor("fc1_weights", quantize(model.fc1.weight.data)))
    blocks.append(format_tensor("fc1_bias", quantize(model.fc1.bias.data)))

    blocks.append(format_tensor("fc2_weights", quantize(model.fc2.weight.data)))
    blocks.append(format_tensor("fc2_bias", quantize(model.fc2.bias.data)))

    blocks.append(format_tensor("fc3_weights", quantize(model.fc3.weight.data)))
    blocks.append(format_tensor("fc3_bias", quantize(model.fc3.bias.data)))

    write_file(OUT_FILE, blocks)
    print(f"Done -> {OUT_FILE} generated")

if __name__ == "__main__":
    from model_64 import LeNet5_64

    model = LeNet5_64(num_classes=NUM_CLASSES)

    state = torch.load(PTH_FILE, map_location="cpu")

    if "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    print("fc1 weight shape:", tuple(model.fc1.weight.shape))
    print("fc3 weight shape:", tuple(model.fc3.weight.shape))

    export(model)