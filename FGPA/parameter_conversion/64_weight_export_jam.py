import torch
import numpy as np
import os

SHIFTS = {
    "conv1": 8,
    "conv2": 10,
    "fc1": 10,
    "fc2": 9,
    "fc3": 9,
}


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PTH_FILE = os.path.join(BASE_DIR, "CPU", "weights", "64_lenet5_traffic.pth")
OUT_FILE = os.path.join(BASE_DIR, "FGPA", "64_quantized", "parameters_int8.h")

NUM_CLASSES = 43

INPUT_SCALE = 127.0

def c_shape(shape):
    return "".join([f"[{s}]" for s in shape])

def get_weight_scale(x):
    x = x.detach().cpu().numpy()
    max_abs = np.max(np.abs(x))

    if max_abs == 0:
        return 1.0

    return 127.0 / max_abs

def quantize_weight(x, scale):
    x = x.detach().cpu().numpy()
    q = np.round(x * scale)
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)

def quantize_bias(x, input_scale, weight_scale):
    x = x.detach().cpu().numpy()

    # bias must match accumulator scale
    q = np.round(x * input_scale * weight_scale)

    return q.astype(np.int32)

def format_tensor_int8(name, tensor):
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

def format_tensor_int32(name, tensor):
    shape = tensor.shape
    flat = tensor.reshape(-1)

    s = f"// Layer: {name} | Shape: {shape}\n"
    s += f"const int32_t {name}{c_shape(shape)} = {{\n\t"

    for i, v in enumerate(flat):
        s += f"{int(v)}, "
        if (i + 1) % 10 == 0:
            s += "\n\t"

    s += "\n};\n\n"
    return s

def write_file(filename, blocks, scale_defs):
    with open(filename, "w") as f:
        f.write("#ifndef PARAMETERS_INT8_H\n")
        f.write("#define PARAMETERS_INT8_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Automatic Export from PyTorch\n")
        f.write("// Int8 weights + int32 biases for HLS\n\n")

        for line in scale_defs:
            f.write(line + "\n")

        f.write("\n")

        for block in blocks:
            f.write(block)

        f.write("#endif\n")

def export(model):
    blocks = []
    scale_defs = []

    layers = [
        ("conv1", model.conv1),
        ("conv2", model.conv2),
        ("fc1", model.fc1),
        ("fc2", model.fc2),
        ("fc3", model.fc3),
    ]

    prev_activation_scale = INPUT_SCALE

    for name, layer in layers:
        weight_scale = get_weight_scale(layer.weight.data)

        q_w = quantize_weight(layer.weight.data, weight_scale)
        q_b = quantize_bias(layer.bias.data, prev_activation_scale, weight_scale)

        blocks.append(format_tensor_int8(f"{name}_weights", q_w))
        blocks.append(format_tensor_int32(f"{name}_bias", q_b))

        print(f"{name}:")
        print(f"  input activation scale: {prev_activation_scale:.6f}")
        print(f"  weight scale: {weight_scale:.6f}")
        print(f"  bias scale: {prev_activation_scale * weight_scale:.6f}")

        # this is the important part
        prev_activation_scale = (prev_activation_scale * weight_scale) / (2 ** SHIFTS[name])

        print(f"  output activation scale after shift: {prev_activation_scale:.6f}")
    write_file(OUT_FILE, blocks, scale_defs)
    print(f"\nDone -> {OUT_FILE} generated")

if __name__ == "__main__":
    from  CPU.models.model_64 import LeNet5_64 

    model = LeNet5_64(num_classes=NUM_CLASSES)

    state = torch.load(PTH_FILE, map_location="cpu")

    if "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    print("fc1 weight shape:", tuple(model.fc1.weight.shape))
    print("fc3 weight shape:", tuple(model.fc3.weight.shape))

    export(model)