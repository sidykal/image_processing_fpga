import re
import numpy as np

INPUT_FILE = "parameters.h"
OUTPUT_FILE = "parameters_int8.h"

def quantize_array(values):
    values = np.array(values, dtype=np.float32)
    max_val = np.max(np.abs(values))
    
    if max_val == 0:
        scale = 1.0
    else:
        scale = 127.0 / max_val

    q = np.round(values * scale).astype(np.int8)
    return q, scale

def parse_floats(block):
    # extract all float numbers
    nums = re.findall(r'-?\d+\.\d+(?:e[-+]?\d+)?', block)
    return [float(n) for n in nums]

def replace_block(block, quantized_vals):
    # replace floats with int8 values
    idx = 0

    def repl(match):
        nonlocal idx
        val = quantized_vals[idx]
        idx += 1
        return str(int(val))

    return re.sub(r'-?\d+\.\d+(?:e[-+]?\d+)?', repl, block)

def main():
    with open(INPUT_FILE, "r") as f:
        content = f.read()

    output = content

    pattern = r'const float ([^{]+)\{([^;]+)\};'

    matches = re.finditer(pattern, content, re.DOTALL)

    scales = {}

    for match in matches:
        name = match.group(1).strip()
        block = match.group(2)

        values = parse_floats(block)
        q_vals, scale = quantize_array(values)

        print(f"{name} -> scale: {scale:.6f}")
        scales[name] = scale

        new_block = replace_block(block, q_vals)

        # Replace float with int8_t
        new_decl = f"const int8_t {name}{{{new_block}}};"
        old_decl = match.group(0)

        output = output.replace(old_decl, new_decl)

    # Add include for int8_t
    output = "#include <stdint.h>\n\n" + output

    with open(OUTPUT_FILE, "w") as f:
        f.write(output)

    print("\nSaved quantized file to:", OUTPUT_FILE)

    print("\nScales (use in HLS):")
    for k, v in scales.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()