#!/usr/bin/env python3
"""
Split pretrained ResNet-50 into 8 sequential chunks corresponding to:
conv4_block1_add, conv4_block4_add, conv4_block5_out, conv4_block6_add,
conv5_block1_add, conv5_block2_add, conv5_block3_add, and final output.

Each chunk is exported to its own ONNX file for TensorRT partitioned inference.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os

# ======================================================
# 1. Load pretrained ResNet-50
# ======================================================
model = models.resnet50(weights='IMAGENET1K_V1')
model.eval()

# For reference:
# layer1 -> conv2_x
# layer2 -> conv3_x
# layer3 -> conv4_x   (6 Bottleneck blocks)
# layer4 -> conv5_x   (3 Bottleneck blocks)

# ======================================================
# 2. Define custom split points based on the figure
# ======================================================
# We will include all layers up to each specified block.

# Helper to slice a Sequential up to a given index
def slice_blocks(layer, start, end):
    """Return a Sequential of submodules layer[start:end]."""
    return nn.Sequential(*list(layer.children())[start:end])

chunks = nn.ModuleList([
    nn.Sequential(  # Cut 1: up to conv4_block1_add
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2,
        slice_blocks(model.layer3, 0, 1)
    ),
    slice_blocks(model.layer3, 1, 4),  # Cut 2: conv4_block4_add
    slice_blocks(model.layer3, 4, 5),  # Cut 3: conv4_block5_out
    slice_blocks(model.layer3, 5, 6),  # Cut 4: conv4_block6_add
    slice_blocks(model.layer4, 0, 1),  # Cut 5: conv5_block1_add
    slice_blocks(model.layer4, 1, 2),  # Cut 6: conv5_block2_add
    slice_blocks(model.layer4, 2, 3),  # Cut 7: conv5_block3_add
    nn.Sequential(model.avgpool, nn.Flatten(), model.fc)  # Cut 8: output
])

# ======================================================
# 3. Export each chunk to ONNX
# ======================================================
os.makedirs("onnx_chunks", exist_ok=True)

x = torch.randn(1, 3, 224, 224)
for i, chunk in enumerate(chunks, 1):
    onnx_path = f"onnx_chunks/resnet50_chunk{i}.onnx"
    print(f"[INFO] Exporting {onnx_path} ...")
    torch.onnx.export(
        chunk,
        x,
        onnx_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )

    # Forward propagate to get the next input
    with torch.no_grad():
        x = chunk(x)

    print(f"       Output shape: {tuple(x.shape)}")

print("\n✅ All 8 chunks exported under ./onnx_chunks/")
