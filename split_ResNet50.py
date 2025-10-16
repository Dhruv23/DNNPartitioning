#!/usr/bin/env python3
"""
Split pretrained ResNet-50 into six sequential chunks:
[conv1+bn1+relu+maxpool], [layer1], [layer2], [layer3], [layer4], [avgpool+fc]
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

# ======================================================
# 2. Define the chunks
# ======================================================
chunks = nn.ModuleList([
    nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),  # Chunk 1
    model.layer1,                                                     # Chunk 2
    model.layer2,                                                     # Chunk 3
    model.layer3,                                                     # Chunk 4
    model.layer4,                                                     # Chunk 5
    nn.Sequential(model.avgpool, nn.Flatten(), model.fc)              # Chunk 6
])

# ======================================================
# 3. Export each chunk to ONNX
# ======================================================
os.makedirs("onnx_chunks", exist_ok=True)

# Input tensor for ResNet50
x = torch.randn(1, 3, 224, 224)
for i, chunk in enumerate(chunks, 1):
    # Export current chunk
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

    # Forward output becomes input for next chunk
    with torch.no_grad():
        x = chunk(x)

    print(f"       Output shape: {tuple(x.shape)}")

print("\n✅ All 6 chunks exported under ./onnx_chunks/")
