#!/usr/bin/env python3
"""
offline.py
-----------
Model-aware offline partitioning + TensorRT engine building.

Currently supports:
  - resnet50          (torchvision.models.resnet50)
  - efficientnet_b0   (torchvision.models.efficientnet_b0)

Splitting strategy (Option C):
  * ResNet-50:
      - Uses the SAME 8-chunk logic as the provided script:
        1) conv1..maxpool + layer1 + layer2 + conv4_block1
        2) conv4_block2-4
        3) conv4_block5
        4) conv4_block6
        5) conv5_block1
        6) conv5_block2
        7) conv5_block3
        8) avgpool + flatten + fc

  * EfficientNet-B0:
      - Treats each .features[i] module as a "block"
      - Then appends avgpool, flatten, and classifier layers
      - Splits this list into ~equal sequential chunks.

For each model:
  - Build chunks as nn.Sequential modules
  - Export each chunk to ONNX (valid, standalone graphs)
  - Optionally build each ONNX into a TensorRT .engine using trtexec

Usage example:
    python3 offline.py --models resnet50 efficientnet_b0 --splits 8 --outdir engines2

This will create:
  engines2/
    resnet50/
      chunk1.onnx ... chunk8.onnx
      chunk1.engine ... chunk8.engine
    efficientnet_b0/
      chunk1.onnx ... chunkN.onnx
      chunk1.engine ... chunkN.engine
"""

import os
import math
import argparse
import subprocess

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_trtexec(onnx_path: str, engine_path: str, fp16: bool = True):
    """Call TensorRT's trtexec to build an engine from an ONNX file."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]
    if fp16:
        cmd.append("--fp16")

    print(f"[TRT] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[TRT] Saved engine: {engine_path}")


# ============================================================
# ResNet-50 chunking (matches your original script)
# ============================================================

def build_resnet50_chunks(model: nn.Module):
    """
    Build 8 sequential chunks from a torchvision ResNet-50,
    using exactly the same logic as the user's hand-written script.
    """
    model.eval()

    def slice_blocks(layer, start, end):
        """Return a Sequential of submodules layer[start:end]."""
        return nn.Sequential(*list(layer.children())[start:end])

    # Chunk definitions (same as your original script)
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

    print("[INFO] ResNet-50: built 8 chunks (fixed scheme).")
    return list(chunks)


# ============================================================
# EfficientNet-B0 chunking (stage/block-based)
# ============================================================

def build_efficientnet_b0_chunks(model: nn.Module, num_chunks: int):
    """
    Build ~num_chunks sequential chunks from torchvision EfficientNet-B0,
    splitting at feature "blocks"/stages.

    Strategy:
      - Flatten model.features into a list of modules
      - Append avgpool, flatten, classifier layers
      - Partition this list into num_chunks contiguous groups
    """
    model.eval()

    modules = []

    # 1) Feature blocks (each is a "stage"/block)
    for m in model.features.children():
        modules.append(m)

    # 2) AvgPool + Flatten
    # EfficientNet-B0 in torchvision has avgpool attribute
    if hasattr(model, "avgpool"):
        modules.append(model.avgpool)
    else:
        # fallback to a generic adaptive pool if needed
        modules.append(nn.AdaptiveAvgPool2d(1))

    modules.append(nn.Flatten())

    # 3) Classifier (Sequential: Dropout, Linear, etc.)
    for m in model.classifier.children():
        modules.append(m)

    total_blocks = len(modules)
    blocks_per_chunk = math.ceil(total_blocks / num_chunks)

    print(f"[INFO] EfficientNet-B0: total blocks = {total_blocks}, "
          f"target chunks = {num_chunks}, ~{blocks_per_chunk} blocks/chunk")

    chunks = []
    for i in range(num_chunks):
        start = i * blocks_per_chunk
        end = min((i + 1) * blocks_per_chunk, total_blocks)
        if start >= end:
            break
        chunk = nn.Sequential(*modules[start:end])
        chunks.append(chunk)
        print(f"[INFO]   Chunk {i+1}: blocks {start} → {end}")

    print(f"[INFO] EfficientNet-B0: built {len(chunks)} chunks.")
    return chunks


# ============================================================
# Export chunks to ONNX + optionally build engines
# ============================================================

def export_chunks_to_onnx_and_trt(
    model_name: str,
    chunks: list,
    outdir: str,
    build_engines: bool = True,
    fp16: bool = True,
    input_shape=(1, 3, 224, 224),
    opset: int = 17,
):
    """
    For a given list of chunks (nn.Sequential modules), export each to ONNX
    and optionally build a TensorRT engine per chunk.
    """
    ensure_dir(outdir)

    x = torch.randn(*input_shape)
    x = x.to(torch.float32)

    onnx_paths = []

    for idx, chunk in enumerate(chunks, start=1):
        onnx_path = os.path.join(outdir, f"{model_name}_chunk{idx}.onnx")
        engine_path = os.path.join(outdir, f"{model_name}_chunk{idx}.engine")

        print(f"[INFO] Exporting {onnx_path} ...")

        torch.onnx.export(
            chunk,
            x,
            onnx_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )

        # Forward propagate to get next input
        with torch.no_grad():
            x = chunk(x)

        print(f"[INFO]   Chunk {idx} output shape: {tuple(x.shape)}")
        onnx_paths.append(onnx_path)

        # Build TensorRT engine if requested
        if build_engines:
            run_trtexec(onnx_path, engine_path, fp16=fp16)

    print(f"\n[OK] {model_name}: exported {len(chunks)} chunks to {outdir}\n")
    return onnx_paths


# ============================================================
# Model factory
# ============================================================

def load_model_by_name(name: str) -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        print("[INFO] Loading torchvision.models.resnet50(pretrained IMAGENET1K_V1)")
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif name == "efficientnet_b0":
        print("[INFO] Loading torchvision.models.efficientnet_b0(pretrained IMAGENET1K_V1)")
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model name: {name}")


def build_chunks_for_model(name: str, splits: int):
    model = load_model_by_name(name)

    if name.lower() == "resnet50":
        # Ignore 'splits' and use the fixed 8-chunk scheme
        if splits != 8:
            print(f"[WARN] ResNet-50 uses a fixed 8-chunk scheme; ignoring --splits={splits}.")
        return build_resnet50_chunks(model)

    elif name.lower() == "efficientnet_b0":
        return build_efficientnet_b0_chunks(model, splits)

    else:
        raise ValueError(f"No chunking strategy implemented for model: {name}")


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Offline DNN partitioning + TensorRT engine generation")

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names to process (e.g., resnet50 efficientnet_b0)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=8,
        help="Number of chunks for models that use dynamic splitting (e.g., EfficientNet-B0). "
             "ResNet-50 always uses 8 fixed chunks.",
    )
    parser.add_argument(
        "--outdir",
        default="engines2",
        help="Base output directory for ONNX chunks and TensorRT engines",
    )
    parser.add_argument(
    "--no-engines",
    action="store_true",
    help="If set, only export ONNX chunks and skip TensorRT engine building.",
    )

    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 in TensorRT build (default: FP16 enabled).",
    )

    args = parser.parse_args()

    build_engines = not args.no_engines
    fp16 = not args.no_fp16

    print("\n=== Offline Model-Aware Partitioning ===\n")
    print(f"[INFO] Models: {args.models}")
    print(f"[INFO] Splits (for dynamic models like EfficientNet-B0): {args.splits}")
    print(f"[INFO] Base output directory: {args.outdir}")
    print(f"[INFO] Build TensorRT engines: {build_engines}")
    print(f"[INFO] TensorRT FP16: {fp16}\n")

    for model_name in args.models:
        model_name_clean = model_name.lower()
        model_outdir = os.path.join(args.outdir, model_name_clean)
        ensure_dir(model_outdir)

        print(f"\n[====] Processing model: {model_name_clean} [====]\n")

        # Build chunks
        chunks = build_chunks_for_model(model_name_clean, args.splits)

        # Export ONNX + TRT
        export_chunks_to_onnx_and_trt(
            model_name=model_name_clean,
            chunks=chunks,
            outdir=model_outdir,
            build_engines=build_engines,
            fp16=fp16,
            input_shape=(1, 3, 224, 224),  # both resnet50 & eff_b0 use 224x224
            opset=17,
        )

    print("\n[✓] All requested models processed.\n")


if __name__ == "__main__":
    main()
