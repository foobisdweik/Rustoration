#!/usr/bin/env python3
"""
Convert YOLOv8 .pt checkpoints to SafeTensors format.
Required because `yolo export format=safetensors` is unsupported.
"""

import torch
from pathlib import Path
from safetensors.torch import save_file
import argparse


def convert_yolo_to_safetensors(input_pt: str, output_safetensors: str):
    """
    Convert ultralytics .pt checkpoint to SafeTensors format.
    
    Args:
        input_pt: Path to YOLOv8 .pt file
        output_safetensors: Path to output .safetensors file
    """
    print(f"Loading YOLOv8 model from {input_pt}...")
    model = torch.load(input_pt, map_location='cpu')
    
    if isinstance(model, dict):
        state_dict = model
    else:
        state_dict = model.state_dict()
    
    print(f"Converting {len(state_dict)} tensors to SafeTensors format...")
    save_file(state_dict, output_safetensors)
    print(f"✓ Saved to {output_safetensors}")
    
    # Verify
    from safetensors.torch import load_file
    loaded = load_file(output_safetensors)
    print(f"✓ Verification: {len(loaded)} tensors loaded successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv8 .pt to SafeTensors")
    parser.add_argument("input", help="Input .pt file")
    parser.add_argument("--output", "-o", help="Output .safetensors file (default: same name)")
    
    args = parser.parse_args()
    
    output = args.output or str(Path(args.input).with_suffix('.safetensors'))
    convert_yolo_to_safetensors(args.input, output)
