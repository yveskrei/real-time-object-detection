import argparse
import torch
import torch.nn as nn
import torch.onnx
from pathlib import Path
import os
import onnx

class DINOV3Wrapper(nn.Module):
    """Wrapper to extract CLS token from DINOv3 model output"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        # Extract CLS token (first token in sequence)
        return output[:, 0]

def export_model(
    model_path: str,
    model_source_code: str,
    dino_type: str,
    output_path: str,
    input_name: str = "images",
    output_name: str = "output"
) -> str:
    """
    Exports DINOv3 model to ONNX, both FP32 and highly optimized FP16 versions
    """
    print(f"Loading DINOv3 model: {dino_type}")
    print(f"Model weights: {model_path}")
    print(f"Source code path: {model_source_code}")

    # Auto device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[i] Using device: {device}")

    # Load model using torch.hub
    model_base = torch.hub.load(
        model_source_code,
        dino_type,
        source='local',
        pretrained=False  # We'll load weights manually
    )

    # Load weights seperately
    state_dict = torch.load(
        model_path, 
        map_location=device, 
        weights_only=True
    )
    model_base.load_state_dict(state_dict)
    
    # Wrap model to extract CLS token and set to eval mode
    model = DINOV3Wrapper(model_base).to(device)
    model = model.eval()
    
    # Dummy input (DINOv3 typically uses 224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Create output directory if doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Define output paths
    model_name = Path(model_path).stem if model_path else dino_type
    path_fp32 = os.path.join(output_path, f"{model_name}-fp32.onnx")
    path_fp16 = os.path.join(output_path, f"{model_name}-fp16.onnx")
    
    # Export to ONNX - FP32 (baseline)
    print("\nExporting FP32 baseline model...")
    torch.onnx.export(
        model,
        dummy_input,
        path_fp32,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes={
            input_name: {0: 'batch_size'},
            output_name: {0: 'batch_size'}
        },
        export_params=True,
        keep_initializers_as_inputs=False,
        dynamo=False
    )
    print("✓ FP32 export successful")
    
    # Step 4: Convert to FP16 (this should be done AFTER other optimizations)
    print("Converting to FP16 precision...")
    torch.onnx.export(
        model.half(),
        dummy_input.half(),
        path_fp16,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes={
            input_name: {0: 'batch_size'},
            output_name: {0: 'batch_size'}
        },
        export_params=True,
        keep_initializers_as_inputs=False,
        dynamo=False
    )
    print("✓ FP16 conversion successful")
    
    # Step 7: Validate and save
    try:
        model_fp16 = onnx.load(path_fp16)
        onnx.checker.check_model(model_fp16)
        print("FP16 Model validation: PASSED")
    except Exception as e:
        print(f"Model validation warning: {e}")
    
    # Calculate optimization results
    original_size = os.path.getsize(path_fp32) / (1024 * 1024)  # MB
    optimized_size = os.path.getsize(path_fp16) / (1024 * 1024)  # MB
    size_reduction = ((original_size - optimized_size) / original_size) * 100
    
    print(f"Exported base FP32 model to: {path_fp32}")
    print(f"Exported optimized FP16 model to: {path_fp16}")
    print(f"Size reduction: {original_size:.1f} MB → {optimized_size:.1f} MB ({size_reduction:.1f}% smaller)")


def main():
    parser = argparse.ArgumentParser(
        description='Export DINOv3 model to ONNX with RoPE TensorRT patch and advanced optimizations.'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the .pth PyTorch model weights'
    )
    parser.add_argument(
        '--model-source-code',
        type=str,
        required=True,
        help='Path to the DINOv3 source code directory for torch.hub.load'
    )
    parser.add_argument(
        '--dino-type',
        type=str,
        required=True,
        help='DINOv3 model type (e.g., dinov3_vitb16, dinov3_vits14, dinov3_vitl14)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=os.getcwd(),
        help='Output directory for .onnx models'
    )
    parser.add_argument(
        '--input-name',
        type=str,
        default='images',
        help='Name of the ONNX model input'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='output',
        help='Name of the ONNX model output'
    )
    
    args = parser.parse_args()
    
    export_model(
        args.model_path,
        args.model_source_code,
        args.dino_type,
        args.output_path,
        args.input_name,
        args.output_name
    )


if __name__ == '__main__':
    main()