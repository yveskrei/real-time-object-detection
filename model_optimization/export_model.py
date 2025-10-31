import argparse
import torch
import torch.onnx
from pathlib import Path
import os
import onnx

# Custom modules
from export_utils import ModelType, get_dinov3_model, get_yolov9_model

def export_model(
    model_name: str,
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_name: str = "images",
    output_name: str = "output"
) -> str:
    """
    Exports a given model to ONNX format with both FP32 and FP16 versions.
    """
    print(f"Exporting model: {model_name}")

    # Send model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    print(f"Using device: {device}")
    
    # Define output paths
    path_fp32 = os.path.join(output_path, f"{model_name}-fp32.onnx")
    path_fp16 = os.path.join(output_path, f"{model_name}-fp16.onnx")
    
    # Export to ONNX - FP32 (baseline)
    print("Exporting FP32 baseline model...")
    torch.onnx.export(
        model.float(),
        dummy_input.float(),
        path_fp32,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes={
            input_name: {0: 'batch_size'},
            output_name: {0: 'batch_size'}
        },
        export_params=True,
        keep_initializers_as_inputs=False,
        dynamo=False
    )
    print("FP32 export successful")
    
    # Convert to FP16
    print("Converting to FP16 precision...")
    torch.onnx.export(
        model.half(),
        dummy_input.half(),
        path_fp16,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes={
            input_name: {0: 'batch_size'},
            output_name: {0: 'batch_size'}
        },
        export_params=True,
        keep_initializers_as_inputs=False,
        dynamo=False
    )
    print("FP16 conversion successful")
    
    # Validate and save
    try:
        model_fp32 = onnx.load(path_fp32)
        onnx.checker.check_model(model_fp32)
        print("FP32 Model validation: PASSED")
    except Exception as e:
        print(f"Model validation warning: {e}")
    
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
        description='Export models to ONNX FP32 and FP16 formats'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=ModelType._member_names_,
        help='Type of model to export (e.g. YOLOV9, DINOv3)'
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

    # Load model to context
    model_type = ModelType[args.model_type]
    model = None
    dummy_input = None

    if model_type == ModelType.DINOV3:
        if not args.dino_type:
            raise ValueError("DINOv3 model type must be specified with --dino-type")
        
        model, dummy_input = get_dinov3_model(
            args.model_source_code,
            args.model_path,
            args.dino_type
        )
    elif model_type == ModelType.YOLOV9:
        model, dummy_input = get_yolov9_model(
            args.model_source_code,
            args.model_path
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if model is None or dummy_input is None:
        raise Exception('Failed to load model or dummy input')
    
    # Export model
    export_model(
        Path(args.model_path).stem,
        model,
        dummy_input,
        args.output_path,
        args.input_name,
        args.output_name
    )


if __name__ == '__main__':
    main()