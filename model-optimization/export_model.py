import argparse
import torch
import torch.onnx
from pathlib import Path
import os
import onnx

# Custom modules
from export_utils import ModelType, get_dinov3_model, get_yolov9_model, get_yolo26_model

def export_model(
    model_name: str,
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_name: str = "images",
    output_name: str = "output",
    simplify: bool = False
) -> str:
    """
    Exports a given model to ONNX format with FP32 precision.
    """
    print(f"Exporting model: {model_name}")

    # Send model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    print(f"Using device: {device}")

    # Resolve the real output shape by running the model once. Only the batch
    # axis is dynamic (see dynamic_axes below), so every remaining dimension is
    # static by construction and safe to pin into the graph afterwards.
    with torch.no_grad():
        reference_output = model.float()(dummy_input.float())

    if not isinstance(reference_output, torch.Tensor):
        raise TypeError(
            f"Model wrapper must return a single tensor, got {type(reference_output).__name__}. "
            "Reduce the output to one tensor inside the wrapper's forward()."
        )

    output_shape = tuple(reference_output.shape[1:])
    print(f"Detected output shape: batch_size, {', '.join(map(str, output_shape))}")

    # Define output paths
    path_fp32 = os.path.join(output_path, f"{model_name}-fp32.onnx")

    # Export to ONNX - FP32
    print("Exporting FP32 model...")
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

    model_fp32 = onnx.load(path_fp32)

    # Simplify the graph (constant folding / dead node removal). Purely
    # structural - it must not change the numbers, but it CAN freeze dynamic
    # axes, so always re-verify batching afterwards.
    if simplify:
        try:
            import onnxslim
            before = len(model_fp32.graph.node)
            model_fp32 = onnxslim.slim(model_fp32)
            onnx.save(model_fp32, path_fp32)
            print(f"Simplified graph: {before} -> {len(model_fp32.graph.node)} nodes")
        except Exception as e:
            print(f"Simplify warning, keeping unsimplified graph: {e}")
            model_fp32 = onnx.load(path_fp32)

    # Pin the static output dimensions, so the graph advertises the real shape
    # (e.g. [batch_size, 84, 8400]) instead of symbolic placeholders. ONNX shape
    # inference cannot always derive these - a view()/reshape() on the symbolic
    # batch leaves the trailing dims as 'Concatoutput_dim_N' placeholders, which
    # Triton reads as [-1, -1, -1] and rejects against its config dims.
    dims = model_fp32.graph.output[0].type.tensor_type.shape.dim
    if len(dims) != len(output_shape) + 1:
        print(
            f"Output shape warning: graph declares {len(dims)} dimensions, "
            f"model returned {len(output_shape) + 1}. Leaving the graph untouched."
        )
    else:
        for index, size in enumerate(output_shape, start=1):
            dims[index].dim_value = size
        onnx.save(model_fp32, path_fp32)
        print(f"Pinned output shape to: batch_size, {', '.join(map(str, output_shape))}")

    # Validate and save
    try:
        onnx.checker.check_model(model_fp32)
        print("FP32 Model validation: PASSED")
    except Exception as e:
        print(f"Model validation warning: {e}")

    # Calculate size of the model
    original_size = os.path.getsize(path_fp32) / (1024 * 1024)  # MB
    print(f"Exported FP32 model to: {path_fp32}, size {original_size:.1f} MB")

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
        help='Path to the model source code directory (YOLOV9, DINOV3). '
             'Not needed for models loaded from an installed package (e.g. YOLO26)'
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
        '--input-shape',
        type=str,
        default='3,640,640',
        help='Shape of input, seperated by comma (e.g. 3,640,640)'
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
    parser.add_argument(
        '--max-det',
        type=int,
        default=300,
        help='Maximum detections per image, baked into the graph (YOLO26)'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify the ONNX graph with onnxslim after export'
    )

    args = parser.parse_args()

    # Load model to context
    model_type = ModelType[args.model_type]
    model = None

    if model_type == ModelType.DINOV3:
        if not args.dino_type:
            raise ValueError("DINOv3 model type must be specified with --dino-type")
        if not args.model_source_code:
            raise ValueError("DINOv3 source code must be specified with --model-source-code")

        model = get_dinov3_model(
            args.model_source_code,
            args.model_path,
            args.dino_type
        )
    elif model_type == ModelType.YOLOV9:
        if not args.model_source_code:
            raise ValueError("YOLOv9 source code must be specified with --model-source-code")

        model = get_yolov9_model(
            args.model_source_code,
            args.model_path
        )
    elif model_type == ModelType.YOLO26:
        model = get_yolo26_model(
            args.model_path,
            args.max_det
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create dummy input
    input_shape = tuple(map(int, args.input_shape.split(',')))
    if len(input_shape) != 3:
        raise Exception("Invalid input shape. Must be 3 dimensions (e.g. 3,640,640)")
    
    dummy_input = torch.randn(1, *input_shape)
    
    if model is None or dummy_input is None:
        raise Exception('Failed to load model or dummy input')
    
    # Export model
    export_model(
        Path(args.model_path).stem,
        model,
        dummy_input,
        args.output_path,
        args.input_name,
        args.output_name,
        args.simplify
    )

if __name__ == '__main__':
    main()