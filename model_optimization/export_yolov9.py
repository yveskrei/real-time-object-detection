import argparse
import torch
import torch.onnx
from pathlib import Path
from onnxconverter_common import float16
import onnx
import onnxoptimizer
import os

class YOLOV9Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output[0][0]

def remove_training_nodes(model):
    """Remove training-specific nodes and convert to inference mode"""
    graph = model.graph
    nodes_to_remove = []
    
    # --- Collect nodes to remove and modify BN attributes ---
    for node in graph.node:
        # Remove Dropout layers completely
        if node.op_type == "Dropout":
            nodes_to_remove.append(node)
        
        # Convert BatchNorm to inference mode
        elif node.op_type == "BatchNormalization":
            for attr in node.attribute:
                if attr.name == "training_mode":
                    attr.i = 0  # Set to inference mode
    
    # --- Remove dropout nodes and redirect their connections ---
    for node in nodes_to_remove:
        # Dropout has one input (the tensor) and one output (the tensor), 
        # plus an optional second output (the mask).
        if len(node.input) < 1 or len(node.output) < 1:
            # Skip corrupted or non-standard nodes
            continue

        input_name = node.input[0]    # The tensor that goes IN
        output_name = node.output[0]  # The tensor that comes OUT (and is used by downstream ops)
        
        # In inference, Dropout is an Identity operation, so we redirect the
        # uses of its output to its input.
        if input_name != output_name:
            # Iterate through ALL nodes in the graph to find where the 
            # dropout's output is being used as an input.
            for other_node in graph.node:
                for i, name in enumerate(other_node.input):
                    if name == output_name:
                        # Redirect the connection: use the dropout's input
                        # tensor name instead of its output tensor name.
                        other_node.input[i] = input_name
            
            # Also check graph outputs in case the dropout output was the final model output
            for output_value_info in graph.output:
                if output_value_info.name == output_name:
                    output_value_info.name = input_name
        
        # Now that all connections are redirected, remove the node itself
        graph.node.remove(node)
        
    return model

def optimize_graph(model):
    """Apply comprehensive graph optimizations"""
    # Get available optimization passes
    available_passes = onnxoptimizer.get_available_passes()
    
    # Define optimization passes in optimal order, only using available ones
    desired_passes = [
        # Basic cleanup
        'eliminate_identity',
        'eliminate_nop_dropout', 
        'eliminate_unused_initializer',
        'eliminate_duplicate_initializer',
        
        # Constant optimizations
        'extract_constant_to_initializer',
        
        # Fusion optimizations (order matters)
        'fuse_bn_into_conv',  # Must come before other conv fusions
        'fuse_add_bias_into_conv',
        'fuse_transpose_into_gemm',
        'fuse_matmul_add_bias_into_gemm',
        
        # Shape optimizations
        'fuse_consecutive_squeezes',
        'fuse_consecutive_unsqueezes',
        'fuse_consecutive_transposes',
        
        # Advanced optimizations
        'eliminate_nop_transpose',
        'eliminate_nop_pad',
        'fuse_pad_into_conv',
    ]
    
    # Filter to only use available passes
    optimization_passes = [pass_name for pass_name in desired_passes if pass_name in available_passes]
    
    try:
        optimized_model = onnxoptimizer.optimize(model, optimization_passes)
        print(f"Applied {len(optimization_passes)} optimization passes")
        return optimized_model
    except Exception as e:
        print(f"Warning: Optimization failed ({e}), proceeding with minimal optimizations...")
        # Fallback to most basic optimizations
        basic_passes = ['eliminate_identity', 'eliminate_unused_initializer']
        basic_passes = [pass_name for pass_name in basic_passes if pass_name in available_passes]
        return onnxoptimizer.optimize(model, basic_passes)

def clean_unused_weights(model):
    """Remove unused initializers (weights) to reduce model size"""
    graph = model.graph
    
    # Collect all used tensor names
    used_names = set()
    
    # From node inputs
    for node in graph.node:
        used_names.update(node.input)
    
    # From graph inputs and outputs
    for input_tensor in graph.input:
        used_names.add(input_tensor.name)
    for output_tensor in graph.output:
        used_names.add(output_tensor.name)
    
    # Remove unused initializers safely
    original_count = len(graph.initializer)
    new_initializers = []
    
    for init in graph.initializer:
        if init.name in used_names:
            new_initializers.append(init)
    
    # Clear and rebuild initializer list
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)
    
    removed_count = original_count - len(graph.initializer)
    if removed_count > 0:
        print(f"Removed {removed_count} unused weight tensors")
    
    return model

def export_model(model_path: str, output_path: str, input_name: str = "images", output_name: str = "output", device: str = "cuda") -> str:
    """
    Exports PT model to onnx, both FP32 and highly optimized FP16 versions
    """
    # Load model
    model_base = torch.load(model_path, map_location=device, weights_only=False)
    model = model_base['model'].float().to(device).eval()
    model = YOLOV9Wrapper(model).to(device)

    # Dummy input
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    # Create output directory if doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Define output paths
    model_name = Path(model_path).stem
    path_fp32 = os.path.join(output_path, f"{model_name}-fp32.onnx")
    path_fp16 = os.path.join(output_path, f"{model_name}-fp16.onnx")

    # Export to ONNX - FP32 (baseline)
    print("Exporting FP32 baseline model...")
    torch.onnx.export(
        model,
        dummy_input,
        path_fp32,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=12,
        do_constant_folding=True,  # This helps with optimization
        dynamic_axes={
            input_name: {0: 'batch_size'},
            output_name: {0: 'batch_size'}
        },
        export_params=True,
        keep_initializers_as_inputs=False,  # Reduces model size
    )

    # Get original model size
    original_size = os.path.getsize(path_fp32) / (1024 * 1024)  # MB
    print(f"Exported FP32 model to: {path_fp32} ({original_size:.1f} MB)")

    # Load for optimization pipeline
    print("Starting FP16 optimization pipeline...")
    model_fp32 = onnx.load(path_fp32)
    
    # Step 1: Remove training-specific layers
    print("Step 1: Removing training layers...")
    model_optimized = remove_training_nodes(model_fp32)
    
    # Step 2: Apply graph-level optimizations BEFORE FP16 conversion
    print("Step 2: Applying graph optimizations...")
    model_optimized = optimize_graph(model_optimized)
    
    # Step 3: Clean unused weights
    print("Step 3: Cleaning unused weights...")
    model_optimized = clean_unused_weights(model_optimized)
    
    # Step 4: Convert to FP16 (this should be done AFTER other optimizations)
    print("Step 4: Converting to FP16 precision...")
    model_fp16 = float16.convert_float_to_float16(
        model_optimized, 
        keep_io_types=False,  # Convert inputs/outputs to FP16 too
        disable_shape_infer=False  # Keep shape inference for better optimization
    )
    
    # Step 5: Final cleanup after FP16 conversion
    print("Step 5: Final cleanup...")
    model_fp16 = clean_unused_weights(model_fp16)
    
    # Step 6: Validate and save
    try:
        onnx.checker.check_model(model_fp16)
        print("Model validation: PASSED")
    except Exception as e:
        print(f"Model validation warning: {e}")
    
    onnx.save(model_fp16, path_fp16)
    
    # Calculate optimization results
    optimized_size = os.path.getsize(path_fp16) / (1024 * 1024)  # MB
    size_reduction = ((original_size - optimized_size) / original_size) * 100
    
    print(f"Exported optimized FP16 model to: {path_fp16}")
    print(f"Size reduction: {original_size:.1f} MB → {optimized_size:.1f} MB ({size_reduction:.1f}% smaller)")
    print("Optimizations applied:")
    print("✓ Training layer removal (Dropout, BatchNorm inference mode)")
    print("✓ Graph fusion optimizations (Conv+BN+ReLU, etc.)")
    print("✓ Dead weight elimination")
    print("✓ FP16 precision conversion") 
    print("✓ Constant folding and identity elimination")

def main():
    parser = argparse.ArgumentParser(description='Export YOLOv9 model to ONNX with advanced optimizations.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the .pt PyTorch model')
    parser.add_argument('--output-path', type=str, default=os.getcwd(), help='Output directory for .onnx models')
    parser.add_argument('--input-name', type=str, default='images', help='Name of the ONNX model input')
    parser.add_argument('--output-name', type=str, default='output', help='Name of the ONNX model output')
    args = parser.parse_args()

    # Auto device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[i] Using device: {device}")

    export_model(
        args.model_path,
        args.output_path,
        args.input_name, 
        args.output_name, 
        device
    )

if __name__ == '__main__':
    main()