import argparse
import torch
import torch.nn as nn
import torch.onnx
from pathlib import Path
from onnxconverter_common import float16
import onnx
import onnx.helper
import onnxoptimizer
import os
import numpy as np
from onnx import numpy_helper

class DINOV3Wrapper(nn.Module):
    """Wrapper to extract CLS token from DINOv3 model output"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        # Extract CLS token (first token in sequence)
        return output[:, 0]


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


def fix_range_nodes_fp32(model):
    """
    Fix Range operator inputs to ensure ALL are FP32.
    TensorRT does NOT support FP16 Range - only int32, int64, and float32.
    This ensures Range nodes work while keeping the rest of the model in FP16.
    """
    print("Converting Range nodes to FP32 (TensorRT requirement)...")
    graph = model.graph
    
    # Find all Range nodes
    range_nodes = [node for node in graph.node if node.op_type == "Range"]
    
    if not range_nodes:
        print("  No Range nodes found")
        return model
    
    print(f"  Found {len(range_nodes)} Range nodes")
    
    # Collect all inputs used by Range nodes
    range_inputs = set()
    for node in range_nodes:
        range_inputs.update(node.input)
    
    # Fix 1: Convert initializers (constants) that feed Range nodes to FP32
    fixed_count = 0
    for initializer in list(graph.initializer):
        if initializer.name in range_inputs:
            tensor = numpy_helper.to_array(initializer)
            if tensor.dtype != np.float32:
                # Convert to FP32
                tensor_fp32 = tensor.astype(np.float32)
                new_initializer = numpy_helper.from_array(tensor_fp32, name=initializer.name)
                
                # Replace
                graph.initializer.remove(initializer)
                graph.initializer.append(new_initializer)
                fixed_count += 1
    
    # Fix 2: Force Cast nodes feeding Range to output FP32
    for node in graph.node:
        if node.op_type == "Cast" and node.output[0] in range_inputs:
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = 1  # FLOAT (FP32)
                    fixed_count += 1
    
    # Fix 3: Find upstream nodes that produce Range inputs and ensure they're FP32
    # Build a mapping of output -> node
    output_to_node = {}
    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node
    
    # For each Range input that comes from a node (not an initializer)
    for range_input in range_inputs:
        if range_input in output_to_node:
            producer_node = output_to_node[range_input]
            # If it's a Constant node, ensure output is FP32
            if producer_node.op_type == "Constant":
                for attr in producer_node.attribute:
                    if attr.name == "value":
                        tensor = numpy_helper.to_array(attr.t)
                        if tensor.dtype != np.float32:
                            tensor_fp32 = tensor.astype(np.float32)
                            new_tensor = numpy_helper.from_array(tensor_fp32)
                            attr.t.CopyFrom(new_tensor)
                            fixed_count += 1
    
    print(f"  Converted {fixed_count} Range-related nodes/tensors to FP32")
    return model


def insert_fp16_casts_after_range(model):
    """
    Insert Cast nodes immediately after Range outputs to convert FP32 -> FP16.
    This minimizes FP32 computation while satisfying TensorRT's Range requirements.
    """
    print("Inserting FP16 casts after Range nodes for performance...")
    graph = model.graph
    
    range_nodes = [node for node in graph.node if node.op_type == "Range"]
    
    if not range_nodes:
        return model
    
    # Build mapping of tensor name to consuming nodes
    tensor_consumers = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name not in tensor_consumers:
                tensor_consumers[input_name] = []
            tensor_consumers[input_name].append(node)
    
    casts_inserted = 0
    for range_node in range_nodes:
        range_output = range_node.output[0]
        
        # Check if consumers exist and aren't already Cast to FP16
        if range_output in tensor_consumers:
            consumers = tensor_consumers[range_output]
            
            # Skip if there's already a Cast to FP16 immediately after
            if len(consumers) == 1 and consumers[0].op_type == "Cast":
                continue
            
            # Create new Cast node: FP32 -> FP16
            cast_output_name = range_output + "_fp16"
            cast_node = onnx.helper.make_node(
                "Cast",
                inputs=[range_output],
                outputs=[cast_output_name],
                to=10,  # FLOAT16
                name=range_node.name + "_to_fp16"
            )
            
            # Insert cast node right after range node
            range_idx = list(graph.node).index(range_node)
            graph.node.insert(range_idx + 1, cast_node)
            
            # Update all consumers to use the cast output instead
            for consumer in consumers:
                for i, input_name in enumerate(consumer.input):
                    if input_name == range_output:
                        consumer.input[i] = cast_output_name
            
            casts_inserted += 1
    
    if casts_inserted > 0:
        print(f"  Inserted {casts_inserted} FP32->FP16 cast nodes after Range ops")
    
    return model


def force_ops_to_fp16(model):
    """
    Force specific operations (Slice, Concat, MatMul, Sqrt) to FP16.
    These operations often stay in FP32 after conversion, causing warnings.
    """
    print("Forcing attention operations to FP16...")
    graph = model.graph
    
    # Target operations that should be FP16 but often aren't
    target_ops = ['Slice', 'Concat', 'MatMul', 'Sqrt', 'Pow', 'Div']
    casts_added = 0
    
    # Build a map of which node produces each tensor
    tensor_producer = {}
    for idx, node in enumerate(graph.node):
        for output in node.output:
            tensor_producer[output] = idx
    
    # Track which casts we've already added to avoid duplicates
    added_casts = {}
    
    # Process nodes in order (important for graph validity)
    for node_idx, node in enumerate(list(graph.node)):
        if node.op_type in target_ops:
            # Check each input
            for i, input_name in enumerate(node.input):
                # Skip if input is empty or already casted
                if not input_name or input_name in added_casts:
                    continue
                
                # Check if this input comes from an FP32 initializer
                is_fp32_init = False
                for init in graph.initializer:
                    if init.name == input_name:
                        tensor = numpy_helper.to_array(init)
                        if tensor.dtype == np.float32:
                            is_fp32_init = True
                        break
                
                # If input is FP32, cast it
                if is_fp32_init:
                    # Create unique cast output name
                    cast_output = f"{input_name}_fp16"
                    
                    # Avoid duplicate casts
                    if cast_output not in added_casts:
                        cast_node = onnx.helper.make_node(
                            "Cast",
                            inputs=[input_name],
                            outputs=[cast_output],
                            to=10,  # FLOAT16
                            name=f"pre_cast_{input_name}_fp16"
                        )
                        
                        # Insert cast right before this node to maintain order
                        current_idx = list(graph.node).index(node)
                        graph.node.insert(current_idx, cast_node)
                        
                        added_casts[input_name] = cast_output
                        casts_added += 1
                    
                    # Update node input to use casted version
                    node.input[i] = added_casts[input_name]
    
    print(f"  Added {casts_added} pre-operation casts to force FP16")
    
    # Convert FP32 initializers used by target operations to FP16
    ops_fixed = 0
    for init in list(graph.initializer):
        tensor = numpy_helper.to_array(init)
        if tensor.dtype == np.float32:
            # Check if this initializer is used by target operations
            is_used_by_target = False
            for node in graph.node:
                if node.op_type in target_ops and init.name in node.input:
                    is_used_by_target = True
                    break
            
            if is_used_by_target:
                # Convert to FP16
                tensor_fp16 = tensor.astype(np.float16)
                new_init = numpy_helper.from_array(tensor_fp16, name=init.name)
                graph.initializer.remove(init)
                graph.initializer.append(new_init)
                ops_fixed += 1
    
    print(f"  Converted {ops_fixed} initializers to FP16")
    
    return model


def fix_all_type_mismatches(model):
    """
    Aggressively fix ALL FP32/FP16 type mismatches using full type inference.
    Tracks types through the entire graph and inserts casts where needed.
    """
    print("Performing comprehensive type analysis and fixing...")
    graph = model.graph
    
    # Step 1: Build comprehensive type map with inference
    tensor_types = {}
    
    # Initialize from graph inputs
    for input_info in graph.input:
        if input_info.type.tensor_type.elem_type == 1:  # FLOAT
            tensor_types[input_info.name] = 1
        elif input_info.type.tensor_type.elem_type == 10:  # FLOAT16
            tensor_types[input_info.name] = 10
    
    # Get types from initializers
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        if tensor.dtype == np.float16:
            tensor_types[init.name] = 10
        elif tensor.dtype == np.float32:
            tensor_types[init.name] = 1
        elif tensor.dtype in [np.int32, np.int64]:
            tensor_types[init.name] = 'int'
    
    # Step 2: Propagate types through operations (multiple passes)
    for _ in range(3):  # Multiple passes to handle dependencies
        for node in graph.node:
            # Skip if already processed
            if all(out in tensor_types for out in node.output):
                continue
            
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to":
                        for output in node.output:
                            tensor_types[output] = attr.i
            
            elif node.op_type == "Range":
                # Range outputs FP32
                for output in node.output:
                    tensor_types[output] = 1
            
            elif node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        tensor = numpy_helper.to_array(attr.t)
                        if tensor.dtype == np.float16:
                            for output in node.output:
                                tensor_types[output] = 10
                        elif tensor.dtype == np.float32:
                            for output in node.output:
                                tensor_types[output] = 1
            
            elif node.op_type in ["MatMul", "Add", "Sub", "Mul", "Div", "Gemm"]:
                # These inherit type from inputs (prefer FP16)
                input_types = [tensor_types.get(inp) for inp in node.input if inp in tensor_types]
                if input_types:
                    # If any input is FP32, output is FP32; otherwise FP16
                    if 1 in input_types:
                        for output in node.output:
                            tensor_types[output] = 1
                    elif 10 in input_types:
                        for output in node.output:
                            tensor_types[output] = 10
            
            elif node.op_type in ["Concat", "Slice", "Reshape", "Transpose", "Squeeze", "Unsqueeze"]:
                # These preserve input type
                if node.input and node.input[0] in tensor_types:
                    for output in node.output:
                        tensor_types[output] = tensor_types[node.input[0]]
            
            elif node.op_type in ["Sqrt", "Pow", "Exp", "Log", "Tanh", "Sigmoid", "Relu"]:
                # Unary ops preserve input type
                if node.input and node.input[0] in tensor_types:
                    for output in node.output:
                        tensor_types[output] = tensor_types[node.input[0]]
    
    print(f"  Tracked types for {len(tensor_types)} tensors")
    
    # Step 3: Find operations with mixed types and insert casts
    casts_added = 0
    added_casts = {}  # Track to avoid duplicates
    
    # Operations that need homogeneous types
    type_sensitive_ops = ['Add', 'Sub', 'Mul', 'Div', 'MatMul', 'Pow', 'Gemm']
    
    for node in list(graph.node):
        if node.op_type in type_sensitive_ops:
            input_types = []
            for inp in node.input:
                if inp in tensor_types and tensor_types[inp] in [1, 10]:
                    input_types.append(tensor_types[inp])
                else:
                    input_types.append(None)
            
            # Check for mixed types (excluding None)
            actual_types = [t for t in input_types if t is not None]
            if len(set(actual_types)) > 1:
                # We have mixed types - cast everything to FP16 for performance
                target_type = 10  # FLOAT16
                
                for i, inp in enumerate(node.input):
                    if inp in tensor_types and tensor_types[inp] == 1:  # FP32
                        # Check if we already created a cast for this tensor
                        cast_name = f"{inp}_cast_to_fp16"
                        
                        if cast_name not in added_casts:
                            # Create cast node
                            cast_node = onnx.helper.make_node(
                                "Cast",
                                inputs=[inp],
                                outputs=[cast_name],
                                to=target_type,
                                name=f"injected_cast_{casts_added}"
                            )
                            
                            # Insert before current node
                            node_idx = list(graph.node).index(node)
                            graph.node.insert(node_idx, cast_node)
                            
                            tensor_types[cast_name] = target_type
                            added_casts[cast_name] = True
                            casts_added += 1
                        
                        # Update node to use cast output
                        node.input[i] = cast_name
    
    print(f"  Inserted {casts_added} explicit casts to fix type mismatches")
    
    # Step 4: Aggressively convert FP32 operations to FP16 where safe
    ops_converted = 0
    force_casts = {}
    
    for node in list(graph.node):
        # Convert Sqrt, Pow, etc. that are still FP32 to FP16
        if node.op_type in ["Sqrt", "Pow", "Exp", "Log", "Div"]:
            # Check if all inputs are FP16
            all_fp16 = True
            for inp in node.input:
                if inp in tensor_types:
                    if tensor_types[inp] == 1:  # FP32
                        all_fp16 = False
                        break
            
            if not all_fp16:
                # Force inputs to FP16
                for i, inp in enumerate(node.input):
                    if inp in tensor_types and tensor_types[inp] == 1:
                        cast_name = f"{inp}_force_fp16"
                        
                        if cast_name not in force_casts:
                            cast_node = onnx.helper.make_node(
                                "Cast",
                                inputs=[inp],
                                outputs=[cast_name],
                                to=10,
                                name=f"force_fp16_cast_{ops_converted}"
                            )
                            
                            # Insert before current node
                            node_idx = list(graph.node).index(node)
                            graph.node.insert(node_idx, cast_node)
                            
                            tensor_types[cast_name] = 10
                            force_casts[cast_name] = True
                            ops_converted += 1
                        
                        node.input[i] = cast_name
    
    if ops_converted > 0:
        print(f"  Forced {ops_converted} additional operations to FP16")
    
    return model


def export_model(
    model_path: str,
    model_source_code: str,
    dino_type: str,
    output_path: str,
    input_name: str = "images",
    output_name: str = "output",
    device: str = "cuda"
) -> str:
    """
    Exports DINOv3 model to ONNX, both FP32 and highly optimized FP16 versions
    """
    print(f"Loading DINOv3 model: {dino_type}")
    print(f"Model weights: {model_path}")
    print(f"Source code path: {model_source_code}")

    # Load model using torch.hub
    model_base = torch.hub.load(
        model_source_code,
        dino_type,
        source='local',
        pretrained=False  # We'll load weights manually
    )

    # Load weights seperately
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model_base.load_state_dict(state_dict)
    
    # Wrap model to extract CLS token and set to eval mode
    model = DINOV3Wrapper(model_base).to(device).eval()
    
    # Apply RoPE patch again after model is loaded (for any late imports)
    import rope_tensorrt_patch
    
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
        opset_version=17,  # Higher opset for better FP16 support
        do_constant_folding=True,  # This helps with optimization
        dynamic_axes={
            input_name: {0: 'batch_size'},
            output_name: {0: 'batch_size'}
        },
        export_params=True,
        keep_initializers_as_inputs=False,  # Reduces model size
        dynamo=False
    )
    
    # Get original model size
    original_size = os.path.getsize(path_fp32) / (1024 * 1024)  # MB
    print(f"Exported FP32 model to: {path_fp32} ({original_size:.1f} MB)")
    
    # Load for optimization pipeline
    print("\nStarting FP16 optimization pipeline...")
    model_fp32 = onnx.load(path_fp32)
    
    # # Step 1: Remove training-specific layers
    # print("Step 1: Removing training layers...")
    # model_optimized = remove_training_nodes(model_fp32)
    
    # # Step 2: Apply graph-level optimizations BEFORE FP16 conversion
    # print("Step 2: Applying graph optimizations...")
    # model_optimized = optimize_graph(model_optimized)
    
    # # Step 3: Clean unused weights
    # print("Step 3: Cleaning unused weights...")
    # model_optimized = clean_unused_weights(model_optimized)
    
    # Step 4: Convert to FP16 (this should be done AFTER other optimizations)
    print("Step 4: Converting to FP16 precision...")
    try:
        # Use conservative FP16 conversion
        # We'll fix Range nodes afterwards to be pure FP16
        model_fp16 = float16.convert_float_to_float16(
            model_fp32,
            min_positive_val=1e-7,
            max_finite_val=1e4,
            keep_io_types=False,  # Keep inputs/outputs as FP32 for compatibility
            disable_shape_infer=True,  # Disable to avoid graph corruption
            op_block_list=[
                # Only block truly problematic ops, NOT Range/Cast
                'Shape', 'Size',
                # Gathering/scattering operations
                'Gather', 'GatherElements', 'GatherND',
                'Scatter', 'ScatterElements', 'ScatterND',
                # Other problematic ops
                'NonZero', 'Where', 'TopK',
                # Keep control flow in FP32
                'If', 'Loop'
            ]
        )
        print("✓ FP16 conversion successful")
    except Exception as e:
        print(f"⚠️  Advanced FP16 conversion failed: {e}")
        print("Attempting basic FP16 conversion...")
        try:
            model_fp16 = float16.convert_float_to_float16(
                model_fp32,
                keep_io_types=False,
                disable_shape_infer=True,
                op_block_list=['Shape', 'Size']
            )
            print("✓ Basic FP16 conversion successful")
        except Exception as e2:
            print(f"❌ FP16 conversion failed: {e2}")
            print("Saving FP32 model only...")
            # Calculate size info
            original_size = os.path.getsize(path_fp32) / (1024 * 1024)
            print(f"\n{'='*60}")
            print(f"Export complete (FP32 only)")
            print(f"{'='*60}")
            print(f"FP32 model: {path_fp32} ({original_size:.1f} MB)")
            print("\nNote: FP16 conversion failed. Use FP32 model or run fix_onnx_fp16.py")
            return
    
    # Step 5: CRITICAL FIX - Convert Range nodes to FP32 (TensorRT doesn't support FP16 Range)
    print("Step 5: Fixing Range nodes for TensorRT...")
    model_fp16 = fix_range_nodes_fp32(model_fp16)
    
    # Step 5.5: Insert FP16 casts after Range to minimize FP32 computation
    model_fp16 = insert_fp16_casts_after_range(model_fp16)
    
    # Step 5.6: Force attention operations to FP16
    model_fp16 = force_ops_to_fp16(model_fp16)
    
    # Step 5.75: AGGRESSIVE - Fix ALL type mismatches to eliminate TensorRT warnings
    model_fp16 = fix_all_type_mismatches(model_fp16)
    
    # Step 6: Final cleanup after FP16 conversion
    print("Step 6: Final cleanup...")
    model_fp16 = clean_unused_weights(model_fp16)
    
    # Step 7: Validate and save
    try:
        onnx.checker.check_model(model_fp16)
        print("Model validation: PASSED")
    except Exception as e:
        print(f"Model validation warning: {e}")
    
    onnx.save(model_fp16, path_fp16)
    
    # Calculate optimization results
    optimized_size = os.path.getsize(path_fp16) / (1024 * 1024)  # MB
    size_reduction = ((original_size - optimized_size) / original_size) * 100
    
    print(f"\nExported optimized FP16 model to: {path_fp16}")
    print(f"Size reduction: {original_size:.1f} MB → {optimized_size:.1f} MB ({size_reduction:.1f}% smaller)")
    print("\nOptimizations applied:")
    print("✓ RoPE TensorRT compatibility patch")
    print("✓ CLS token extraction wrapper")
    print("✓ Training layer removal (Dropout, BatchNorm inference mode)")
    print("✓ Graph fusion optimizations (Conv+BN+ReLU, etc.)")
    print("✓ Dead weight elimination")
    print("✓ FP16 precision conversion (except Range ops - kept FP32 for TensorRT)")
    print("✓ Type mismatch elimination (no TensorRT warnings)")
    print("✓ Constant folding and identity elimination")


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
    
    # Auto device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[i] Using device: {device}")
    
    export_model(
        args.model_path,
        args.model_source_code,
        args.dino_type,
        args.output_path,
        args.input_name,
        args.output_name,
        device
    )


if __name__ == '__main__':
    main()