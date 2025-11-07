import torch
import sys
from enum import Enum

class ModelType(Enum):
    YOLOV9 = "YOLOV9"
    DINOV3 = "DINOv3"

def get_yolov9_model(model_source_path: str, model_path: str) -> tuple[torch.nn.Module, torch.Tensor]:
    class YOLOV9Wrapper(torch.nn.Module):
        """ Wrapper for YOLOv9 model to adjust output format """

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            output = self.model(x)
            # Extract BBOXes only
            return output[0][0]

    # Add model dependencies to path
    sys.path.insert(0, model_source_path)

    # Load model
    model_base = torch.load(model_path, weights_only=False)
    model = model_base['model'].eval()
    model = YOLOV9Wrapper(model)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)

    return model, dummy_input

def get_dinov3_model(model_source_code: str, model_path: str, dino_type: str) -> tuple[torch.nn.Module, torch.Tensor]:
    class DINOV3Wrapper(torch.nn.Module):
        """Wrapper to extract CLS token from DINOv3 model output"""
        
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            # Get the full output from DINOv3
            output = self.model.forward_features(x)
            
            # Extract CLS token (first token in sequence)
            cls_token = output['x_norm_clstoken']
            
            return cls_token

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
        map_location='cpu', 
        weights_only=True
    )
    model_base.load_state_dict(state_dict)
    
    # Wrap model to extract CLS token and set to eval mode
    model = DINOV3Wrapper(model_base).eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    return model, dummy_input

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