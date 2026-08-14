import torch
import sys
from enum import Enum

class ModelType(Enum):
    YOLOV9 = "YOLOV9"
    DINOV3 = "DINOV3"
    YOLO26 = "YOLO26"

def get_yolov9_model(model_source_path: str, model_path: str) -> torch.nn.Module:
    class YOLOV9Wrapper(torch.nn.Module):
        """ Wrapper for YOLOv9 model to adjust output format. Using YOLOV9-Converted """

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            output = self.model(x)
            # Extract BBOXes only
            return output[0]

    # Add model dependencies to path
    sys.path.insert(0, model_source_path)

    # Load model
    model_base = torch.load(model_path, weights_only=False)
    model = YOLOV9Wrapper(model_base['model']).eval()

    return model

def get_dinov3_model(model_source_code: str, model_path: str, dino_type: str) -> torch.nn.Module:
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
        pretrained=False
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

    return model

def get_yolo26_model(model_path: str, max_det: int = 300) -> torch.nn.Module:
    class YOLO26Wrapper(torch.nn.Module):
        """ Wrapper for YOLO26 model to adjust output format. End2end / NMS-free

        Output is (batch, max_det, 6), each row being
        [x1, y1, x2, y2, score, class_id] - xyxy in pixels of the letterboxed
        network input, sorted by score descending. Every row is returned
        unfiltered, so the consumer applies its own confidence threshold.
        """

        def __init__(self, model, max_det):
            super().__init__()
            self.model = model
            self.max_det = max_det

        def forward(self, x):
            output = self.model(x)
            # Extract BBOXes only - already deduplicated, no NMS needed
            return output.view(-1, self.max_det, 6)

    from ultralytics import YOLO

    # Load model (picks EMA weights when present) and fold BN into conv
    model_base = YOLO(model_path).model.float().eval()
    model_base = model_base.fuse().eval()

    # Switch the detection head into ONNX export mode
    head = model_base.model[-1]
    if not getattr(head, 'end2end', False):
        raise ValueError('Expected an end2end (NMS-free) YOLO26 detection head')

    head.export = True    # emit a bare tensor, and make TopK's k a constant
    head.format = 'onnx'
    head.dynamic = False  # bake anchors as graph constants (fixed HxW)
    head.max_det = max_det
    head.shape = None     # drop the stale cached shape so anchors regenerate

    model = YOLO26Wrapper(model_base, max_det).eval()

    return model
