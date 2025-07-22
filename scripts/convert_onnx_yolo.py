import argparse
import torch
import torch.onnx
from pathlib import Path
from onnxconverter_common import float16
import onnx
import os

def export_model(model_path: str, input_name: str = "images", output_name: str = "output", device: str = "cuda") -> str:
    """
        Exports PT model to onnx, both FP32 and FP16 precisions
    """
    # Load model
    model_base = torch.load(model_path, map_location=device, weights_only=False)
    model = model_base['model'].float().to(device).eval()

    # Wrap model to export only raw predictions
    class WrapperModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            output = self.model(x)[0]

            if isinstance(output, (list, tuple)):
                return output[0]
            else:
                return output

    model = WrapperModel(model).to(device)

    # Dummy input
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    # Define output paths
    model_name = Path(model_path).stem
    path_fp32 = os.path.join(os.getcwd(), f"{model_name}-fp32.onnx")
    path_fp16 = os.path.join(os.getcwd(), f"{model_name}-fp16.onnx")

    # Export to ONNX - FP32
    torch.onnx.export(
        model,
        dummy_input,
        path_fp32,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes={
            input_name: {0: 'batch_size'},
            output_name: {0: 'batch_size'}
        }
    )

    print(f"Exported F32 model to: {path_fp32}")

    # Export to ONNX - FP16
    # This converts all layers into FP16, including input/output
    model_fp32 = onnx.load(path_fp32)
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=False)
    onnx.save(model_fp16, path_fp16)

    print(f"Exported F16 model to: {path_fp16}")
def main():
    parser = argparse.ArgumentParser(description='Export YOLOv9 model to ONNX and test inference.')
    parser.add_argument('--model-path', required=True, help='Path to the .pt PyTorch model')
    parser.add_argument('--input-name', default='images', help='Name of the ONNX model input')
    parser.add_argument('--output-name', default='output', help='Name of the ONNX model output')
    args = parser.parse_args()

    # Auto device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[i] Using device: {device}")

    export_model(
        args.model_path,
        args.input_name, 
        args.output_name, 
        device
    )

if __name__ == '__main__':
    main()