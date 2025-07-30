import os
import onnx
import torch
import torch.nn as nn
from onnxconverter_common import float16
import utils.tal.anchor_generator as _m
import onnxslim

def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ResourceWarning)

def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)

_m.dist2bbox.__code__ = _dist2bbox.__code__

class DeepStreamOutputDual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[1].transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)

class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def yolov9_export(weights, device, inplace=True, fuse=True):
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()

    if not hasattr(ckpt, 'stride'):
        ckpt.stride = torch.tensor([32.])

    if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
        ckpt.names = dict(enumerate(ckpt.names))

    model = ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval()
    for m in model.modules():
        t = type(m)
        if t.__name__ in ('Hardswish', 'LeakyReLU', 'ReLU', 'ReLU6', 'SiLU', 'Detect', 'Model'):
            m.inplace = inplace
        elif t.__name__ == 'Upsample' and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

    model.eval()

    head = 'Detect'
    for k, m in model.named_modules():
        if m.__class__.__name__ in ('Detect', 'DDetect', 'DualDetect', 'DualDDetect'):
            m.inplace = False
            m.dynamic = False
            m.export = True
            head = m.__class__.__name__

    return model, head

def main(args):
    suppress_warnings()

    print(f'\nStarting: {args.model_path}')

    print('Opening YOLOv9 model')

    device = torch.device('cpu')
    model, head = yolov9_export(args.model_path, device)

    if len(model.names.keys()) > 0:
        print('Creating labels.txt file')
        with open('labels.txt', 'w', encoding='utf-8') as f:
            for name in model.names.values():
                f.write(f'{name}\n')

    if head in ('Detect', 'DDetect'):
        model = nn.Sequential(model, DeepStreamOutput())
    else:
        model = nn.Sequential(model, DeepStreamOutputDual())

    # Set input size(Dummy input)
    img_size = args.size * 2 if len(args.size) == 1 else args.size
    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)

    # Set dynamic batching
    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'output': {
            0: 'batch'
        }
    }

    # Export models
    onnx_fp32 = f'{args.model_path}-fp32.onnx'
    onnx_fp16 = f'{args.model_path}-fp16.onnx'
    onnx_slim = f'{args.model_path}-slim.onnx'

    print('Exporting the model to ONNX')
    torch.onnx.export(
        model, 
        onnx_input_im, 
        onnx_fp32, 
        verbose=False, 
        opset_version=12, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes=dynamic_axes
    )

    # Load model to export it in lighter versions
    model_fp32 = onnx.load(onnx_fp32)

    # Converts to slim version
    model_slim = onnxslim.slim(model_fp32)
    onnx.save(model_slim, onnx_slim)

    # This converts all layers into FP16, including input/output
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    onnx.save(model_fp16, onnx_fp16)

    print(f'Successfull exported {args.model_path} to ONNX')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DeepStream YOLOv9 conversion')
    parser.add_argument('--model-path', required=True, help='Input model (.pt) file path (required)')
    parser.add_argument('--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise SystemExit('Invalid model file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
