import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
#%%
class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors
#%%
def read_cfg(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    lines: list[str] = list(map(lambda x: x.strip(), lines))
    lines = [line for line in lines if len(line) > 0 and line[0] != '#']

    blocks = []
    for line in lines:
        if line.startswith('['):
            blocks.append({})
            blocks[-1]['type'] = line.strip('[]')
        else:
            key, value = map(lambda x: x.strip(), line.split('='))

            try:
                value = float(value)
            except Exception:
                pass
            else:
                value = int(value) if float(value).is_integer() else float(value)

            blocks[-1][key] = value

    return blocks
#%%
def create_modules(blocks: list[dict[str, str | int | float]]):
    net = blocks[0]
    del blocks[0]

    prev_filters = 3
    filters = 3

    out_filters = []

    modules = nn.ModuleList()
    for i, block in enumerate(blocks):
        module = nn.Sequential()
        block_type = block['type']

        if block_type == 'convolutional':
            batch_normalize = block.get('batch_normalize', 0)
            is_bias = not batch_normalize
            filters = block['filters']
            kernel_size = block['size']
            stride = block['stride']
            padding = block['pad']
            activation = block['activation']

            module.add_module(
                f"conv_{i}",
                nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=is_bias)
            )

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{i}", bn)

            if activation == 'leaky':
                activation = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{i}", activation)
        elif block_type == 'upsample':
            stride = block['stride']

            module.add_module(
                f"upsample_{i}",
                nn.Upsample(scale_factor=stride, mode='bilinear')
            )
        elif block_type == 'route':
            start, end = block['layers'], 0
            if isinstance(start, str):
                _split = map(int, start.split(','))
                start, end = _split

            if start > 0:
                start -= i
            if end > 0:
                end -= i

            module.add_module(f"route_{i}", EmptyLayer())

            if end < 0:
                filters = out_filters[i + start] + out_filters[i + end]
            else:
                filters = out_filters[i + start]
        elif block_type == 'shortcut':
            module.add_module(
                f"shortcut_{i}", EmptyLayer()
            )
        elif block_type == 'yolo':
            mask = list(map(int, block['mask'].split(',')))
            anchors = list(map(int, block['anchors'].split(',')))
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)

            module.add_module(f"detection_{i}", detection)

        modules.append(module)
        prev_filters = filters
        out_filters.append(prev_filters)

    return net, modules
#%%
blocks = read_cfg('detector/yolov3.cfg')
print(create_modules(blocks))