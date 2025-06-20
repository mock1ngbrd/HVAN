import os
import torch
import torch.nn as nn
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from einops import rearrange
from model_components import *
from attn_block import *

class HVAN(nn.Module):
    def __init__(self, in_channels=1, n_class=2, input_size=[32, 16, 8, 4], dims=[64, 128, 256, 512],
                 proj_size=[128, 64, 64, 32], num_heads=[4, 8, 8, 16], transformer_dropout_rate=0.1):
        super().__init__()

        block_nums = [2, 2, 2, 2]
        block = BasicBlock
        self.in_channels = dims[0]
        self.stem_layer_tv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, bias=False, stride=2),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))
        self.stem_layer_sv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, bias=False, stride=2),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))

        self.res_blocks_tv = nn.ModuleList()
        for i in range(4):
            self.res_blocks_tv.append(self._make_layer(block, dims[i], block_nums[i], 2))

        self.in_channels = dims[0]
        self.res_blocks_sv = nn.ModuleList()
        for i in range(4):
            self.res_blocks_sv.append(self._make_layer(block, dims[i], block_nums[i], 2))

        self.iva_sv = nn.ModuleList()
        for i in range(4):
            self.iva_sv.append(PlaneTransformer(input_size=input_size[i]**2, hidden_size=dims[i], num_heads=num_heads[i],
                                 dropout_rate=transformer_dropout_rate, pos_embed=True, proj_size=proj_size[i],
                                 plane='sagittal'))

        self.iva_tv = nn.ModuleList()
        for i in range(4):
            self.iva_tv.append(PlaneTransformer(input_size=input_size[i]**2, hidden_size=dims[i], num_heads=num_heads[i],
                                 dropout_rate=transformer_dropout_rate, pos_embed=True, proj_size=proj_size[i],
                                 plane='transverse'))

        self.cva_tv = nn.ModuleList()
        for i in range(4):
            self.cva_tv.append(CrossPlaneTransformer(input_size=input_size[i]**2, hidden_size=dims[i], num_heads=num_heads[i],
                                 dropout_rate=transformer_dropout_rate, pos_embed=True, proj_size=proj_size[i],
                                 plane='sagittal'))

        self.cva_sv = nn.ModuleList()
        for i in range(4):
            self.cva_sv.append(
                CrossPlaneTransformer(input_size=input_size[i]**2, hidden_size=dims[i], num_heads=num_heads[i],
                                      dropout_rate=transformer_dropout_rate, pos_embed=True, proj_size=proj_size[i],
                                      plane='transverse'))

        self.fusion_block = Fusion(in_channels=1024)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.cls_head = nn.Linear(1024, n_class)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    @staticmethod
    def get_model_py_path():
        return os.path.abspath(__file__)

    def forward(self, x_tv, x_sv):
        b, _, _, _, _ = x_tv.size()
        x_tv = self.stem_layer_tv(x_tv)
        x_sv = self.stem_layer_sv(x_sv)

        for i in range(4):
            x_tv = self.res_blocks_tv[i](x_tv)
            x_tv_plane = self.plane_attn_tv[i](x_tv)

            x_sv = self.res_blocks_sv[i](x_sv)
            x_sv_plane = self.plane_attn_sv[i](x_sv)

            x_tv = self.cva_tv[i](x_tv_plane, x_sv_plane)
            x_sv = self.cva_sv[i](x_sv_plane, x_tv_plane)

        x = self.fusion_block(x_tv, x_sv)
        x = self.pool(x).view(b, -1)
        x = self.cls_head(x)
        return x

def benchmark_model(model, name):
    model.eval()
    NUM_RUNS = 30
    WARM_UP = 10
    with torch.no_grad():
        # Warm-up
        for _ in range(WARM_UP):
            _ = model(tensor_input1, tensor_input2)

        # Timed runs
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(NUM_RUNS):
            _ = model(tensor_input1, tensor_input2)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_time = end_time - start_time
        time_per_batch = total_time / NUM_RUNS
        time_per_sample = time_per_batch / BATCH_SIZE

        print(f"{name} - Time per sample: {time_per_sample:.4f} seconds")

if __name__ == '__main__':
    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    BATCH_SIZE = 1
    INPUT_SHAPE = (BATCH_SIZE, 1, 128, 128, 128)
    tensor_input1 = torch.randn(INPUT_SHAPE).cuda()
    tensor_input2 = torch.randn(INPUT_SHAPE).cuda()
    model = HVAN(1, 1).cuda()
    benchmark_model(model, name="ours")
