import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Union
import numpy as np

def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_conv3d_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = "relu",
    norm: Optional[Union[str, Tuple]] = "batch",
    dropout: Optional[float] = None,
    bias: bool = False,
    conv_only: bool = True,
):
    padding = get_padding(kernel_size, stride)

    # Convolution layer (transposed or regular)

    conv_layer = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

    # Add activation
    act_layer = None
    if act == "relu":
        act_layer = nn.ReLU(inplace=True)
    elif act == "prelu":
        act_layer = nn.PReLU(num_parameters=out_channels)
    elif act == "sigmoid":
        act_layer = nn.Sigmoid()

    # Add normalization
    norm_layer = None
    if norm == "batch":
        norm_layer = nn.BatchNorm3d(out_channels)
    elif norm == "instance":
        norm_layer = nn.InstanceNorm3d(out_channels)

    # Add dropout
    dropout_layer = None
    if dropout is not None:
        dropout_layer = nn.Dropout3d(p=dropout)

    # Combine layers
    layers = [conv_layer]
    if not conv_only:
        if norm_layer:
            layers.append(norm_layer)
        if act_layer:
            layers.append(act_layer)
        if dropout_layer:
            layers.append(dropout_layer)

    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """
    A skip-connection-based residual block for U-Net architectures.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of the convolutional kernel.
        stride: stride for the convolution.
        norm_name: normalization type ("batch", "instance", or "group").
        act_name: activation function type ("relu", "leakyrelu", etc.).
        dropout: dropout probability.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: str = "instance",
            act_name: str = "leakyrelu",
            dropout: Optional[float] = None,
    ):
        super().__init__()

        # Convolutional layers
        self.conv1 = get_conv3d_layer(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout=dropout, conv_only=True
        )
        self.conv2 = get_conv3d_layer(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        # Activation function
        if act_name == "leakyrelu":
            self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif act_name == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation type: {act_name}")

        # Normalization
        if norm_name == "batch":
            self.norm1 = nn.BatchNorm3d(out_channels)
            self.norm2 = nn.BatchNorm3d(out_channels)
        elif norm_name == "instance":
            self.norm1 = nn.InstanceNorm3d(out_channels)
            self.norm2 = nn.InstanceNorm3d(out_channels)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_name}")
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv3d_layer(
                in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
            )
            self.norm3 = (
                nn.BatchNorm3d(out_channels)
            ) if norm_name == "batch" else (
                nn.InstanceNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x

        # First convolution block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        # Second convolution block
        out = self.conv2(out)
        out = self.norm2(out)

        # Downsampling residual if necessary
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)

        # Add residual connection
        out += residual
        out = self.activation(out)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        # Reduction ratio to control the bottleneck size
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # Output shape: (b, c, 1, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)  # Output shape: (b, c, 1, 1, 1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w, d = x.shape

        # Global average pooling
        avg_out = self.avg_pool(x).view(b, c)  # Shape: (b, c)
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))  # Shape: (b, c)

        # Global max pooling
        max_out = self.max_pool(x).view(b, c)  # Shape: (b, c)
        max_out = self.fc2(self.relu(self.fc1(max_out)))  # Shape: (b, c)

        # Combine and apply sigmoid
        out = avg_out + max_out  # Shape: (b, c)
        scale = self.sigmoid(out).view(b, c, 1, 1, 1)  # Shape: (b, c, 1, 1, 1)

        # Scale the input tensor
        return x * scale  # Shape: (b, c, h, w, d)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # operation on spatial dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(out))
        return attn * x

class IPPA(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, plane_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.head_dim = hidden_size // num_heads
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(plane_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)  # 4, B, num_heads, N, C // num_heads

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)  # 4, B, num_heads, C // num_heads, N
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)
        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x


class CPPA(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, plane_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_ca = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_sa = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(plane_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x, q):
        B, N, C = x.shape

        q_shared = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k_shared = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads)
        v_CA = self.v_ca(x).reshape(B, N, self.num_heads, C // self.num_heads)
        v_SA = self.v_sa(x).reshape(B, N, self.num_heads, C // self.num_heads)

        q_shared = q_shared.permute(0, 2, 1, 3)  # B, num_heads, N, C // num_heads
        k_shared = k_shared.permute(0, 2, 1, 3)
        v_CA = v_CA.permute(0, 2, 1, 3)
        v_SA = v_SA.permute(0, 2, 1, 3)

        q_shared = q_shared.transpose(-2, -1)  # B, num_heads, C // num_heads, N
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)  # B, num_heads, C // num_heads, N_pj

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature  # B, num_heads, C // num_heads, C // num_heads

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2  # B, num_heads, N, N_pj

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

class PlaneTransformer(nn.Module):
    """
        A transformer block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            plane: str = "transverse",  # default to transverse plane
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.ippa_block = IPPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads,
                             channel_attn_drop=dropout_rate, plane_attn_drop=dropout_rate)
        self.conv51 = ResBlock(hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="instance")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            # feature_size = int(math.sqrt(input_size))
            # self.pos_embed = nn.Parameter(torch.zeros(1, hidden_size, feature_size, feature_size, feature_size))
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        self.plane = plane
    def forward(self, x):
        B, C, H, W, D = x.shape
        # if self.pos_embed is not None:
        #     x = x + self.pos_embed
        # x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        # plane
        if self.plane == "transverse":
            x = x.permute(0, 4, 2, 3, 1).reshape(B * D, H * W, C)
        elif self.plane == "sagittal":
            x = x.permute(0, 2, 3, 4, 1).reshape(B * H, W * D, C)
        # x = rearrange(x, "B C H W D -> (B D) (H W) C")
        # x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.ippa_block(self.norm(x))

        # attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        # plane
        if self.plane == "transverse":
            attn_skip = attn.reshape(B, D, H, W, C).permute(0, 4, 2, 3, 1)
        elif self.plane == "sagittal":
            attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        # attn_skip = rearrange(attn, "(B D) (H W) C -> B C H W D", B=B, C=C, H=H, W=W)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x

class CrossPlaneTransformer(nn.Module):
    """
        A transformer block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            plane: str = "transverse",  # default to transverse plane
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.cppa_block = CPPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads,
                               channel_attn_drop=dropout_rate, plane_attn_drop=dropout_rate)
        self.conv51 = ResBlock(hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")  # batch instance
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
            self.pos_embed2 = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        self.plane = plane
    def forward(self, x, q):  # x is the main view features waiting to be refined, q is the other view features.
        B, C, H, W, D = x.shape

        if self.plane == "transverse":
            x = x.permute(0, 4, 2, 3, 1).reshape(B * D, H * W, C)
            q = q.permute(0, 4, 2, 3, 1).reshape(B * D, H * W, C)
        elif self.plane == "sagittal":
            x = x.permute(0, 2, 3, 4, 1).reshape(B * H, W * D, C)
            q = q.permute(0, 2, 3, 4, 1).reshape(B * H, W * D, C)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.cppa_block(self.norm(x), self.norm2(q))

        # attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        # plane
        if self.plane == "transverse":
            attn_skip = attn.reshape(B, D, H, W, C).permute(0, 4, 2, 3, 1)
        elif self.plane == "sagittal":
            attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        # attn_skip = rearrange(attn, "(B D) (H W) C -> B C H W D", B=B, C=C, H=H, W=W)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x
