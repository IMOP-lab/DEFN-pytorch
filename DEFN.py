import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
import torch.nn.functional as F


import torch
import pytorch_lightning as pl
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class HSE(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim),
        )
        self.se = SEBlock(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.block(x)
        out = self.se(out)
        out = self.drop_path(out)
        out += identity
        return out

class FuGH(nn.Module):
  def __init__(self, channels, groups):
    super(FuGH, self).__init__()
    self.group_linear1 = nn.Conv3d(channels, channels, kernel_size=1, groups=groups)
    self.gelu = nn.GELU()
    self.group_linear2 = nn.Conv3d(channels, channels, kernel_size=1, groups=groups)
    
  def forward(self, x):
    x_fft = torch.fft.fftn(x, dim=(2, 3, 4))
    x_fft_real = torch.real(x_fft)
    x_fft_imag = torch.imag(x_fft)  

    y_real = self.group_linear1(x_fft_real)
    y_real = self.gelu(y_real)
    y_real = self.group_linear2(y_real)
    y_real = y_real + x_fft_real

    y_imag = self.group_linear1(x_fft_imag)
    y_imag = self.gelu(y_imag)
    y_imag = self.group_linear2(y_imag)
    y_imag = y_imag + x_fft_imag

    y = torch.complex(y_real, y_imag)

    y_ifft = torch.fft.ifftn(y, dim=(2, 3, 4))
    y_ifft_real = y_ifft.real

    return y_ifft_real


class HSE_conv(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
                FuGH(channels=in_chans,groups=in_chans),
                nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                FuGH(channels=dims[i],groups=dims[i]),
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
                LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[HSE(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        FuGHBlock = FuGH(channels=in_chns, groups=in_chns)
        self.add_module('FuGH',FuGHBlock)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x
    

class S3DSA(torch.nn.Module):
    def __init__(self, spatial_dims, in_channels):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels

        self.conv = torch.nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        out = x * attention
        return out

class DEFN(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 64, 128, 256, 512, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        depths=[2, 2, 2, 2],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 512,
        conv_block: bool = True,
        res_block: bool = True,
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        self.hidden_size = hidden_size
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = features[:4]
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []

        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims


        self.HSE_conv = HSE_conv(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )

        
        def create_encoder(spatial_dims, in_chans, out_chans, norm, res_block):
            return UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=3,
                stride=1,
                norm_name=norm,
                res_block=res_block
            )
        
        
        self.encoder1 = create_encoder(spatial_dims, self.in_chans, self.feat_size[0], norm, res_block)
        self.encoders = nn.ModuleList(
            [create_encoder(spatial_dims, self.feat_size[i], self.feat_size[i + 1], norm, res_block) for i in range(3)]
        )

        self.encoder_hidden = create_encoder(spatial_dims, self.feat_size[3], self.hidden_size, norm, res_block)

        def create_decoder(spatial_dims, in_chans, out_chans, norm, res_block):
            return UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm,
                res_block=res_block
            )
        
        self.decoders = nn.ModuleList(
            [create_decoder(spatial_dims, self.feat_size[i + 1], self.feat_size[i], norm, res_block) for i in range(3)]
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)
        self.decoder5 = create_decoder(spatial_dims, self.hidden_size, self.feat_size[3], norm, res_block)
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm,
            res_block=res_block,
        )

        self.spatial_attention = S3DSA(3, self.in_chans)

    def forward(self, x: torch.Tensor):
        x = self.spatial_attention(x)
        outs = self.HSE_conv(x)
        enc1 = self.encoder1(x)
        x2 = outs[0]
        enc2 = self.encoders[0](x2)
        x3 = outs[1]
        enc3 = self.encoders[1](x3)
        x4 = outs[2]
        enc4 = self.encoders[2](x4)
        enc_hidden=self.encoder_hidden(outs[3])

        x0 = self.conv_0(x)+enc1
        x1 = self.down_1(x0)+enc2
        x2 = self.down_2(x1)+enc3
        x3 = self.down_3(x2)+enc4
        x4 = self.down_4(x3)+enc_hidden

        dec3 = self.decoder5(x4, x3)
        dec2 = self.decoders[2](dec3, x2)
        dec1 = self.decoders[1](dec2, x1)
        dec0 = self.decoders[0](dec1, x0)
        out = self.decoder1(dec0)

        return self.out(out)







