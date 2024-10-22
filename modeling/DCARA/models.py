from functools import partial
import numpy as np

import torch
from torch import nn

from .pvt_v2 import PyramidVisionTransformerV2
from timm.models.vision_transformer import _cfg
from .AFF import AFF,MS_CAM,MS_CAM_with_Fusion

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)



class MSCAMAndConv2dAdd(nn.Module):
    def __init__(self, channels, r=8):
        super(MSCAMAndConv2dAdd, self).__init__()
        # 定义MS_CAM支路
        self.mscam = MS_CAM(channels=channels, r=r)
        # 定义Conv2d支路
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        # 通过MS_CAM支路
        mscam_out = self.mscam(x)
        mscam_out = self.conv(mscam_out)
        # 通过Conv2d支路
        conv_out = self.conv(x)
        # 将两个支路的输出相加
        out = mscam_out + conv_out
        # out = torch.cat([mscam_out, conv_out], dim=1)
        return conv_out,out


class MSCAMAndConv2dAdd_up(nn.Module):
    def __init__(self, channels, r=16):
        super(MSCAMAndConv2dAdd_up, self).__init__()
        # 定义MS_CAM支路
        # self.mscam = MS_CAM(channels=channels, r=r)
        self.mscam = MS_CAM(channels=channels, r=r)
        # 定义Conv2d支路
        # self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
    def forward(self, x):
        # 通过MS_CAM支路
        mscam_out = self.mscam(x)
        # 通过up支路
        mscam_out = self.up(mscam_out)
        conv_out = self.up(x)
        # 将两个支路的输出相加
        out = mscam_out + conv_out
        return out



class RAFC(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=5,
        n_levels_up=5,
        n_RBs=2,
        in_resolution=352,
    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)
        ])
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        n_levels_down = 5  # 确保进行5次下采样

        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                # enc_block_chans.append(ch)
            # 下采样, 在每个level结束时进行
            if level ==0:
                self.enc_blocks.append(
                    # nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2)
                    MSCAMAndConv2dAdd(channels=ch,r=8)
                )
            else:
                self.enc_blocks.append(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2)
                # MSCAMAndConv2dAdd(channels=ch)
                )
            enc_block_chans.append(ch)


        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))


        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]
            ch_cat = ch + enc_block_chans.pop()  # 更新通道数以包括cat操作
            self.dec_blocks.append(
                RB(
                    ch_cat,
                    min_channel_mult * min_level_channels,
                )
            )
            ch = min_channel_mult * min_level_channels  # 更新通道数为RB操作后的值

            # 第二个RB，不进行cat操作
            self.dec_blocks.append(
                RB(
                    ch,
                    ch,  # 这里通道数不变，因为没有额外融合
                )
            )

            # 添加上采样层
            # 在RB后面添加上采样，可能需要调整通道数
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    # MS_CAM(channels=ch,r=16),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    # MSCAMAndConv2dAdd_up(channels=ch)
                )
            )
            # self.dec_blocks.append(nn.Sequential(*layers))
                # 添加额外的残差块
        extra_RBs = [RB(ch, ch) for _ in range(2)]
        self.dec_blocks.extend(extra_RBs)

        # 创建特征处理层的列表
        input_channels = [32, 32, 64, 64]  # 这里是每层的输入通道数
        output_channels = [32, 64, 64, 128]  # 这里是每层的输出通道数

        self.enf = nn.ModuleList([
            nn.Sequential(
                MS_CAM(channels=in_ch, r=8),  # 假设 MS_CAM 接收 channels 和 r 作为参数
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            ) for in_ch, out_ch in zip(input_channels, output_channels)
        ])


    def forward(self, x):
        # 编码器部分
        enc_features = []
        out = self.enc_blocks[0](x)
        enc_features.append(out)
        
     
        
        # 初始化通道维度索引
        channel_idx = 0

        for idx, layer in enumerate(self.enc_blocks[1:], 1):
            if isinstance(layer, MSCAMAndConv2dAdd):
                # 特别处理 MSCAMAndConv2dAdd 层
                out, out_g = layer(out)  # 获取两个输出
                # print(out.shape)
                enc_features.append(out_g)  # 将 out_g 作为跳连接的张量添加到 enc_features

            elif isinstance(layer, nn.Conv2d):

                out = layer(out)
                out_g = self.enf[channel_idx](out_g)
                out_g = out_g + out
                enc_features.append(out_g)
                channel_idx += 1

            else:
                out = layer(out)
                # print(out.shape)    



        out = self.middle_block(out)

            # 解码器部分
        for level in range(5):
            # 解码器的残差块前进行拼接
            out = torch.cat([out, enc_features[-(level+1)]], dim=1)
            out = self.dec_blocks[level * 3](out)
            out = self.dec_blocks[level * 3 + 1](out)

            # 解码器的上采样
            out = self.dec_blocks[level * 3 + 2](out)

        # 额外的残差块
        out = self.dec_blocks[-2](out)
        out = self.dec_blocks[-1](out)

        return out


class PVT(nn.Module):
    def __init__(self):

        super().__init__()

        backbone = PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("pvt_v2_b3.pth")
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                nn.Conv2d([64, 128, 320, 512][i], 64, kernel_size=1, stride=1, padding=0),nn.Upsample(size=88),
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

        self.AFFs = nn.ModuleList([AFF(channels=64) for _ in range(3)])  #

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        pyramid_new = []
        
        for i in range(len(pyramid_emph) -1, 0,-1):
            aff = self.AFFs[i - 1]  # 选择对应的AFF实例
            fused_tensor = aff(pyramid_emph[i-1], pyramid_emph[i])
            pyramid_new.append(fused_tensor)
        pyramid_new.append(pyramid_emph[-1])


        l_i = pyramid_new[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_new[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l


class DCARA(nn.Module):
    def __init__(self, nclass=5,size=352):

        super().__init__()

        self.PVT = PVT()

        self.RAFC = RAFC(in_resolution=size)
        self.PH = nn.Sequential(
            RB(64 + 32, 64), RB(64, 64), nn.Conv2d(64, nclass, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)

    def forward(self, x):
        x1 = self.PVT(x)
        x2 = self.RAFC(x)
        x1 = self.up_tosize(x1)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)

        return out