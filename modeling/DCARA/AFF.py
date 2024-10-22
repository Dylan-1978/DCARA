import torch
import torch.nn as nn
import torch.nn.functional as F




class MS_CAM(nn.Module):


    def __init__(self, channels=64, r=8      
                 ):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)
        # vss_channels = int(channels // channels)


        # 局部注意力升级：加入3x3卷积路径
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att_3x3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            # nn.ReLU(inplace=True),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            # nn.Sigmoid()
        )
     
      
        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数γ
        # self.scale = nn.Parameter(torch.full((1,), -0.5))  # 初始化为接近0的负值


    def forward(self, x):
        xl = self.local_att(x) + self.local_att_3x3(x)  # 
        xg = self.global_att(x)

        gamma = torch.sigmoid(self.scale)  # 确保γ在[0,1]范围内
        xlg = gamma * xl + (1 - gamma) * xg  # 使用动态权重融合局部和全局注意力
        wei = self.sigmoid(xlg)
        return x * wei


class MultiScaleDilatedConv(nn.Module):
    def __init__(self, in_channels,  reduction_ratio=8, dilations=[1, 2, 3,4]):
        super(MultiScaleDilatedConv, self).__init__()
        
        # 降维
        self.reduce = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        
        # 使用不同膨胀率的卷积层
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(in_channels // reduction_ratio, 2*in_channels // reduction_ratio, kernel_size=3, 
                      padding=dilation, dilation=dilation)
            for dilation in dilations
        ])

    def forward(self, x):
        # 降维
        reduced = self.reduce(x)
        
        # 获取不同膨胀率的特征图
        features = [dilated_conv(reduced) for dilated_conv in self.dilated_convs]
        
        # 串联特征图
        output = torch.cat(features, dim=1)
        
        return output


class AFF(nn.Module):


    def __init__(self, channels=64, r=8):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力升级：加入1x1和多尺度膨胀卷积路径
        self.local_att_1x1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        # 使用多尺度膨胀卷积增强特征提取
        self.local_att_dilated = MultiScaleDilatedConv(in_channels=channels, reduction_ratio=8)

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, residual):
        xa = x + residual
        xl_1x1 = self.local_att_1x1(xa)
        xl_dilated = self.local_att_dilated(xa)
        xl = xl_1x1 + xl_dilated
        xg = self.global_att(xa)

        gamma = torch.sigmoid(self.scale)
        xlg = gamma * xl + (1 - gamma) * xg
        
        wei = self.sigmoid(xlg)
        xo = x * wei + residual * (1 - wei)
        # xo = residual * wei + x * (1 - wei)
        return xo





class MS_CAM_with_Fusion(nn.Module):


    def __init__(self, channels=64, r=8):
        super(MS_CAM_with_Fusion, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力升级
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att_3x3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力升级
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.fusion_weight = nn.Parameter(torch.randn(1))  # 可学习的融合权重参数

    def forward(self, x):
        local_features = self.local_att(x) + self.local_att_3x3(x)  # 局部特征提取
        global_features = self.global_att(x)  # 全局特征提取
        
        # 使用可学习的融合权重动态调整局部和全局特征的重要性
        fusion_weight_normalized = torch.sigmoid(self.fusion_weight)  # 确保融合权重在0到1之间
        fused_features = (1 - fusion_weight_normalized) * local_features + fusion_weight_normalized * global_features
        
        output = x * self.sigmoid(fused_features)  # 应用注意力加权
        return output
