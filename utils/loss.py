import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'bce':
            return self.BCE_Loss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.dice_loss
        elif mode == 'soft_dice':
            return self.soft_dice_loss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss
    
    def BCE_Loss(self, logit, target):
        device = logit.device  # 确定 logit 所在的设备
        if self.weight is not None:
            self.weight = self.weight.to(device)  # 确保 weight 在同一设备上

        criterion = nn.BCEWithLogitsLoss(weight=self.weight)

        # 确保 target 在同一设备上
        target = target.to(device)

        # 确保 target 的尺寸与 logit 一致
        if target.dim() == 3:
            target = target.unsqueeze(1)  # 从 [10, 352, 352] 变为 [10, 1, 352, 352]
        elif target.size(1) != logit.size(1):
            raise ValueError(f"Expected target to have {logit.size(1)} channels, but got {target.size(1)} channels instead.")

        # 对目标标签进行归一化处理
        target = target / 255.0  # 假设目标标签的最大值是255

        # 打印logit和target的范围
        # print(f'logit range: {logit.min().item()}, {logit.max().item()}')
        # print(f'target range: {target.min().item()}, {target.max().item()}')

        # 计算二值交叉熵损失
        loss = criterion(logit, target)

        # 如果 batch_average 为 True, 则对 loss 进行 batch 平均
        if self.batch_average:
            loss /= logit.size(0)

        return loss

    def soft_dice_loss(self, logits, targets, smooth=1.0):
        device = logits.device  # 确定 logits 所在的设备
        targets = targets.to(device)  # 确保 targets 在同一设备上

        # 确保 targets 的尺寸与 logits 一致
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)  # 从 [10, 352, 352] 变为 [10, 1, 352, 352]
        elif targets.size(1) != logits.size(1):
            raise ValueError(f"Expected targets to have {logits.size(1)} channels, but got {targets.size(1)} channels instead.")

        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        
        # 计算交集
        intersection = m1 * m2

        # 计算 Dice 系数
        score = (
            2.0 * intersection.sum(1) + smooth
        ) / (m1.sum(1) + m2.sum(1) + smooth)
        
        # 计算 Dice 损失
        score = 1 - score.sum() / num
        
        # 确保损失值非负
        score = torch.clamp(score, min=0)
        # print(score)
        return score

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):   # 2 0.5  300epochs: 0.5701
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss





    def dice_loss(self,inputs, target, beta=1, smooth=1e-5):
        n, c, h, w = inputs.size()
        # 将 target 转换为 one-hot 形式
        target=target.long()
        # 打印调试信息
        # print(f"Target shape: {target.shape}")
        # print(f"Target unique values: {torch.unique(target)}")
        # print(f"Num classes: {num_classes}")
        target_one_hot = F.one_hot(target, num_classes=c).permute(0, 3, 1, 2).float()
        
        if inputs.shape[2:] != target_one_hot.shape[2:]:
            inputs = F.interpolate(inputs, size=target_one_hot.shape[2:], mode="bilinear", align_corners=True)
        
        inputs = torch.softmax(inputs, dim=1)
        
        # 展平 inputs 和 target_one_hot
        inputs_flat = inputs.view(n, c, -1)
        target_flat = target_one_hot.view(n, c, -1)
        
        # 计算 true positives, false positives, 和 false negatives
        tp = torch.sum(target_flat * inputs_flat, dim=2)
        fp = torch.sum(inputs_flat, dim=2) - tp
        fn = torch.sum(target_flat, dim=2) - tp
        
        # 计算 Dice 分数
        dice_score = ((1 + beta**2) * tp + smooth) / ((1 + beta**2) * tp + beta**2 * fn + fp + smooth)
        dice_loss = 1 - torch.mean(dice_score)
        
        return dice_loss
    
##### Adaptive tvMF Dice loss #####
class Adaptive_tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(Adaptive_tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes

    ### one-hot encoding ###
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    ### tvmf dice loss ###
    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        smooth = 1.0

        score = F.normalize(score, p=2, dim=[0,1,2])
        target = F.normalize(target, p=2, dim=[0,1,2])
        cosine = torch.sum(score * target)
        intersect =  (1. + cosine).div(1. + (1.- cosine).mul(kappa)) - 1.
        loss = (1 - intersect)**2.0

        return loss

    ### main ###
    def forward(self, inputs, target, kappa=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0

        for i in range(0, self.n_classes):
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], kappa[i])
            loss += tvmf_dice
        return loss / self.n_classes

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




