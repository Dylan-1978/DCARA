import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def dice_coefficient(self):
        smooth = 1e-6  # 平滑项
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = np.sum(self.confusion_matrix, axis=1)
        predicted_set = np.sum(self.confusion_matrix, axis=0)
        union = ground_truth_set + predicted_set
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice = np.mean(dice)
        return dice

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(np.nan_to_num(Acc, nan=0.0, posinf=0.0, neginf=0.0))
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU_per = np.nan_to_num(MIoU, nan=0.0, posinf=0.0, neginf=0.0)
        MIoU = np.mean(MIoU_per)
        return MIoU, MIoU_per

    def Specificity(self):
        smooth = 1e-6  # 平滑项
        true_negatives = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        false_positives = np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)
        Sp = true_negatives / (true_negatives + false_positives + smooth)
        Sp = np.nanmean(np.nan_to_num(Sp, nan=0.0, posinf=0.0, neginf=0.0))
        return Sp

    def Sensitivity(self):
        Se = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1))
        Se = np.nanmean(np.nan_to_num(Se, nan=0.0, posinf=0.0, neginf=0.0))
        return Se

    def Pr(self):
        Pr = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        Pr = np.nan_to_num(Pr, nan=0.0, posinf=0.0, neginf=0.0)
        return Pr

    def F1_score(self):
        # 计算灵敏度（Se）和精确度（Pr），并添加平滑项以避免除以零
        sum_axis1 = np.sum(self.confusion_matrix, axis=1)
        sum_axis0 = np.sum(self.confusion_matrix, axis=0)
        Se = np.diag(self.confusion_matrix) / (sum_axis1 + 1e-6)  # 添加平滑项
        Pr = np.diag(self.confusion_matrix) / (sum_axis0 + 1e-6)  # 添加平滑项

        # 检查Se和Pr中是否有无效值，并将其替换为0
        Se = np.nan_to_num(Se, nan=0.0, posinf=0.0, neginf=0.0)
        Pr = np.nan_to_num(Pr, nan=0.0, posinf=0.0, neginf=0.0)

        # 计算F1分数
        F1_score = np.zeros_like(Se)
        valid_mask = (Pr + Se) > 0
        F1_score[valid_mask] = 2 * Pr[valid_mask] * Se[valid_mask] / (Pr[valid_mask] + Se[valid_mask])
        
        # 计算F1分数的平均值
        F1_score_mean = np.mean(F1_score)
        
        return F1_score_mean



    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        FWIoU = np.nan_to_num(FWIoU, nan=0.0, posinf=0.0, neginf=0.0)
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def evaluate_single_image(self, gt_image, pred_image):
        assert gt_image.shape == pred_image.shape
        confusion_matrix = self._generate_matrix(gt_image, pred_image)
        self.confusion_matrix = confusion_matrix

        metrics = {
            "Pixel_Accuracy": self.Pixel_Accuracy(),
            "Pixel_Accuracy_Class": self.Pixel_Accuracy_Class(),
            "Mean_IoU": self.Mean_Intersection_over_Union()[0],
            "Dice_Coefficient": self.dice_coefficient(),
            "F1_Score": self.F1_score(),
            "Specificity": self.Specificity(),
            "Sensitivity": self.Sensitivity(),
            "Frequency_Weighted_IoU": self.Frequency_Weighted_Intersection_over_Union()
        }
        return metrics




# class Evaluator(object):
#     def __init__(self, num_class):
#         self.num_class = num_class
#         self.confusion_matrix = np.zeros((self.num_class,)*2)

#     def Pixel_Accuracy(self):
#         Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
#         return Acc

#     def Pixel_Accuracy_Class(self):
#         Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
#         Acc = np.nanmean(np.nan_to_num(Acc, nan=0.0, posinf=0.0, neginf=0.0))
#         return Acc

#     def Mean_Intersection_over_Union(self):
#         MIoU = np.diag(self.confusion_matrix) / (
#                     np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
#                     np.diag(self.confusion_matrix))
#         MIoU_per = np.nan_to_num(MIoU, nan=0.0, posinf=0.0, neginf=0.0)
#         MIoU = np.mean(MIoU_per)
#         return MIoU, MIoU_per
    
#     def Mean_Intersection_over_Union_per_Class(self):
#         smooth = 1e-6
#         MIoU = np.diag(self.confusion_matrix) / (
#                     np.sum(self.confusion_matrix, axis=1) +
#                     np.sum(self.confusion_matrix, axis=0) -
#                     np.diag(self.confusion_matrix) + smooth)
#         MIoU = np.nan_to_num(MIoU, nan=0.0, posinf=0.0, neginf=0.0)
#         return MIoU

#     def Specificity(self):
#         Sp = (np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)) / (
#             np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix) + np.sum(self.confusion_matrix, axis=0) -
#             np.diag(self.confusion_matrix))
#         Sp = np.nanmean(np.nan_to_num(Sp, nan=0.0, posinf=0.0, neginf=0.0))
#         return Sp

#     def Sensitivity(self):
#         Se = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1))
#         Se = np.nanmean(np.nan_to_num(Se, nan=0.0, posinf=0.0, neginf=0.0))
#         return Se

#     def Pr(self):
#         Pr = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
#         Pr = np.nan_to_num(Pr, nan=0.0, posinf=0.0, neginf=0.0)
#         return Pr

#     def F1_score(self):
#         Se = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1))
#         Pr = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
#         F1_score = 2 * Pr * Se / (Pr + Se)
#         F1_score = np.nanmean(np.nan_to_num(F1_score, nan=0.0, posinf=0.0, neginf=0.0))
#         return F1_score

#     def Frequency_Weighted_Intersection_over_Union(self):
#         freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
#         iu = np.diag(self.confusion_matrix) / (
#             np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
#             np.diag(self.confusion_matrix))
#         FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
#         FWIoU = np.nan_to_num(FWIoU, nan=0.0, posinf=0.0, neginf=0.0)
#         return FWIoU

#     def _generate_matrix(self, gt_image, pre_image):
#         mask = (gt_image >= 0) & (gt_image < self.num_class)
#         label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
#         count = np.bincount(label, minlength=self.num_class**2)
#         confusion_matrix = count.reshape(self.num_class, self.num_class)
#         return confusion_matrix

#     def add_batch(self, gt_image, pre_image):
#         assert gt_image.shape == pre_image.shape
#         self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

#     def reset(self):
#         self.confusion_matrix = np.zeros((self.num_class,) * 2)
