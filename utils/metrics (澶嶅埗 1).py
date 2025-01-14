import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU_per = MIoU
        MIoU = np.mean(MIoU)
        return MIoU , MIoU_per
    
    def Mean_Intersection_over_Union_per_Class(self):
        smooth = 1e-6
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) +
                    np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix) + smooth)
        return MIoU


    def Specificity(self):
        Sp=(np.diag(self.confusion_matrix).sum()-np.diag(self.confusion_matrix))/ (
        np.diag(self.confusion_matrix).sum()-np.diag(self.confusion_matrix)+np.sum(self.confusion_matrix,axis=0)-
        np.diag(self.confusion_matrix))
        Sp = np.mean(Sp)
        return Sp

    def Sensitivity(self):
        Se =np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix,axis=1))
        Se = np.mean(Se)
        return Se

    def Pr(self):
        Pr =np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix,axis=0)
        return Pr

    def F1_score(self):
        Se = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1))
        Pr = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        F1_score = 2*Pr *Se / (Pr+Se)
        F1_score = np.mean(F1_score)
        return F1_score

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
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




