import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree=15):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        
        if random.random() < 0.5:
            # 随机缩放和裁剪
            short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            w, h = img.size
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            
            if short_size < self.crop_size:
                padh = self.crop_size - oh if oh < self.crop_size else 0
                padw = self.crop_size - ow if ow < self.crop_size else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
            
            w, h = img.size
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        else:
            # 简单调整尺寸base_size
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)

        return {'image': img, 'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class RGB_HSV(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            transform_degree = random.uniform(1, 1+self.degree)

            def HSV2BGR(_img, hsv):
                img = _img.copy() / 255.
                # get max and min
                max_v = np.max(img, axis=2).copy()
                min_v = np.min(img, axis=2).copy()
                out = np.zeros_like(img)
                H = hsv[..., 0]
                S = hsv[..., 1]
                V = hsv[..., 2]
                C = S
                H_ = H / 60.
                X = C * (1 - np.abs(H_ % 2 - 1))
                Z = np.zeros_like(H)
                vals = [[Z, X, C], [Z, C, X], [X, C, Z], [C, X, Z], [C, Z, X], [X, Z, C]]
                for i in range(6):
                    ind = np.where((i <= H_) & (H_ < (i + 1)))
                    out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
                    out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
                    out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]
                out[np.where(max_v == min_v)] = 0
                out = np.clip(out, 0, 1)
                out = (out * 255).astype(np.uint8)
                return out

            def BGR2HSV(_img):
                img = _img.copy() / 255.
                hsv = np.zeros_like(img, dtype=np.float32)
                # get max and min
                max_v = np.max(img, axis=2).copy()
                min_v = np.min(img, axis=2).copy()
                min_arg = np.argmin(img, axis=2)
                # H
                hsv[..., 0][np.where(max_v == min_v)] = 0
                ## if min == B
                ind = np.where(min_arg == 0)
    #            hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
                ## if min == R
                ind = np.where(min_arg == 2)
                hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
                ## if min == G
                ind = np.where(min_arg == 1)
                #hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
                # S
                hsv[..., 1] = max_v.copy() - min_v.copy()
                # V
                hsv[..., 2] = max_v.copy()
                return hsv

            img=np.asarray(img)
            hsv = BGR2HSV(img)
            # hsv[..., 0] = (hsv[..., 0] + 270) % 360
            hsv[..., 2] = (hsv[..., 2]) * transform_degree
            hsv[..., 1] = (hsv[..., 1]) * transform_degree
            out = HSV2BGR(img, hsv)
            img=Image.fromarray(np.uint8(out))


        return {'image': img,
                'label': mask}