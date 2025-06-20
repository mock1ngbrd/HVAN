import random
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

np.seterr(divide='ignore', invalid='ignore')


class Normalization(object):
    def __init__(self, volume_key='volume2'):
        self.volume_key = volume_key

    def __call__(self, sample):
        image_array = sample[self.volume_key]
        arr = image_array.reshape(-1)
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        image_array = (image_array - arr_min) / (arr_max - arr_min + 1e-6)
        sample[self.volume_key] = image_array
        return sample


class NormalizationFrame(object):
    def __init__(self, volume_key='volume2'):
        self.volume_key = volume_key

    def __call__(self, sample):
        image_array = sample[self.volume_key]
        sample[self.volume_key] = image_array / 255.0
        return sample


class SynthesizeTransView(object):
    def __init__(self, volume_key='volume2'):
        self.volume_key = volume_key

    def __call__(self, sample):
        image_array = sample[self.volume_key]
        image_array = np.transpose(image_array, (2, 1, 0))
        if self.volume_key == 'volume2':
            image_array = np.flip(image_array, axis=0)  # sagittal
        else:
            image_array = np.flip(image_array, axis=2)  # transverse
        sample[self.volume_key] = image_array
        return sample


class RandomRotateTransform(object):
    def __init__(self, angle_range=(-10, 10), p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.angle_range = angle_range

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']

        if np.random.uniform() < self.p_per_sample:
            rand_angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            volume1 = rotate(volume1, angle=rand_angle, axes=(0, 1), reshape=False, order=1)  # axes=(-2, -3)
        if np.random.uniform() < self.p_per_sample:
            rand_angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            volume2 = rotate(volume2, angle=rand_angle, axes=(0, 1), reshape=False, order=1)

        sample['volume1'] = volume1
        sample['volume2'] = volume2

        return sample


class RandomTranslation(object):
    def __init__(self, max_shift=(5, 5, 5), p=0.5):
        """
        Randomly translate a 3D image along each axis.

        Args:
            max_shift (tuple): Maximum translation (D, H, W) in each direction.
            p (float): Probability of applying the augmentation.
        """
        self.max_shift = max_shift
        self.p = p

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']
        if np.random.uniform() < self.p:
            volume1 = self.translate(volume1)
        if np.random.uniform() < self.p:
            volume2 = self.translate(volume2)

        sample['volume1'] = volume1
        sample['volume2'] = volume2

        return sample

    def translate(self, image):
        """
        Apply a random translation to a 3D image.

        Parameters:
        - image (numpy.ndarray): 3D array representing the image.
        - max_shift (tuple): Maximum shift in (z, y, x) directions.

        Returns:
        - Translated image (numpy.ndarray) with the same shape as input.
        """
        assert image.ndim == 3, "Input image must be a 3D array"

        shifts = [np.random.randint(-self.max_shift[i], self.max_shift[i] + 1) for i in range(3)]
        translated_image = np.zeros_like(image)

        z_shift, y_shift, x_shift = shifts

        z_range = slice(max(0, z_shift), min(image.shape[0], image.shape[0] + z_shift))
        y_range = slice(max(0, y_shift), min(image.shape[1], image.shape[1] + y_shift))
        x_range = slice(max(0, x_shift), min(image.shape[2], image.shape[2] + x_shift))

        z_src = slice(max(0, -z_shift), min(image.shape[0], image.shape[0] - z_shift))
        y_src = slice(max(0, -y_shift), min(image.shape[1], image.shape[1] - y_shift))
        x_src = slice(max(0, -x_shift), min(image.shape[2], image.shape[2] - x_shift))

        translated_image[z_range, y_range, x_range] = image[z_src, y_src, x_src]

        return translated_image


class RandomRotateTransform2(object):
    def __init__(self, angle_range=(-10, 10), p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.angle_range = angle_range

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']

        if np.random.uniform() < self.p_per_sample:
            rand_angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            volume1 = rotate(volume1, angle=rand_angle, axes=(1, 2), reshape=False, order=1)  # axes=(-2, -3)
            volume2 = rotate(volume2, angle=rand_angle, axes=(1, 2), reshape=False, order=1)

        sample['volume1'] = volume1
        sample['volume2'] = volume2

        return sample


class ScaleTransform(object):
    def __init__(self, zoom_range=(0.8, 1.3), p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.zoom_range = zoom_range

    def __call__(self, sample):
        volume, label = sample['volume'], sample['label']
        if np.random.uniform() < self.p_per_sample:
            zoom_factor = np.random.randint(self.zoom_range[0] * 10, self.zoom_range[1] * 10) / 10
            volume = zoom(volume, zoom_factor, order=1)
            label = zoom(label, zoom_factor, order=0)
        sample['volume'], sample['label'] = volume, label

        return sample


class MirrorTransform(object):
    def __init__(self, box_prefix=False, axes=(0, 1, 2)):
        self.axes = axes
        self.box_prefix = box_prefix

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']
        if self.box_prefix:
            box = sample['box']
        if isinstance(self.axes, int):
            if np.random.uniform() < 0.5:
                volume1 = np.flip(volume1, self.axes)
                volume2 = np.flip(volume2, self.axes)
                if self.box_prefix:
                    box = np.flip(box, self.axes)
        else:
            for axis in self.axes:
                if np.random.uniform() < 0.5:
                    volume1 = np.flip(volume1, axis=axis)
                    volume2 = np.flip(volume2, axis=axis)

                    if self.box_prefix:
                        box = np.flip(box, axis=axis)

        sample['volume1'] = volume1
        sample['volume2'] = volume2

        if self.box_prefix:
            sample['box'] = box
        return sample


class MirrorTransform2(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']

        if np.random.uniform() < self.prob:
            volume1 = np.flip(volume1, axis=2)  # x
            volume2 = np.flip(volume2, axis=2)

        sample['volume1'] = volume1
        sample['volume2'] = volume2

        return sample


class GaussianBlur(object):
    def __init__(self, sigma=3.0):
        self.sigma = sigma

    def __call__(self, sample):
        array = sample['mask']
        sample['mask'] = gaussian_filter(array, sigma=self.sigma)
        return sample
