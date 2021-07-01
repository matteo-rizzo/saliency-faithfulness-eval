import random
from typing import Tuple

import cv2
import numpy as np

from augmenters.DataAugmenter import DataAugmenter
from auxiliary.utils import rgb_to_bgr, bgr_to_rgb


class TemporalDataAugmenter(DataAugmenter):

    def __init__(self, train_size: Tuple = (512, 512), angle: int = 15, scale: Tuple = (0.8, 1.0), color: float = 0.0):
        super().__init__(train_size, angle, scale, color)

    def __resize_image(self, img: np.ndarray, size: Tuple = None) -> np.ndarray:
        if size is None:
            size = self._train_size
        return cv2.resize(img, size)

    def resize_sequence(self, img: np.ndarray, size: Tuple = None) -> np.ndarray:
        return np.stack([self.__resize_image(img[i], size) for i in range(img.shape[0])])

    def get_random_color_bias(self):
        color_bias = np.zeros(shape=(3, 3))
        for i in range(3):
            color_bias[i, i] = 1 + random.random() * self._color - 0.5 * self._color
        return color_bias

    def __augment_image(self, img: np.ndarray, color_bias: np.ndarray = None) -> np.ndarray:
        img = self._random_flip(self.__resize_image(self._rotate_and_crop(self._rescale(img))))
        return np.clip(self.__apply_color_bias(img, color_bias), 0, 255)

    def __augment_illuminant(self, illuminant: np.ndarray, color_bias: np.ndarray = None) -> np.ndarray:
        if color_bias is None:
            color_bias = self.get_random_color_bias()
        illuminant = rgb_to_bgr(illuminant)
        new_illuminant = np.array([[illuminant[j] * color_bias[i, j] for j in range(3)] for i in range(3)])
        return rgb_to_bgr(np.clip(new_illuminant, 0.01, 100))

    def __apply_color_bias(self, img: np.ndarray, color_bias: np.ndarray) -> np.ndarray:
        if color_bias is None:
            color_bias = self.get_random_color_bias()
        return img * np.array([[[color_bias[0][0], color_bias[1][1], color_bias[2][2]]]], dtype=np.float32)

    def augment_sequence(self, seq: np.ndarray, illuminant: np.ndarray) -> Tuple:
        color_bias = self.get_random_color_bias()

        augmented_frames, augmented_illuminants = [], []
        for i in range(seq.shape[0]):
            augmented_frames.append(self.__augment_image(seq[i], color_bias))
            augmented_illuminants.append(self.__augment_illuminant(illuminant, color_bias))

        color_bias = np.array([[[color_bias[0][0], color_bias[1][1], color_bias[2][2]]]], dtype=np.float32)

        return np.stack(augmented_frames), color_bias

    def augment_mimic(self, seq: np.ndarray) -> np.ndarray:
        num_steps = seq.shape[0]
        shot_frame = seq[-1]

        augmented_frames, img_temp = [], bgr_to_rgb(shot_frame) * (1.0 / 255)
        for _ in range(num_steps):
            scale = min(max(int(round(min(img_temp.shape[:2]) * 0.95)), 10), min(img_temp.shape[:2]))
            img_temp = self.__resize_image(self._rotate_and_crop(self._rescale(img_temp, scale)))
            img_temp = np.clip(img_temp.astype(np.float32), 0, 255)
            augmented_frames.append(img_temp)

        return np.stack(augmented_frames)
