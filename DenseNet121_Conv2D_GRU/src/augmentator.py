import numpy as np
import imgaug.augmenters as iaa

class Augmentator:
    
    def __init__(
        self,
        rotation_range=(-5, 5),
        width_shift_range=(-0.1, 0.1),
        height_shift_range=(-0.1, 0.1),
        scale_x_range=(0.77, 1.3),
        scale_y_range=(0.77, 1.3),
        horizontal_flip=True,
        preprocessing_function=None,
        seed=None
    ):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.horizontal_flip = horizontal_flip
        self.preprocessing_function = preprocessing_function
        self.seed = seed
        
    def augment(self, frame_sequence):
        if self.seed is not None:
            rs = np.random.RandomState(self.seed)
        else:
            rs = np.random
        rotate = rs.randint(self.rotation_range[0], self.rotation_range[1])
        translate_percent = {
            'x': rs.uniform(self.width_shift_range[0], self.width_shift_range[1]),
            'y': rs.uniform(self.height_shift_range[0], self.height_shift_range[1])
        }
        scale = {
            'x': rs.uniform(self.scale_x_range[0], self.scale_x_range[1]),
            'y': rs.uniform(self.scale_x_range[0], self.scale_x_range[1])
        }
        affine = iaa.Affine(
            rotate=rotate,
            translate_percent=translate_percent,
            scale=scale,
        )
        horizontal_flip = iaa.Fliplr(1)
        aug_sort_flg = rs.randint(0, 2)
        if aug_sort_flg:
            aug = [affine, horizontal_flip]
        else:
            aug = [horizontal_flip, affine]
        seq = iaa.Sequential(aug)
        aug_sequence = seq(images=frame_sequence)
        if self.preprocessing_function is not None:
            aug_sequence = np.array(list(map(self.preprocessing_function, aug_sequence)))
        return aug_sequence