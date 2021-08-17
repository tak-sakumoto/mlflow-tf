"""
https://www.tensorflow.org/tutorials/images/data_augmentation
https://www.tensorflow.org/guide/keras/preprocessing_layers
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing \
    import RandomFlip, RandomRotation, RandomZoom, RandomTranslation

FILL_MODE = ['constant', 'reflect', 'wrap', 'nearest']
INTERPOLATION = ['nearest', 'bilinear']

def create_image_augmentator(
        width_shift_range=0.0,
        height_shift_range=0.0,
        rotation_range=0.0,
        zoom_range=0.0,
        fill_mode='nearest',
        interpolation='nearest',
        horizontal_flip=False,
        vertical_flip=False,
        seed=None
    ):
    data_augmentation = Sequential()

    # Random flip
    flip_mode = None
    if horizontal_flip and vertical_flip:
        flip_mode = "horizontal_and_vertical"
    elif horizontal_flip:
        flip_mode = "horizontal"
    elif vertical_flip:
        flip_mode = "vertical"
    if flip_mode is not None:
        rendom_flip = RandomFlip(
            flip_mode,
            seed=seed
        )
        data_augmentation.add(rendom_flip)
    
    # Random rotation    
    random_rotation = RandomRotation(
        rotation_range,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed
    )
    data_augmentation.add(random_rotation)

    # Random zoom
    random_zoom = RandomZoom(
        height_factor=zoom_range,
        width_factor=zoom_range,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed
    )
    data_augmentation.add(random_zoom)
    
    # Random translation
    rendom_translation = RandomTranslation(
        height_factor=height_shift_range,
        width_factor=width_shift_range,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
    )
    data_augmentation.add(rendom_translation)

    return data_augmentation
