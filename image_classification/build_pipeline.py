"""
This module provides functions to build a pipline for tf.data.Dataset.
This module coded in reference to 
https://www.tensorflow.org/guide/data
https://www.tensorflow.org/datasets/keras_example                                            
"""
import tensorflow as tf

SHUFFLE_BUFFER_SIZE = 1000

def build_pipline(
    ds, resized_2d_shape=None, 
    shuffle=False, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    batch_size=32, augmentator=None
):  
    # Resize image data
    if resized_2d_shape:
        ds = ds.map(
            lambda x, y: resize_img(x, y, resized_2d_shape), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    # Normalize image data
    ds = ds.map(
        normalize_img, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    ds = ds.cache()

    # Shuffle data
    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)
    
    ds = ds.batch(batch_size)

    # Apply data augmentations
    if augmentator:
        ds = ds.map(
            lambda x, y: (augmentator(x, training=True), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

def resize_img(image, label, resized_2d_shape):
    return tf.image.resize(image, resized_2d_shape), label
