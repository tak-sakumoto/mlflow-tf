"""
This module provides functions to load datasets.
This module coded in reference to 
https://www.tensorflow.org/datasets/keras_example
https://www.tensorflow.org/datasets/catalog/overview
"""
import tensorflow_datasets as tfds

DATASET_NAM= ['mnist', 'cifar10', 'cifar100']
def load_data(dataset_name='mnist'):
    # Load a dataset
    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    return ds_train, ds_test, ds_info
