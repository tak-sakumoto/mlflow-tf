"""
This module provides functions to sample a dataset with data augmentation.
This module coded in reference to 
https://www.tensorflow.org/tensorboard/image_summaries
"""
import tensorflow as tf

import matplotlib.pyplot as plt
import re
import datetime
import tempfile
import argparse
from pathlib import Path

from load_data import load_data
from image_augmentator import create_image_augmentator
from build_pipeline import build_pipline
from count_class_data import count_class_data

TMP_FILE_PATH = "./tmp"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="rock_paper_scissors", type=str)
parser.add_argument("--num", default=32, type=int)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--resized_2d_shape", default=None, type=list)
parser.add_argument("--width_shift_range", default=0.2, type=float)
parser.add_argument("--height_shift_range", default=0.2, type=float)
parser.add_argument("--rotation_range", default=0.2, type=float)
parser.add_argument("--zoom_range", default=0.2, type=float)
parser.add_argument("--fill_mode", default="reflect", type=str)
parser.add_argument("--interpolation", default="bilinear", type=str)
parser.add_argument("--horizontal_flip", action="store_true")
parser.add_argument("--vertical_flip", action="store_true")
parser.add_argument("--seed", default=None, type=int)

def get_image(idx, image, label, class_names):
    # Create a figure
    figure = plt.figure(figsize=(5,5))
    plt.title("{}: {}".format(idx, class_names[label]))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image)

    return figure

if __name__ == '__main__':
    args = parser.parse_args()

    # Get hyperparameters from args
    dataset_name = args.dataset_name
    num = args.num
    resized_2d_shape = args.resized_2d_shape
    shuffle = args.shuffle
    
    # Get an augumentator for the dataset
    image_augmentator_params = {
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'rotation_range': args.rotation_range,
        'zoom_range': args.zoom_range,
        'fill_mode': args.fill_mode,
        'interpolation': args.interpolation,
        'horizontal_flip': args.horizontal_flip,
        'vertical_flip': args.vertical_flip,
        'seed': args.seed
    }
    img_augmentor = create_image_augmentator(**image_augmentator_params)

    # Load a dataset
    ds_train, ds_test, ds_info = load_data(
        dataset_name=dataset_name
    )
    print(ds_info)
    class_names = ds_info.features["label"].names
    print(class_names)
    input_shape = ds_info.features['image'].shape
    num_classes = ds_info.features['label'].num_classes
    num_train = ds_info.splits['train'].num_examples
    
    class_counts = count_class_data(ds_train, num_classes) 
    
    # Make the dataset pipline
    ds_train = build_pipline(
        ds_train,
        resized_2d_shape = args.resized_2d_shape,
        shuffle=True,
        shuffle_buffer_size=ds_info.splits['train'].num_examples,
        batch_size=1,
        augmentator=img_augmentor
    )

    ds_test = build_pipline(
        ds_test,
        batch_size=num
    )

    # Prepare a temporary directory for saving models
    datestr = re.sub(r'[\ \:\.\-]', '', str(datetime.datetime.now()))[:-5]
    tmp_image_path = tempfile.mkdtemp(
        prefix='image_' + datestr + '_',
        dir=TMP_FILE_PATH
    )
 
    for i, (image, label) in enumerate(ds_train.take(num)):
        lbl = label[0].numpy()
        img = image[0,:,:,:]
        figure = get_image(i, img, lbl, class_names)
        figure.savefig(Path(tmp_image_path) / "{:08d}_{:04d}.png".format(i, lbl))
