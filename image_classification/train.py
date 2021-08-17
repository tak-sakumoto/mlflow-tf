"""
This module provides functions to train a neural network model.
This module coded in reference to 
https://www.tensorflow.org/datasets/keras_example
"""
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics, callbacks
import mlflow
import re
import os
from pathlib import Path
import argparse
import tempfile
import shutil
import datetime

from load_data import load_data
from image_augmentator import create_image_augmentator
from build_pipeline import build_pipline
from create_model import create_model
from count_class_data import count_class_data
from calc_class_weight import calc_class_weight
from save_mlflow_model_callback import SaveMLflowModelCallback
from mlflow_log_callback import MLflowLogCallback

LOCAL_MLRUNS_PATH = "../mlruns"
EXPERIMENT_NAME = "Image Classification"
TMP_FILE_PATH = "./tmp"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="cifar10", type=str)
parser.add_argument("--epochs", default=12, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--resized_2d_shape", default=None, type=list)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument(
    "--hub_feature_vector",
    default="https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
    type=str
)
#https://tfhub.dev/google/imagenet/inception_v2/feature_vector/5
#https://tfhub.dev/google/imagenet/resnet_v1_101/classification/5
parser.add_argument("--width_shift_range", default=0.2, type=float)
parser.add_argument("--height_shift_range", default=0.2, type=float)
parser.add_argument("--rotation_range", default=0.2, type=float)
parser.add_argument("--zoom_range", default=0.2, type=float)
parser.add_argument("--fill_mode", default="reflect", type=str)
parser.add_argument("--interpolation", default="nearest", type=str)
parser.add_argument("--horizontal_flip", action="store_true")
parser.add_argument("--vertical_flip", action="store_true")
parser.add_argument("--seed", default=None, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Separate characters in a path are unified as '/'
    local_mlruns_path = \
        str(Path(LOCAL_MLRUNS_PATH).resolve()).replace(os.sep, '/')
    mlflow.set_tracking_uri("file:" + local_mlruns_path)
    mlflow.set_experiment(EXPERIMENT_NAME)
        
    # Get hyperparameters from args
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dataset_name = args.dataset_name
    hub_feature_vector = args.hub_feature_vector
    resized_2d_shape = args.resized_2d_shape

    # Get an augumentator for the training dataset
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

    # Enable auto-logging
    mlflow.tensorflow.autolog(log_models=False)

    datestr = re.sub(r'[\ \:\.\-]', '', str(datetime.datetime.now()))[:-5]

    with mlflow.start_run() as run:
        # Get a temporary artifact URI
        tmp_artifact_path = tempfile.mkdtemp(
            prefix='artifact_' + datestr + '_',
            dir=TMP_FILE_PATH
        )

        # Prepare a temporary directory for saving models
        tmp_models_path = Path(tmp_artifact_path) / "models"
        os.makedirs(tmp_models_path, exist_ok=True)
        mlflow.log_artifacts(tmp_models_path)

        # Prepare a temporary directory for saving TensorBoard logs
        tmp_tensorboard_logs_path = \
            Path(tmp_artifact_path) / "tensorboard_logs"
        os.makedirs(tmp_tensorboard_logs_path, exist_ok=True)
        mlflow.log_artifacts(tmp_tensorboard_logs_path)
        
        # Load a dataset
        ds_train, ds_test, ds_info = load_data(
            dataset_name=dataset_name
        )
        mlflow.log_param("dataset_name", dataset_name)

        # Get the dataset infomation
        print(ds_info)
        input_shape = ds_info.features['image'].shape
        num_classes = ds_info.features['label'].num_classes
        num_train = ds_info.splits['train'].num_examples

        # Count data number for each class
        class_counts = count_class_data(ds_train, num_classes)
        # Calculate class weights for training
        class_weight = calc_class_weight(num_train, num_classes, class_counts)

        # Get image augmentator
        img_augmentor =\
            create_image_augmentator(**image_augmentator_params)

        # Make the dataset pipline
        ds_train = build_pipline(
            ds_train,
            resized_2d_shape=resized_2d_shape,
            shuffle=True,
            batch_size=batch_size,
            augmentator=img_augmentor
        )
        ds_test = build_pipline(
            ds_test,
            resized_2d_shape=resized_2d_shape,
            batch_size=batch_size
        )
        mlflow.log_param("resized_2d_shape", resized_2d_shape)

        # Get a callback for saving models
        save_model_callback = SaveMLflowModelCallback(
            tmp_models_path
        )
        
        # Get a callback for saving TensorBoard logs
        mlflow_log_callback = MLflowLogCallback(
            tmp_models_path=tmp_models_path,
            tmp_tensorboard_logs_path=tmp_tensorboard_logs_path 
        )

        # Get a callback for reducing the learning rate in training
        reduce_lr_callback = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.001
        )

        # Get a callback list for fit() 
        callbacks = []
        callbacks.append(save_model_callback)
        callbacks.append(reduce_lr_callback)
        callbacks.append(mlflow_log_callback)

        if len(callbacks) <= 0:
            callbacks = None

        # Create a model
        model = create_model(input_shape, num_classes, hub_feature_vector)
        mlflow.log_param("hub_feature_vector", hub_feature_vector)
        print(model.summary())

        # Configure the model with an optimizer, losses, and metrics
        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=learning_rate),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[
                metrics.SparseCategoricalAccuracy(name="accuracy")
            ]
        )

        model.fit(
            ds_train,
            epochs=epochs, 
            validation_data=ds_test,
            class_weight=class_weight,
            callbacks=callbacks
        )

    shutil.rmtree(tmp_artifact_path)

