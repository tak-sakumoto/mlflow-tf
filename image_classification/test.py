"""
This module provides functions to test a trained model.
This module coded in reference to 
https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#mlflow.tensorflow.load_model
"""
import tensorflow as tf
import mlflow

import argparse

from load_data import load_data
from build_pipeline import build_pipline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_uri', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='mnist')

    args = parser.parse_args()
    """
    model_uri = args.model_uri
    if model_uri is None:
        exit()
    """
    model_path = args.model_path
    if model_path is None:
        exit()
    dataset_name = args.dataset_name

    _, ds_test, ds_info = load_data(dataset_name=dataset_name)

    # Load a trained model by mlflow.keras.load_model()
    # model_uri: file:/path/to/mlflow-tf/mlruns/artifacts/models/model
    #model = mlflow.keras.load_model(model_uri)

    # Load a trained model by tensorflow.keras.models.load_model()
    # model_path: mlruns/artifacts/models/model/data/model

    batch_size = 64
    ds_test = build_pipline(
        ds_test,
        batch_size=batch_size
    )

    model = tf.keras.models.load_model(model_path)
    for l in model.layers[1:]:
        l.trainable = False

    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    vals = model.evaluate(ds_test)
    for metrics_name, val in zip(model.metrics_names, vals):
        print('{}: {}'.format(metrics_name, val))
