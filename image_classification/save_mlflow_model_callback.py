"""
This module provides a callback to save the best models in a run.
"""
from tensorflow.keras import callbacks
import mlflow
from pathlib import Path

class SaveMLflowModelCallback(callbacks.Callback):
    def __init__(self, models_dir_path):
        self.models_dir_path = models_dir_path
        #self.min_val_loss = float('inf')
        self.max_val_acc = float('-inf')
        
    def on_epoch_end(self, epoch, logs=None):
        #val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')

        if val_accuracy >= self.max_val_acc:
            epoch_str = "epoch-{:04d}".format(epoch)
            model_path = Path(self.models_dir_path) / epoch_str

            mlflow.keras.save_model(self.model, model_path)

            self.max_val_acc = val_accuracy
            self.epoch_max_val_acc = epoch
        
            # if log the models using a MLflowLogCallback class,
            # the following log_artifacts() is unnecessary.
            """
            mlflow.log_artifacts(
                local_dir=model_path,
                artifact_path="models"+"/"+epoch_str
            )
            """

