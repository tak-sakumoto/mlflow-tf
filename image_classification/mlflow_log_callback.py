"""
This module provides a callback to save the TensorBoard logs in a run.
"""
from tensorflow.keras import callbacks
import mlflow
from pathlib import Path

class MLflowLogCallback(callbacks.Callback):
    def __init__(
        self,
        tmp_models_path,
        tmp_tensorboard_logs_path,
    ):
        self.tmp_models_path = Path(tmp_models_path)
        self.tmp_tensorboard_logs_path = Path(tmp_tensorboard_logs_path)
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_str = "epoch-{:04d}".format(epoch)
        epoch_model_dir = self.tmp_models_path / epoch_str
        if epoch_model_dir.exists():
            mlflow.log_artifacts(
                local_dir=self.tmp_models_path / epoch_str,
                artifact_path="models"+"/"+epoch_str
            )
        mlflow.log_artifacts(self.tmp_tensorboard_logs_path)
