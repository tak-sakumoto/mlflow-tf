"""
This module provides functions to create a model to be trained.
This module coded in reference to 
https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
https://www.tensorflow.org/tutorials/images/transfer_learning#create_the_base_model_from_the_pre-trained_convnets
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import applications
import tensorflow_hub as hub

def create_model(input_shape, num_classes, hub_feature_vector=None):
    if hub_feature_vector:
        pretrained_layers = hub.KerasLayer(
            hub_feature_vector,
            trainable=True
        )

        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(pretrained_layers)
        model.add(Dense(num_classes, activation='softmax'))

    else:
        model = applications.efficientnet.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None,
            classes=num_classes
        )

    return model