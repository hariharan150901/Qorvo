import os
import pybamm
import numpy as np
import tensorflow as tf
from src.SOC_Prediction import logger
from pathlib import Path
from SOC_Prediction.entity.config_entity import (PrepareModelConfig)

class PrepareBaseModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config

    
    def get_model(self):
        model = tf.keras.Sequential(
            [tf.keras.layers.LSTM(10),
            tf.keras.layers.Dense(10,activation = 'relu'),
            tf.keras.layers.Dense(1,activation = 'linear')
            ]
        )
 
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        model.summary()
        self.save_model(path=self.config.model_path, model=self.model)
        return model


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)