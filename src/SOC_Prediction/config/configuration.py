from SOC_Prediction.constants import *
import os
from SOC_Prediction.utils.common import read_yaml,create_directories
from SOC_Prediction.entity.config_entity import (DataIngestionConfig,PrepareModelConfig,TrainModelConfig)

class ConfigurationManager:
    def __init__(self,config_filepath = CONFIG_FILE_PATH,params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            data_dir=config.data_dir,
            ncycles_param = self.params.data_ingestion.n_cycles_per_temperature   
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        data_ingestion_config = PrepareModelConfig(
            model_path = config.model_path,
            batch_size = self.params.prepare_base_model.BATCH_SIZE,
            learning_rate = self.params.prepare_base_model.LR,
            epochs = self.params.prepare_base_model.EPOCHS 
        )

        return data_ingestion_config
    
    def train_model_config(self) -> TrainModelConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.data_dir, training.train_chemistry)
        test_data_1 = os.path.join(self.config.data_ingestion.data_dir, training.test_chemistry_1)
        test_data_2 = os.path.join(self.config.data_ingestion.data_dir, training.test_chemistry_2)
        
        create_directories([Path(training.root_dir)])

        train_model_config = TrainModelConfig(
            model_path = prepare_base_model.model_path,
            root_dir = training.root_dir,
            training_data = training_data,
            test_data_1 = test_data_1,
            test_data_2 = test_data_2,
            trained_model_path = training.trained_model_path,
            batch_size = params.prepare_base_model.BATCH_SIZE,
            learning_rate = params.prepare_base_model.LR,
            epochs = params.prepare_base_model.EPOCHS,
            train_val_split = training.train_val_split,
            sampling_rate = training.sampling_rate,
            avg_window = training.avg_window,
            LSTM_window_length = training.LSTM_window_length,
            should_avg_be_the_feature = training.should_avg_be_the_feature
            
        )

        return train_model_config
    

