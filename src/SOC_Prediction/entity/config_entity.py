from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
  root_dir: Path
  data_dir: Path
  ncycles_param: int


@dataclass(frozen=True)
class PrepareModelConfig:
  model_path: Path
  batch_size: int
  learning_rate: float
  epochs: int
  
@dataclass(frozen=True)
class TrainModelConfig:
  model_path: Path
  root_dir: Path
  training_data: Path
  test_data_1: Path
  test_data_2: Path
  trained_model_path: Path
  batch_size: int
  learning_rate: float
  epochs: int
  train_val_split: float
  sampling_rate: int
  avg_window: int
  LSTM_window_length: int
  should_avg_be_the_feature: bool
    
    
