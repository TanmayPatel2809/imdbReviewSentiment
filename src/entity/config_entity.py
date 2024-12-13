from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionconfig:
    root_dir: Path
    hf_dataset_name: str
    local_data_file: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_dir: Path
    all_schema: dict

@dataclass
class DataTransformationconfig:
    root_dir: Path
    data_path: Path
    filtered_data_path: Path
    tokenizer_name: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    evaluation_strategy: str
    metric_for_best_model: str
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    num_train_epochs: int
    save_strategy: str