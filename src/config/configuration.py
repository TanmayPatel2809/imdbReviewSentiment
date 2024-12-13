from src.constants import *
from src.utilis.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionconfig, DataValidationConfig, DataTransformationconfig, ModelTrainerConfig)

class ConfigurationManager:
    def __init__(self,
                 config_filepath= CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self)-> DataIngestionconfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config =DataIngestionconfig(

            root_dir=config.root_dir,
            hf_dataset_name= config.hf_dataset_name,
            local_data_file=config.local_data_file,
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_dir= config.data_dir,
            all_schema=schema
        )
        return data_validation_config
    
    def get_data_transformation_config(self)-> DataTransformationconfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationconfig(
            root_dir= config.root_dir,
            data_path=config.data_path,
            filtered_data_path = config.filtered_data_path,
            tokenizer_name=config.tokenizer_name,
        )
        return data_transformation_config
    
    def get_model_trainer_config(self)-> ModelTrainerConfig:
        config=self.config.model_trainer
        model_params=self.params

        create_directories([config.root_dir])

        model_trainer_config=ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt = config.model_ckpt,
            evaluation_strategy = model_params.evaluation_strategy,
            metric_for_best_model = model_params.metric_for_best_model,
            learning_rate = model_params.learning_rate,
            per_device_train_batch_size = model_params.per_device_train_batch_size,
            per_device_eval_batch_size = model_params.per_device_eval_batch_size,
            gradient_accumulation_steps = model_params.gradient_accumulation_steps,
            weight_decay = model_params.weight_decay,
            num_train_epochs = model_params.num_train_epochs,
            save_strategy= model_params.save_strategy
        )

        
        return model_trainer_config