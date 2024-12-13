from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    def initiate_model_trainer(self):
        config = ConfigurationManager()
        bart_model_trainer_config= config.get_model_trainer_config()
        bart_model_trainer_config = ModelTrainer(config=bart_model_trainer_config)
        bart_model_trainer_config.train()
        