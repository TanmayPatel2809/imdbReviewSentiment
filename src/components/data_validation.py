import os
from src import logger
from datasets import load_from_disk
from src.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self)-> bool:
        try:
            validation_status = None
            dataset= load_from_disk(self.config.data_dir)
            df= dataset.to_pandas()
            all_cols = list(df.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation Status : {validation_status}")
                else:
                    validation_status=True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation Status : {validation_status}")

            return validation_status

        except Exception as e:
            raise e
