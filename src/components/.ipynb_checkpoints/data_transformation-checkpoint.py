import os
from src import logger
from datasets import load_from_disk, DatasetDict
from transformers import DistilBertTokenizer
from src.entity.config_entity import DataTransformationconfig

class DataTransformation:
    def __init__(self, config: DataTransformationconfig):
        self.config = config
        self.tokenizer= DistilBertTokenizer.from_pretrained(self.config.tokenizer_name)
    

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['text'], 
                                        max_length=512, 
                                        truncation=True, 
                                        padding='max_length')
        labels = example_batch['label']
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels
    }
    
    def convert(self):
        dataset = load_from_disk(self.config.data_path)
        train_dataset = dataset.train_test_split(test_size=0.1, shuffle=True)
        valid_test_dataset = train_dataset['test'].train_test_split(test_size=0.5, shuffle=True)
        train_dataset = train_dataset['train']
        val_dataset = valid_test_dataset['train']
        test_dataset = valid_test_dataset['test']
        final_dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        dataset_pt = final_dataset_dict.map(self.convert_examples_to_features, batched = False)
        dataset_pt.save_to_disk(os.path.join(self.config.filtered_data_path))