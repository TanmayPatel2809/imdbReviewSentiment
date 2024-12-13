from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_from_disk
import os
from src.entity.config_entity import ModelTrainerConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_ckpt, num_labels=2).to(device)
        
        dataset = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            evaluation_strategy=self.config.evaluation_strategy,
            metric_for_best_model=self.config.metric_for_best_model,
            learning_rate=float(self.config.learning_rate),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.num_train_epochs,
            logging_dir=os.path.join(self.config.root_dir, 'logs'),
        )

        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,  # Add compute_metrics function
        )

        torch.cuda.empty_cache()
        
        trainer.train()

        model.save_pretrained(os.path.join(self.config.root_dir, "distilBERT-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
