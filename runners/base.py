from abc import abstractmethod
from typing import Dict

from datasets import DatasetDict
from registrable import Registrable
from data_process.base import BaseDataLoader
import os
import shutil
import logging
from dataclasses import asdict
from transformers import Seq2SeqTrainingArguments
from utils.metrics import compute_metrics_text
from data_process import MultiTeacherDataCollator
from trainers.multiteacher import BaseTrainer
from datetime import datetime
from easyllm_kit.utils import get_logger, save_json
from easyllm_kit.models import LLM

logger = get_logger('kd_runner', 'kd_runner.log')


class Runner(Registrable):
    @staticmethod
    def build_from_config(config):
        runner_cls = Runner.by_name(config["runner_name"].lower())
        return runner_cls(config)

    @abstractmethod
    def run(self):
        pass


# knowledge distillation
@Runner.register('kd_runner')
class KnowledgeDistillationRunner(Runner):

    def __init__(self, config):
        self.model_config = config["model_config"]
        self.generation_config = config["generation_config"]
        self.training_config = config["training_config"]
        self.data_config = config["data_config"]
        self.config = config

        # Setup directories
        self.setup_directories()

    @staticmethod
    def get_config_dir(args):
        """
        Constructs a directory path based on training arguments for model configurations.

        Args:
            args: Command-line arguments or any arguments object with necessary attributes.

        Returns:
            A string representing the path to the configuration directory.
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config_dir = f'{timestamp}_{args.trainer_name}'
        return config_dir

    def setup_directories(self):
        """Setup output and logging directories."""
        config_dir = self.get_config_dir(self.training_config)
        run = self.training_config.run_id

        # Setup output directory
        self.training_config.output_dir = f'ckpts/{config_dir}/{run}'
        if os.path.exists(self.training_config.output_dir):
            logging.info('Found existing checkpoint directory. Deleting for fresh start.')
            shutil.rmtree(self.training_config.output_dir)
        else:
            os.makedirs(self.training_config.output_dir)

        # Setup logging directory
        self.training_config.logging_dir = f'logs/{config_dir}/{run}'

        # save the config to the directory
        save_json(self.config, f'{self.training_config.output_dir}/config.json')

    def setup_training_args(self) -> Seq2SeqTrainingArguments:
        """Setup training arguments for the trainer."""
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.training_config.output_dir,
            remove_unused_columns=False,
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.eval_steps,
            logging_dir=self.training_config.logging_dir,
            logging_strategy=self.training_config.logging_strategy,
            logging_steps=self.training_config.eval_steps,
            max_steps=self.training_config.max_steps,
            learning_rate=self.training_config.learning_rate,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            predict_with_generate=True,
            seed=self.training_config.run_id,
            local_rank=self.training_config.local_rank,
            bf16=self.training_config.bf16,
            generation_max_length=self.training_config.generation_max_length,
            prediction_loss_only=False,
            # Add any additional training arguments from config
            **asdict(self.training_config)
        )
        return training_args

    def setup_student_model(self):
        self.student_model, self.student_tokenizer = LLM.build_from_config(self.model_config)

    def setup_data(self) -> DatasetDict:
        """Setup and load the dataset."""
        data_loader = BaseDataLoader.build_from_config(self.config)
        self.dataset = data_loader.load_datasets()

        # Tokenize the dataset
        self.dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        return self.dataset

    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize inputs, labels, and rationales for training.

        Args:
            examples: Dictionary containing batch of examples


        model_inputs['llama_input_ids'] = model_inputs['input_ids'] # encode the input
        model_inputs['llama_attention_mask'] = model_inputs['attention_mask'] # encode the input
        model_inputs['llama_labels'] = model_inputs['labels'] # encode the output
        model_inputs['llama_rationale'] = model_inputs['rationale'] # encode the rationale


        Returns:
            Dictionary containing tokenized inputs and labels
        """
        tokenizer = self.student_tokenizer
        max_input_length = self.training_config.max_input_length
        teacher_prefixes = self.training_config.teacher_prefixes  # e.g., ['t5', 'llama', 'gpt']

        # Tokenize main prediction inputs
        model_inputs = tokenizer(
            ['analyze the search query: ' + str(ex) for ex in examples['inputs']],
            max_length=max_input_length,
            padding="max_length",
            truncation=True
        )

        # Tokenize inputs for each teacher
        for prefix in teacher_prefixes:
            teacher_inputs = tokenizer(
                [f'rationale: ' + str(ex) for ex in examples['inputs']],
                max_length=max_input_length,
                padding="max_length",
                truncation=True
            )

            # Add teacher-specific tokenized inputs
            model_inputs.update({
                f'{prefix}_input_ids': teacher_inputs['input_ids'],
                f'{prefix}_attention_mask': teacher_inputs['attention_mask']
            })

        # Tokenize outputs (labels and rationales)
        with tokenizer.as_target_tokenizer():
            # Tokenize main task labels
            labels = tokenizer(
                [str(ex) for ex in examples['outputs']],
                max_length=max_input_length,
                truncation=self.generation_config.truncation
            )
            model_inputs['labels'] = labels['input_ids']

            # Tokenize rationales for each teacher
            for prefix in teacher_prefixes:
                rationale_key = f'{prefix}_rationale'
                if rationale_key in examples:
                    teacher_rationales = tokenizer(
                        [str(ex) for ex in examples[rationale_key]],
                        max_length=256,
                        truncation=True
                    )
                    model_inputs[f'{prefix}_labels'] = teacher_rationales['input_ids']

            return model_inputs

    def setup_trainer(self):
        """Setup the trainer with all components."""
        # Setup training arguments
        training_args = self.setup_training_args()

        # Setup data collator
        data_collator = MultiTeacherDataCollator(
            tokenizer=self.student_tokenizer,
            model=self.student_model,
            teacher_prefixes=self.training_config.teacher_prefixes
        )

        # Setup trainer kwargs
        trainer_kwargs = {
            'model': self.student_model,
            'args': training_args,
            'train_dataset': self.dataset["train"],
            'eval_dataset': self.dataset["valid"],
            'data_collator': data_collator,
            'tokenizer': self.student_tokenizer,
            'compute_metrics': compute_metrics_text(self.student_tokenizer),
            # Add teacher-specific parameters
            'alpha': self.training_config.alpha,
            'beta': self.training_config.beta,
            'gamma': self.training_config.gamma,
            'output_rationale': self.training_config.output_rationale,
        }

        # Initialize trainer using registrable pattern
        self.trainer = BaseTrainer.build_from_config(self.config, **trainer_kwargs)

    def run(self):
        """Run the knowledge distillation process."""
        # Setup all components
        self.setup_data()
        self.setup_student_model()
        self.setup_trainer()

        # Train the model
        train_result = self.trainer.train()

        # Save the final model
        if self.training_config.save_strategy != 'no':
            self.trainer.save_model(self.output_dir)

        # Evaluate the model
        metrics = self.trainer.evaluate()

        return {
            "train_results": train_result,
            "eval_metrics": metrics,
            "model_path": self.output_dir
        }
