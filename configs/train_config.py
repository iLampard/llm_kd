from .base import Config
from dataclasses import dataclass
from typing import List
from transformers import TrainingArguments
from easyllm_kit.configs.llm_base_config import ModelArguments, GenerationArguments


@dataclass
class DatasetArguments:
    train_dir: str
    valid_dir: str
    test_dir: str
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4

    input_keys: List[str]
    output_keys: List[str]
    rationale_key: str


@Config.register('kd_trainer_config')
class KDTrainerConfig(Config):
    @staticmethod
    def parse_from_yaml_config(config: dict, **kwargs):
        model_config = ModelArguments(**config.get('model', {}))
        generation_config = GenerationArguments(**config.get('generation', {}))
        training_config = TrainingArguments(**config.get('training', {}))
        data_config = DatasetArguments(**config.get('data', {}))
        return {'model_config': model_config,
                'generation_config': generation_config,
                'training_config': training_config,
                'data_config': data_config}
