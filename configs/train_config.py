from dataclasses import dataclass, field
from typing import List, Optional
from transformers import Seq2SeqTrainingArguments
from .base import Config
from easyllm_kit.configs.llm_base_config import ModelArguments, GenerationArguments


@dataclass
class DatasetArguments:
    # Required arguments
    input_keys: List[str]
    output_keys: List[str]
    data_name: str

    # Optional dataset arguments
    train_dir: Optional[str] = None
    valid_dir: Optional[str] = None
    test_dir: Optional[str] = None
    batch_size: int = 32
    rationale_key: Optional[str] = None


@dataclass
class KDTrainingArguments(Seq2SeqTrainingArguments):
    # Knowledge distillation specific arguments
    alpha: float = field(default=0.5, metadata={"help": "Weight for knowledge distillation loss"})
    beta: float = field(default=0.3, metadata={"help": "Weight for teacher prediction loss"})
    gamma: float = field(default=0.2, metadata={"help": "Weight for rationale generation loss"})
    output_rationale: bool = field(default=False, metadata={"help": "Whether to output rationale during generation"})

    # Training specific arguments
    eval_steps: int = field(default=500, metadata={"help": "Number of steps between evaluations"})
    max_steps: int = field(default=10000, metadata={"help": "Maximum number of training steps"})
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps for gradient accumulation"})
    gen_max_len: int = field(default=128, metadata={"help": "Maximum length for generation"})

    # Hardware and optimization arguments
    bf16: bool = field(default=False, metadata={"help": "Whether to use bf16 precision"})
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training"})

    def __post_init__(self):
        super().__post_init__()
        # Set default values for parent class attributes
        self.remove_unused_columns = False
        self.evaluation_strategy = "steps"
        self.save_strategy = "no"
        self.predict_with_generate = True
        self.prediction_loss_only = False


@Config.register('kd_trainer_config')
class KDTrainerConfig(Config):
    @staticmethod
    def parse_from_yaml_config(config: dict, **kwargs):
        model_config = ModelArguments(**config.get('model', {}))
        generation_config = GenerationArguments(**config.get('generation', {}))
        training_config = KDTrainingArguments(**config.get('training', {}))
        data_config = DatasetArguments(**config.get('data', {}))
        return {
            'model_config': model_config,
            'generation_config': generation_config,
            'training_config': training_config,
            'data_config': data_config
        }