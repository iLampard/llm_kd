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
    # Knowledge distillation specific parameters
    run_id: int = field(default=0)

    trainer_name: str = field(
        default='multiteacher',
        metadata={"help": "Name of the trainer to use"}
    )

    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device for training and evaluation"}
    )

    alpha: float = field(
        default=0.5,
        metadata={"help": "Weight for teacher prediction loss"}
    )
    beta: float = field(
        default=0.3,
        metadata={"help": "Weight for auxiliary task loss"}
    )
    gamma: float = field(
        default=0.2,
        metadata={"help": "Weight for consistency loss"}
    )
    output_rationale: bool = field(
        default=False,
        metadata={"help": "Whether to output rationales during generation"}
    )
    teacher_prefixes: List[str] = field(
        default_factory=list,
        metadata={"help": "List of teacher model prefixes"}
    )
    pred_weight: float = field(
        default=0.4,
        metadata={"help": "Weight for main prediction model loss"}
    )

    def __post_init__(self):
        """
        Post-initialization processing to set up derived parameters
        and validate configurations.
        """
        super().__post_init__()
        # Dynamically add weight fields for each teacher
        if self.teacher_prefixes:
            default_teacher_weight = (1.0 - self.pred_weight) / len(self.teacher_prefixes)
            for prefix in self.teacher_prefixes:
                if not hasattr(self, f'{prefix}_weight'):
                    setattr(self, f'{prefix}_weight',
                            field(default=default_teacher_weight,
                                  metadata={"help": f"Weight for {prefix} teacher loss"}))


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
            'data_config': data_config,
            **kwargs
        }
