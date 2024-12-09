from transformers import Seq2SeqTrainer
from registrable import Registrable


class BaseTrainer(Seq2SeqTrainer, Registrable):
    """Base trainer class that implements Registrable."""

    @staticmethod
    def build_from_config(config, **kwargs):
        """Build trainer from config."""
        trainer_cls = BaseTrainer.by_name(config["trainer_name"].lower())
        return trainer_cls(
            model=kwargs.get('model'),
            args=config["training_config"],
            train_dataset=kwargs.get('train_dataset'),
            eval_dataset=kwargs.get('eval_dataset'),
            data_collator=kwargs.get('data_collator'),
            tokenizer=kwargs.get('tokenizer'),
            compute_metrics=kwargs.get('compute_metrics'),
            teacher_prefixes=config["training_config"].teacher_prefixes
        )
