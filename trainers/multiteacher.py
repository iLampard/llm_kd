from typing import Dict, List, Optional, Tuple
import torch
from trainers.base import BaseTrainer


@BaseTrainer.register('multiteacher')
class MultiTeacherTrainer(BaseTrainer):
    """
    A flexible trainer for sequence-to-sequence models with multiple teachers.
    """

    def __init__(
            self,
            model,
            args,
            train_dataset,
            eval_dataset,
            data_collator,
            tokenizer,
            compute_metrics,
            teacher_prefixes: List[str]
    ):
        """
        Initialize the trainer with multiple teachers.

        Args:
            model: The model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator
            tokenizer: Tokenizer
            compute_metrics: Metrics computation function
            teacher_prefixes: List of teacher model prefixes
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        self.teacher_prefixes = teacher_prefixes
        # Get loss weights from training args
        self.loss_weights = {
            'pred': getattr(self.args, 'pred_weight', 0.4),
            **{prefix: getattr(self.args, f'{prefix}_weight', 0.3 / len(teacher_prefixes))
               for prefix in teacher_prefixes}
        }
        self.output_rationale = getattr(self.args, 'output_rationale', False)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute combined loss from prediction model and all teachers."""
        outputs = {}
        total_loss = 0.0

        # Compute loss for prediction model
        if 'pred' in inputs:
            pred_outputs = model(**inputs['pred'])
            total_loss += self.loss_weights['pred'] * pred_outputs.loss
            outputs['pred'] = pred_outputs

        # Compute loss for each teacher
        for teacher in self.teacher_prefixes:
            if teacher in inputs:
                teacher_outputs = model(**inputs[teacher])
                total_loss += self.loss_weights[teacher] * teacher_outputs.loss
                outputs[teacher] = teacher_outputs

        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(
            self,
            model: torch.nn.Module,
            inputs: Dict,
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform prediction step for all models."""
        all_losses = []
        all_logits = []
        all_labels = []

        # Get predictions for main model
        if 'pred' in inputs:
            pred_loss, pred_logits, pred_labels = super().prediction_step(
                model, inputs['pred'], prediction_loss_only, ignore_keys
            )
            all_losses.append(self.loss_weights['pred'] * pred_loss)
            all_logits.append(pred_logits)
            all_labels.append(pred_labels)

        # Get predictions for each teacher if rationale generation is enabled
        if self.output_rationale:
            for teacher in self.teacher_prefixes:
                if teacher in inputs:
                    teacher_loss, teacher_logits, teacher_labels = super().prediction_step(
                        model, inputs[teacher], prediction_loss_only, ignore_keys
                    )
                    all_losses.append(self.loss_weights[teacher] * teacher_loss)
                    all_logits.append(teacher_logits)
                    all_labels.append(teacher_labels)

        # Combine losses and return results
        combined_loss = torch.stack(all_losses).sum() if all_losses else None
        return (
            combined_loss,
            torch.cat(all_logits, dim=1) if all_logits else None,
            torch.cat(all_labels, dim=1) if all_labels else None
        )
