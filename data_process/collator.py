from transformers import DataCollatorForSeq2Seq
from typing import Dict, List


class MultiTeacherDataCollator(DataCollatorForSeq2Seq):
    """Data collator that handles multiple teacher models."""

    def __init__(self, tokenizer, model, teacher_prefixes):
        super().__init__(tokenizer=tokenizer, model=model)
        self.teacher_prefixes = teacher_prefixes

    def __call__(self, features: List[Dict]) -> Dict:
        """Prepare batch data for all models."""
        # Process main prediction features using parent class
        pred_features = [
            {
                'input_ids': f['input_ids'],
                'attention_mask': f['attention_mask'],
                'labels': f['labels']
            } for f in features
        ]
        pred_batch = super().__call__(pred_features)

        # Initialize batch with prediction features
        batch = {'pred': pred_batch}

        # Add teacher features if present
        for teacher in self.teacher_prefixes:
            if f'{teacher}_labels' in features[0]:
                teacher_features = [
                    {
                        'input_ids': f['input_ids'],
                        'attention_mask': f['attention_mask'],
                        'labels': f[f'{teacher}_labels']
                    } for f in features
                ]
                teacher_batch = super().__call__(teacher_features)
                batch[teacher] = teacher_batch

        return batch
