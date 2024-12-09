from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from transformers import DataCollatorForSeq2Seq


@dataclass
class MultiTeacherDataCollator(DataCollatorForSeq2Seq):
    """
    A flexible data collator that handles any number of teacher models.

    Args:
        tokenizer: The tokenizer to use for padding
        model: The model to use for preparing inputs
        teacher_prefixes: List of teacher model prefixes
        label_key: Key for main task labels (default: 'labels')
        input_id_key: Key for input ids (default: 'input_ids')
        attention_mask_key: Key for attention mask (default: 'attention_mask')
    """
    # Required arguments first
    teacher_prefixes: List[str] = field(default_factory=list)

    label_key: str = 'labels'
    input_id_key: str = 'input_ids'
    attention_mask_key: str = 'attention_mask'

    def __call__(self, features, return_tensors=None) -> Dict:
        """
        Prepare batch data for all models.

        Args:
            features: List of dictionaries containing the features
            return_tensors: Type of tensors to return

        Returns:
            Dictionary containing prepared features for prediction model and all teachers
        """
        features_df = pd.DataFrame(features)
        prepared_features = {}

        # Get all column names
        all_columns = set(features_df.columns)

        # Prepare main prediction features
        pred_columns = self._get_model_columns(all_columns, prefix='pred')
        if pred_columns:
            pred_features = features_df.loc[:, pred_columns].to_dict('records')
            prepared_features['pred'] = super().__call__(pred_features, return_tensors)

        # Prepare features for each teacher
        for teacher in self.teacher_prefixes:
            teacher_columns = self._get_model_columns(all_columns, prefix=teacher)
            if not teacher_columns:
                continue

            # Extract and rename teacher features
            teacher_features = (
                features_df.loc[:, teacher_columns]
                .rename(columns=self._get_rename_mapping(teacher))
                .to_dict('records')
            )

            prepared_features[teacher] = super().__call__(teacher_features, return_tensors)

        return prepared_features

    def _get_model_columns(self, all_columns: set, prefix: str) -> List[str]:
        """
        Get relevant columns for a specific model prefix.

        Args:
            all_columns: Set of all available column names
            prefix: Model prefix to filter columns for

        Returns:
            List of column names relevant for the model
        """
        if prefix == 'pred':
            # For prediction model, get columns without any teacher prefix
            return [col for col in all_columns
                    if not any(f"{teacher}_" in col
                               for teacher in self.teacher_prefixes)]
        else:
            # For teacher models, get columns with specific teacher prefix
            return [col for col in all_columns if col.startswith(f"{prefix}_")]

    def _get_rename_mapping(self, teacher: str) -> Dict[str, str]:
        """
        Create mapping for renaming teacher-specific columns to standard names.

        Args:
            teacher: Teacher prefix

        Returns:
            Dictionary mapping teacher-specific column names to standard names
        """
        return {
            f'{teacher}_{self.label_key}': self.label_key,
            f'{teacher}_{self.input_id_key}': self.input_id_key,
            f'{teacher}_{self.attention_mask_key}': self.attention_mask_key
        }
