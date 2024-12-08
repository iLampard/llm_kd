from datasets import Dataset, DatasetDict
from registrable import Registrable
from dataclasses import asdict
from typing import List, Dict
from easyllm_kit.utils import read_json


class BaseDataLoader(Registrable):
    """Base class for loading and processing specific datasets."""

    def __init__(self, **kwargs):
        """Base initialization that can be extended by child classes."""
        pass

    @staticmethod
    def build_from_config(config):
        data_config = asdict(config["data_config"])
        return BaseDataLoader.by_name(data_config["data_name"])(**data_config)

    def load_datasets(self) -> DatasetDict:
        """Loads datasets from JSON files."""
        raise NotImplementedError

    def _post_process(self, datasets: DatasetDict) -> DatasetDict:
        """Post-process the datasets. Should be implemented by child classes."""
        raise NotImplementedError

    def _parse_llm_output(self, output: Dict) -> (str, str):
        """Parses LLM output to extract rationale and label."""
        # Implement parsing logic based on your LLM output format
        raise NotImplementedError


@BaseDataLoader.register('query_intention')
class QueryIntentionDataLoader(BaseDataLoader):
    def __init__(self,
                 train_dir: str,
                 valid_dir: str,
                 test_dir: str,
                 batch_size: int,
                 input_keys: List[str] = None,
                 output_keys: List[str] = None,
                 rationale_key: str = None,
                 **kwargs):
        """
        Initializes the dataset loader.

        Args:
            train_dir (str): Directory for training data.
            valid_dir (str): Directory for validation data.
            test_dir (str): Directory for test data.
            batch_size (int): Size of each data batch.
            input_keys (list): Keys for input features.
            output_keys (list): Keys for output labels.
            rationale_key (str): Key for rationale.
        """
        super().__init__(**kwargs)
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.rationale_key = rationale_key

    def _parse_llm_output(self, output: Dict) -> (str, str):
        """Parses LLM output to extract rationale and label."""
        # Implement parsing logic based on your LLM output format
        rationale = output.get('rationale', '')
        label = output.get('label', '')
        return rationale, label

    def _extract_data(self, data: Dict) -> List[Dict]:
        """Extracts input, output, and rationale data from the JSON."""
        extracted_data = []
        for entry in data.values():
            inputs = {key: entry[key] for key in self.input_keys}
            outputs = {key: entry[key] for key in self.output_keys}
            rationale = entry.get(self.rationale_key, "")
            extracted_data.append({'inputs': inputs, 'outputs': outputs, 'rationale': rationale})
        return extracted_data

    def load_datasets(self) -> DatasetDict:
        """Loads datasets from JSON files."""
        datasets = {}
        for split, dir_path in [('train', self.train_dir), ('test', self.test_dir), ('valid', self.valid_dir)]:
            if dir_path:
                data = read_json(dir_path)
                extracted_data = self._extract_data(data)
                # Convert list of dictionaries to dictionary of lists
                dict_data = {
                    key: [item[key] for item in extracted_data]
                    for key in ['inputs', 'outputs', 'rationale']
                }
                datasets[split] = Dataset.from_dict(dict_data)
        return DatasetDict(datasets)
