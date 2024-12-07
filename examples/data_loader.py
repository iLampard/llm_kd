from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import argparse
from data_process import BaseDataLoader
from configs import Config


def test_data_loader(data_loader):
    """Test the data loader functionality"""
    # Load datasets
    datasets = data_loader.load_datasets()

    # Print basic statistics
    print("\nDataset Statistics:")
    for split, dataset in datasets.items():
        print(f"\n{split.upper()} split:")
        print(f"Number of examples: {len(dataset)}")

        # Show first example
        if len(dataset) > 0:
            print("\nFirst example:")
            example = dataset[0]
            print("Inputs:", example['inputs'])
            print("Outputs:", example['outputs'])
            print("Rationale:", example['rationale'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../train_config.yaml')
    parser.add_argument('--test_loader', action='store_true', help='Test the data loader functionality')
    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config)

    if args.test_loader:
        # Create data loader instance from config
        data_loader = BaseDataLoader.build_from_config(config)
        test_data_loader(data_loader)