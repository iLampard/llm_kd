from typing import List, Dict, Optional
import json
from ..arguments import FashionQuery

class DataPreprocessor:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data: List[FashionQuery] = []
    
    def load_data(self) -> List[FashionQuery]:
        """Load and parse JSON data into FashionQuery objects"""
        with open(self.json_path, 'r') as f:
            raw_data = json.load(f)
            
        self.data = [FashionQuery(**item) for item in raw_data]
        return self.data
    
    def get_field_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get value distribution for each field"""
        stats = {field: {} for field in FashionQuery.__annotations__.keys()}
        
        for entry in self.data:
            for field in stats.keys():
                value = getattr(entry, field)
                stats[field][value] = stats[field].get(value, 0) + 1
                
        return stats

    def to_training_format(self, filtered_data: Optional[List[FashionQuery]] = None) -> Dict[str, List]:
        """Convert data to training format"""
        if filtered_data is None:
            filtered_data = self.data
            
        return {
            'queries': [item.query for item in filtered_data],
            'labels': [{
                'main_category': item.main_category,
                'color': item.color,
                'gender': item.gender,
                'intention': item.intention
            } for item in filtered_data],
            'reasoning': [item.reasoning for item in filtered_data]
        }