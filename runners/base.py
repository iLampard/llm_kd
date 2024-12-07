from abc import abstractmethod
from registrable import Registrable
from easyllm_kit.models import LLM
from ..data_process.base import BaseDataLoader

class Runner(Registrable):
    @staticmethod
    def build_from_config(config, **kwargs):
        runner_cls = Runner.by_name(config["model_config"].model_name.lower())
        return runner_cls(config)
    
    @abstractmethod
    def run(self):
        pass
    
    
Runner.register('kd_runner') # knowledge distilation
class KnowledgeDistillationRunner(Runner):
    
    def setup_student_model(self):
        self.student_model, self.student_tokenizer = LLM.build_from_config(self.model_config)
    
    def setup_data(self) -> DatasetDict:
        """Setup and load the dataset."""
        data_loader = BaseDataLoader.build_from_config(self.config)
        self.dataset = data_loader.load_datasets()
        
        # Tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        return self.dataset
    
    def run(self):
        """Run the knowledge distillation process."""
        # Setup all components
        self.setup_data()
        self.setup_tokenizer()
        self.setup_models()
        self.setup_trainer()
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model(self.training_config.output_dir)
        
        # Evaluate the model
        metrics = self.trainer.evaluate()
        
        return {
            "train_results": train_result,
            "eval_metrics": metrics,
            "model_path": self.training_config.output_dir
        }
