import argparse
from runners import Runner
from configs import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_config.yaml')
    args = parser.parse_args()
    
    config = Config.build_from_yaml_file(args.config_dir)
    
    runner = Runner.build_from_config(config)
    
    runner.run()
    