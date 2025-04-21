import os
import sys
import yaml
import logging
import argparse
import subprocess
from scripts.preprocess_fsc import preprocess_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_subprocess(cmd, name):
    logger.info(f"Command: {cmd}")
    process = subprocess.run(cmd, shell=True)
    
    if process.returncode != 0:
        logger.error(f"Error in {name}. Return code: {process.returncode}")
        return False
    
    logger.info(f"{name} completed successfully.")
    return True

def run_pipeline(config_path):
    logger.info("=== Starting Speech Intent Recognition Pipeline ===")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set environment
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Preprocess data
    logger.info("=== STEP 1: DATA PREPROCESSING ===")
    logger.info("Starting: Data Preprocessing")
    
    preprocess_result = preprocess_dataset(
        train_csv=config.get('train_csv', 'data/raw/train.csv'),
        valid_csv=config.get('valid_csv', 'data/raw/valid.csv'),
        test_csv=config.get('test_csv', 'data/raw/test.csv'),
        output_dir=config.get('output_dir', 'data/processed'),
        label_map_path=config.get('label_map_path', 'data/processed/label_map.json')
    )
    
    if not preprocess_result:
        logger.error("Data preprocessing failed. Stopping pipeline.")
        return False
    
    logger.info("Data Preprocessing completed successfully.")
    
    # Step 2: Train model
    logger.info("=== STEP 2: TRAINING MODEL ===")
    logger.info("Starting: Model Training")
    
    train_csv = preprocess_result['train_csv']
    val_csv = preprocess_result['valid_csv']
    label_map = preprocess_result['label_map']
    
    training_cmd = f"python -m scripts.train --config {config_path} --train_csv {train_csv} --val_csv {val_csv} --label_map {label_map}"
    if not run_subprocess(training_cmd, "Model Training"):
        logger.error("Training failed. Stopping pipeline.")
        return False
    
    # Step 3: Evaluate model
    logger.info("=== STEP 3: EVALUATING MODEL ===")
    logger.info("Starting: Model Evaluation")
    
    test_csv = preprocess_result['test_csv']
    model_path = os.path.join(config.get('save_path', 'checkpoints'), 'best_model.pt')
    
    evaluation_cmd = f"python -m scripts.evaluate --config {config_path} --test_csv {test_csv} --label_map {label_map} --model_path {model_path}"
    if not run_subprocess(evaluation_cmd, "Model Evaluation"):
        logger.error("Evaluation failed. Stopping pipeline.")
        return False
    
    logger.info("=== Pipeline Completed Successfully ===")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the full Speech Intent Recognition pipeline')
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    run_pipeline(args.config_path)