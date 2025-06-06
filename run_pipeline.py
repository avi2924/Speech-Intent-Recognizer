import os
import sys
import yaml
import logging
import argparse
import subprocess
import torch
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

def run_subprocess(cmd, name, env=None):
    logger.info(f"Command: {cmd}")
    
    # Use environment variables if provided
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    
    process = subprocess.run(cmd, shell=True, env=env_vars)
    
    if process.returncode != 0:
        logger.error(f"Error in {name}. Return code: {process.returncode}")
        return False
    
    logger.info(f"{name} completed successfully.")
    return True

def run_pipeline(config_path):
    # Set environment variables for better GPU usage
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # This will print GPU memory usage for debugging
    def print_gpu_memory():
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024
            reserved_memory = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(f"GPU Memory: {allocated_memory:.0f}MB allocated / {reserved_memory:.0f}MB reserved")
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    print_gpu_memory()
    
    logger.info("=== Starting Speech Intent Recognition Pipeline ===")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set environment
    python_path = os.path.dirname(os.path.abspath(__file__))
    os.environ['PYTHONPATH'] = python_path
    
    # Check if data files exist, if not, use any available data
    train_csv = config.get('train_csv', 'data/raw/train.csv')
    valid_csv = config.get('valid_csv', 'data/raw/valid.csv')
    test_csv = config.get('test_csv', 'data/raw/test.csv')
    
    # Check if files exist and use fallbacks if needed
    if not os.path.exists(train_csv):
        potential_paths = [
            'data/processed/train_data.csv',
            'data/FSC/fluent_speech_commands_dataset/data/train_data.csv',
            'data/train_data.csv'
        ]
        for path in potential_paths:
            if os.path.exists(path):
                train_csv = path
                logger.info(f"Using alternative train data path: {train_csv}")
                break
    
    if not os.path.exists(valid_csv):
        potential_paths = [
            'data/processed/valid_data.csv',
            'data/FSC/fluent_speech_commands_dataset/data/valid_data.csv',
            'data/valid_data.csv'
        ]
        for path in potential_paths:
            if os.path.exists(path):
                valid_csv = path
                logger.info(f"Using alternative validation data path: {valid_csv}")
                break
    
    if not os.path.exists(test_csv):
        potential_paths = [
            'data/processed/test_data.csv',
            'data/FSC/fluent_speech_commands_dataset/data/test_data.csv',
            'data/test_data.csv'
        ]
        for path in potential_paths:
            if os.path.exists(path):
                test_csv = path
                logger.info(f"Using alternative test data path: {test_csv}")
                break
    
    # Update config with found paths
    config['train_csv'] = train_csv
    config['valid_csv'] = valid_csv
    config['test_csv'] = test_csv
    
    # Check if files exist after fallback search
    if not all(os.path.exists(p) for p in [train_csv, valid_csv, test_csv]):
        logger.error("Could not find required data files. Please check your data paths.")
        return False
    
    # Step 1: Preprocess data (should be first)
    logger.info("=== STEP 1: DATA PREPROCESSING ===")
    logger.info("Starting: Data Preprocessing")
    
    # Create output directory
    output_dir = config.get('output_dir', 'data/processed')
    os.makedirs(output_dir, exist_ok=True)
    
    preprocess_result = preprocess_dataset(
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
        output_dir=output_dir,
        label_map_path=config.get('label_map_path', os.path.join(output_dir, 'label_map.json')),
        use_torchaudio=True  # Use torchaudio for faster validation
    )
    
    if not preprocess_result:
        logger.error("Preprocessing failed. Stopping pipeline.")
        return False
    
    # Update paths with processed data
    train_csv = preprocess_result['train_csv']
    valid_csv = preprocess_result['valid_csv']
    test_csv = preprocess_result['test_csv']
    label_map = preprocess_result['label_map']
    
    # Step 2: Precompute features (after preprocessing)
    use_feature_cache = config.get('use_feature_cache', True)
    if use_feature_cache:
        logger.info("=== STEP 2: PRECOMPUTING FEATURES ===")
        
        cache_dir = config.get('cache_dir', 'data/cached_features')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if cached features already exist
        train_cache_path = os.path.join(cache_dir, f"{os.path.basename(train_csv).replace('.csv', '')}_features.pt")
        
        # Only precompute if cache doesn't exist or force_precompute is True
        if config.get('force_precompute', False) or not os.path.exists(train_cache_path):
            logger.info("Starting: Feature Precomputation")
            
            precompute_cmd = (
                f"python -m scripts.precompute_features "
                f"--train_csv {train_csv} "
                f"--valid_csv {valid_csv} "
                f"--test_csv {test_csv} "
                f"--output_dir {cache_dir} "
                f"--label_map {label_map}"
            )
            
            if not run_subprocess(precompute_cmd, "Feature Precomputation"):
                # If feature precomputation fails, print more info but still continue
                logger.warning("Feature precomputation failed. Will continue without cached features.")
                logger.warning("This will make training slower but it should still work.")
                config['use_feature_cache'] = False
            else:
                logger.info("Feature precomputation completed successfully")
        else:
            logger.info(f"Using existing cached features in {cache_dir}")
    
    # Step 3: Train model (was Step 2 before)
    logger.info("=== STEP 3: TRAINING MODEL ===")
    logger.info("Starting: Model Training")
    
    # Ensure output directory exists
    save_path = config.get('save_path', 'checkpoints')
    os.makedirs(save_path, exist_ok=True)
    
    # Add cache info to training command
    training_cmd = (
        f"python -m scripts.train "
        f"--config {config_path} "
        f"--train_csv {train_csv} "
        f"--val_csv {valid_csv} "
        f"--label_map {label_map}"
    )
    
    if not run_subprocess(training_cmd, "Model Training", {'PYTHONPATH': python_path}):
        logger.error("Training failed. Stopping pipeline.")
        return False
    
    # Step 4: Evaluate model
    logger.info("=== STEP 4: EVALUATING MODEL ===")
    logger.info("Starting: Model Evaluation")
    
    model_path = os.path.join(config.get('save_path', 'checkpoints'), 'best_model.pt')
    
    # Check if the model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    evaluation_cmd = f"python -m scripts.evaluate --config {config_path} --test_csv {test_csv} --label_map {label_map} --model_path {model_path}"
    if not run_subprocess(evaluation_cmd, "Model Evaluation", {'PYTHONPATH': python_path}):
        logger.error("Evaluation failed. Stopping pipeline.")
        return False
    
    logger.info("=== Pipeline Completed Successfully ===")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the full Speech Intent Recognition pipeline')
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--force_precompute', action='store_true', help='Force precomputation of features even if cache exists')
    args = parser.parse_args()
    
    # If force_precompute is set via command line, update config
    if args.force_precompute:
        config = load_config(args.config_path)
        config['force_precompute'] = True
        
        # Save updated config temporarily
        temp_config_path = 'configs/temp_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        run_pipeline(temp_config_path)
        
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    else:
        run_pipeline(args.config_path)