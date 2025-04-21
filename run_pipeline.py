import os
import argparse
import subprocess
import logging
import time
import yaml
from datetime import datetime

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)
logger = logging.getLogger()

# Add console handler to see logs in real-time
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def run_command(command, description):
    """Run a command and log its output"""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    start_time = time.time()
    try:
        # Use PYTHONIOENCODING to ensure proper handling of UTF-8 characters
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
            env=env
        )
        
        # Stream and log output
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                line = line.rstrip()
                # Don't log lines that just update progress bars (lines that start with \r)
                if not line.startswith('\r'):
                    logger.info(line)
                    
        # Get any remaining output
        stdout, stderr = process.communicate()
        if stdout:
            for line in stdout.splitlines():
                if not line.startswith('\r'):
                    logger.info(line.strip())
        if stderr:
            for line in stderr.splitlines():
                logger.error(line.strip())
        
        if process.returncode != 0:
            logger.error(f"Error in {description}. Return code: {process.returncode}")
            return False
        
        duration = time.time() - start_time
        logger.info(f"Completed: {description} in {duration:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Exception in {description}: {str(e)}")
        return False

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_pipeline(args, config):
    """Run the complete speech intent recognition pipeline using config parameters"""
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(config['save_path'], exist_ok=True)
    
    # Step 1: Preprocess the data
    logger.info("=== STEP 1: PREPROCESSING DATA ===")
    preprocess_cmd = (
        f"python -m scripts.preprocess_fsc "
        f"--train_csv {args.train_csv} "
        f"--valid_csv {args.valid_csv} "
        f"--test_csv {args.test_csv} "
        f"--output_dir {args.output_dir} "
        f"--label_map_path {args.output_dir}/label_map.json"
    )
    if not run_command(preprocess_cmd, "Data Preprocessing"):
        logger.error("Preprocessing failed. Stopping pipeline.")
        return
    
    # Step 2: Train the model
    logger.info("=== STEP 2: TRAINING MODEL ===")
    train_cmd = (
        f"python -m scripts.train "
        f"--config {args.config_path} "
        f"--train_csv {args.output_dir}/train_data.csv "
        f"--val_csv {args.output_dir}/valid_data.csv "
        f"--label_map {args.output_dir}/label_map.json"
    )
    if not run_command(train_cmd, "Model Training"):
        logger.error("Training failed. Stopping pipeline.")
        return
    
    # Step 3: Evaluate the model
    logger.info("=== STEP 3: EVALUATING MODEL ===")
    eval_cmd = (
        f"python -m scripts.evaluate "
        f"--config {args.config_path} "
        f"--test_csv {args.output_dir}/test_data.csv "
        f"--label_map {args.output_dir}/label_map.json "
        f"--model_path {config['save_path']}/best_model.pt"
    )
    if not run_command(eval_cmd, "Model Evaluation"):
        logger.error("Evaluation failed.")
        return
    
    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    logger.info(f"Model saved at: {config['save_path']}/best_model.pt")
    logger.info(f"Evaluation results saved at: {config['save_path']}")
    logger.info(f"Log file: {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete Speech Intent Recognition pipeline")
    
    # Data paths
    parser.add_argument("--train_csv", type=str, 
                       default="data/FSC/fluent_speech_commands_dataset/data/train_data.csv")
    parser.add_argument("--valid_csv", type=str, 
                       default="data/FSC/fluent_speech_commands_dataset/data/valid_data.csv")
    parser.add_argument("--test_csv", type=str, 
                       default="data/FSC/fluent_speech_commands_dataset/data/test_data.csv")
    parser.add_argument("--output_dir", type=str, 
                       default="data/processed")
    parser.add_argument("--config_path", type=str, 
                       default="configs/config.yaml",
                       help="Path to the YAML configuration file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config_path)
    
    # Log the start of the pipeline
    logger.info("Starting Speech Intent Recognition Pipeline")
    logger.info(f"Arguments: {args}")
    logger.info(f"Configuration: {config}")
    
    # Run the pipeline
    run_pipeline(args, config)