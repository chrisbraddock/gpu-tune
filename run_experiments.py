import subprocess
import time
import os
import logging
import colorlog
from gpu_metrics_utils import set_gpu_power_limit, initialize_nvml, shutdown_nvml

# Configuration Constants
GPU_IDS = [0, 1]  # List of GPU IDs to use
MAX_WATT_VALUES = [150, 175, 200, 225, 250, 275, 300]  # Example values, adjust as needed
TRAINING_SCRIPT = "./llm_training.py"
INFERENCE_SCRIPT = "./llm_inference.py"
LOG_FILE = "experiment.log"
ITERATION_SLEEP=10

# Setup color logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(name)s:%(message)s'))

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))
logger.addHandler(file_handler)

def run_script(script, max_watt):
    """Run a specified script with max_watt as an environment variable."""
    try:
        env = os.environ.copy()
        env['MAX_WATT'] = str(max_watt)
        subprocess.run(f"python {script}", shell=True, check=True, env=env)
        logger.info(f"Completed running {script}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run script {script}: {e}")

def main():
    initialize_nvml()
    for max_watt in MAX_WATT_VALUES:
        # Set the power limit for both GPUs
        set_gpu_power_limit(GPU_IDS, max_watt)

        # Log the experiment immediately after setting the power limit
        logger.info(f"Starting experiment with max_watt {max_watt}")

        # Run the training script
        logger.info(f"Running the training script {TRAINING_SCRIPT}")
        run_script(TRAINING_SCRIPT, max_watt)

        # Run the inference script
        logger.info(f"Running the inference script {INFERENCE_SCRIPT}")
        run_script(INFERENCE_SCRIPT, max_watt)

        # Pause between iterations (optional)
        time.sleep(ITERATION_SLEEP)

    shutdown_nvml()

if __name__ == "__main__":
    main()
