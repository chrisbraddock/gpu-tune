import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import colorlog
import time
import csv
from datetime import datetime
import os
from gpu_metrics_utils import initialize_nvml, shutdown_nvml, get_gpu_metrics

# Configuration Constants
GPU_IDS = [0, 1]  # List of GPU IDs to use
SEQ_LENGTH = 256  # Sequence length for inference
BATCH_SIZE = 512  # Batch size for inference
MODEL_VARIANT = 'gpt2'  # Model variant ('gpt2', 'gpt2-medium', 'gpt2-large')
LOG_FILE = 'inference_stats.csv'  # Log file for metrics
MAX_ITERATIONS = 5  # Fixed number of iterations for inference

# Get MAX_WATT from environment variable
MAX_WATT = os.getenv('MAX_WATT', 'N/A')

# Setup color logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(name)s:%(message)s'))

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def log_statistics(file_name, headers, data):
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

def load_across_gpus(seq_length, batch_size, model_variant, max_iterations, callback=None):
    logger.info("Starting LLM inference")
    logger.info(f"Sequence length: {seq_length}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Model variant: {model_variant}")
    logger.info(f"Max iterations: {max_iterations}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Allocate GPU memory
    max_memory_mapping = {0: "10GB", 1: "10GB"}
    model = GPT2LMHeadModel.from_pretrained(model_variant, device_map="auto", max_memory=max_memory_mapping)
    tokenizer = GPT2Tokenizer.from_pretrained(model_variant)

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model and tokenizer initialized")

    initialize_nvml()

    start_time = time.time()
    total_tokens = 0

    # Get sample GPU metrics to dynamically generate headers
    sample_metrics = get_gpu_metrics()[0]
    gpu_headers = list(sample_metrics.keys())
    headers = ['timestamp', 'tokens_per_sec'] + gpu_headers + ['max_watt']

    for iteration in range(max_iterations):
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(["Hello, my name is"] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            logger.info(f"Running inference iteration {iteration + 1}/{max_iterations}")
            batch_start_time = time.time()
            outputs = model.generate(inputs['input_ids'], max_length=seq_length, attention_mask=inputs['attention_mask'])
            batch_end_time = time.time()

            batch_time = batch_end_time - batch_start_time
            # Count actual tokens generated
            if hasattr(outputs, 'shape'):
                actual_tokens = outputs.shape[-1] * outputs.shape[0]
            else:
                actual_tokens = sum([o.shape[-1] for o in outputs])
            total_tokens += actual_tokens

            logger.info(f"Inference completed in {batch_time:.4f} seconds")

            # Log statistics after each iteration
            timestamp = datetime.now().isoformat()
            tokens_per_sec = total_tokens / (time.time() - start_time)
            from gpu_metrics_utils import collect_power_draw_all_gpus
            total_power = collect_power_draw_all_gpus()
            gpu_metrics = get_gpu_metrics()[0]
            data = [timestamp, tokens_per_sec] + list(gpu_metrics.values()) + [MAX_WATT, total_power]
            if callback:
                data = callback(data)
            log_statistics(LOG_FILE, headers + ['total_power_draw'], data)
            logger.info(f"Logged statistics: {data}")

    shutdown_nvml()
    logger.info("LLM inference completed")

if __name__ == "__main__":
    def callback(data):
        # Add additional data to the payload
        additional_data = {"example": "example"}  # Replace with actual data
        return data + list(additional_data.values())

    if os.getenv('CALLBACK'):
        load_across_gpus(SEQ_LENGTH, BATCH_SIZE, MODEL_VARIANT, MAX_ITERATIONS, callback)
    else:
        load_across_gpus(SEQ_LENGTH, BATCH_SIZE, MODEL_VARIANT, MAX_ITERATIONS)
