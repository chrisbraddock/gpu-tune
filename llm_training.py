import torch
from torch.nn import DataParallel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import logging
import colorlog
import time
import csv
from datetime import datetime
import os
from gpu_metrics_utils import initialize_nvml, shutdown_nvml, get_gpu_metrics

# Configuration Constants
GPU_IDS = [0, 1]  # List of GPU IDs to use
BATCH_SIZE = 1975  # Batch size for training
SEQ_LENGTH = 2048  # Sequence length for training
EPOCHS = 20  # Number of epochs to train
LEARNING_RATE = 5e-5  # Learning rate for the optimizer

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

def load_across_gpus(gpu_ids, batch_size, seq_length, epochs, learning_rate, callback=None):
    logger.info("Starting LLM training/fine-tuning")
    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Sequence length: {seq_length}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {learning_rate}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    logger.info("Model, tokenizer, and optimizer initialized")

    # Sample data for fine-tuning
    texts = ["Hello, my name is GPT-2.", "I enjoy learning new things!"] * 1000
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    labels = inputs['input_ids']

    # Use DataParallel to wrap the model for multi-GGPU usage
    model = DataParallel(model, device_ids=gpu_ids).to(device)

    initialize_nvml()

    start_time = time.time()
    iteration = 0
    total_tokens = 0
    log_file = 'training_stats.csv'

    # Get sample GPU metrics to dynamically generate headers
    sample_metrics = get_gpu_metrics()[0]
    gpu_headers = list(sample_metrics.keys())
    headers = ['timestamp', 'epoch', 'iteration', 'batch', 'loss', 'tokens_per_sec'] + gpu_headers + [
        'max_watt', 'total_power_draw', 'energy_per_token'
    ]

    model.train()
    for epoch in range(epochs):
        for i in range(0, len(texts), batch_size):
            iteration += 1
            batch_start_time = time.time()
            batch_inputs = {key: value[i:i + batch_size] for key, value in inputs.items()}
            batch_labels = labels[i:i + batch_size]

            outputs = model(**batch_inputs, labels=batch_labels)
            loss = outputs.loss.mean()  # Ensure the loss is a scalar
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            total_tokens += batch_labels.numel()  # More accurate token count

            logger.info(f"Epoch {epoch + 1}/{epochs}, Iteration {iteration}, Batch {i // batch_size + 1} completed, Loss: {loss.item()}")

            # Log statistics after each batch
            timestamp = datetime.now().isoformat()
            tokens_per_sec = total_tokens / (time.time() - start_time)
            from gpu_metrics_utils import collect_power_draw_all_gpus
            total_power = collect_power_draw_all_gpus()
            gpu_metrics = get_gpu_metrics()[0]
            energy_per_token = total_power / tokens_per_sec if tokens_per_sec else 0
            data = [
                timestamp,
                epoch + 1,
                iteration,
                i // batch_size + 1,
                loss.item(),
                tokens_per_sec,
                *list(gpu_metrics.values()),
                MAX_WATT,
                total_power,
                energy_per_token,
            ]
            if callback:
                data = callback(data)
            log_statistics(log_file, headers, data)
            logger.info(f"Logged statistics: {data}")

    shutdown_nvml()
    logger.info("LLM training/fine-tuning completed")

if __name__ == "__main__":
    def callback(data):
        # Add additional data to the payload
        additional_data = {"example": "example"}  # Replace with actual data
        return data + list(additional_data.values())

    if os.getenv('CALLBACK'):
        load_across_gpus(GPU_IDS, BATCH_SIZE, SEQ_LENGTH, EPOCHS, LEARNING_RATE, callback)
    else:
        load_across_gpus(GPU_IDS, BATCH_SIZE, SEQ_LENGTH, EPOCHS, LEARNING_RATE)
