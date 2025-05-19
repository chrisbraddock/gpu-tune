import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import logging
import colorlog
import time
import csv
from datetime import datetime
import os
from gpu_metrics_utils import initialize_nvml, shutdown_nvml, get_gpu_metrics

# Configuration Constants
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

def setup(rank, world_size):
    """Initialize the distributed process group."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()

def load_across_gpus(rank, world_size, batch_size, seq_length, epochs, learning_rate, callback=None):
    logger.info("Starting LLM training/fine-tuning")
    logger.info(f"Rank {rank}/{world_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Sequence length: {seq_length}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {learning_rate}")

    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
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

    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[rank])

    initialize_nvml()

    start_time = time.time()
    iteration = 0
    total_tokens = 0
    log_file = 'training_stats.csv'

    # Get sample GPU metrics to dynamically generate headers
    sample_metrics = get_gpu_metrics()[rank]
    gpu_headers = list(sample_metrics.keys())
    headers = ['timestamp', 'epoch', 'iteration', 'batch', 'loss', 'tokens_per_sec'] + gpu_headers + ['max_watt']

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

            tokens_this_batch = torch.tensor(batch_labels.numel(), device=device)
            dist.all_reduce(tokens_this_batch, op=dist.ReduceOp.SUM)
            total_tokens += tokens_this_batch.item()

            if rank == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Iteration {iteration}, Batch {i // batch_size + 1} completed, Loss: {loss.item()}"
                )

                # Log statistics after each batch
                timestamp = datetime.now().isoformat()
                tokens_per_sec = total_tokens / (time.time() - start_time)
                from gpu_metrics_utils import collect_power_draw_all_gpus
                total_power = collect_power_draw_all_gpus()
                gpu_metrics = get_gpu_metrics()[rank]
                data = [
                    timestamp,
                    epoch + 1,
                    iteration,
                    i // batch_size + 1,
                    loss.item(),
                    tokens_per_sec,
                ] + list(gpu_metrics.values()) + [MAX_WATT, total_power]
                if callback:
                    data = callback(data)
                log_statistics(log_file, headers + ["total_power_draw"], data)
                logger.info(f"Logged statistics: {data}")

    shutdown_nvml()
    cleanup()
    logger.info("LLM training/fine-tuning completed")

if __name__ == "__main__":
    def callback(data):
        # Add additional data to the payload
        additional_data = {"example": "example"}  # Replace with actual data
        return data + list(additional_data.values())

    # torchrun sets these environment variables when launching multiple processes
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    # Provide defaults for local standalone runs
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    if os.getenv("CALLBACK"):
        load_across_gpus(rank, world_size, BATCH_SIZE, SEQ_LENGTH, EPOCHS, LEARNING_RATE, callback)
    else:
        load_across_gpus(rank, world_size, BATCH_SIZE, SEQ_LENGTH, EPOCHS, LEARNING_RATE)
