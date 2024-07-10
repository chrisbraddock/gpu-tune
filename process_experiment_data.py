import pandas as pd

# Configuration Constants
EXPERIMENT_LOG_FILE = 'experiment_log.csv'
GPU_METRICS_LOG_FILE = 'gpu_metrics_log.csv'
INFERENCE_STATS_FILE = 'inference_stats.csv'
TRAINING_STATS_FILE = 'training_stats.csv'
OUTPUT_TRAINING_FILE = 'processed_training_data.csv'
OUTPUT_INFERENCE_FILE = 'processed_inference_data.csv'
MERGE_TOLERANCE = pd.Timedelta(seconds=5)  # Tolerance for merging

def read_csv(file_path):
    """Read CSV file into a DataFrame."""
    df = pd.read_csv(file_path)
    assert len(df) > 0, f"File {file_path} is empty or missing!"
    return df

def merge_data(main_df, additional_df, on='timestamp', tolerance=MERGE_TOLERANCE):
    """Merge dataframes based on the closest preceding timestamp with a tolerance."""
    additional_df['timestamp'] = pd.to_datetime(additional_df['timestamp'])
    merged_df = pd.merge_asof(main_df.sort_values('timestamp'),
                              additional_df.sort_values('timestamp'),
                              on='timestamp', direction='backward', tolerance=tolerance)
    return merged_df

def process_inference_data(experiment_log, inference_stats, gpu_metrics):
    """Process and merge data for inference."""
    experiment_log['timestamp'] = pd.to_datetime(experiment_log['timestamp'])
    inference_stats['timestamp'] = pd.to_datetime(inference_stats['timestamp'])
    gpu_metrics['timestamp'] = pd.to_datetime(gpu_metrics['timestamp'])

    merged_df = merge_data(inference_stats, experiment_log)
    merged_df = merge_data(merged_df, gpu_metrics)

    return merged_df

def process_training_data(experiment_log, training_stats, gpu_metrics):
    """Process and merge data for training."""
    experiment_log['timestamp'] = pd.to_datetime(experiment_log['timestamp'])
    training_stats['timestamp'] = pd.to_datetime(training_stats['timestamp'])
    gpu_metrics['timestamp'] = pd.to_datetime(gpu_metrics['timestamp'])

    merged_df = merge_data(training_stats, experiment_log)
    merged_df = merge_data(merged_df, gpu_metrics)

    return merged_df

def save_to_csv(df, file_path):
    """Save DataFrame to CSV file."""
    df.to_csv(file_path, index=False)

def main():
    # Read data from CSV files
    experiment_log = read_csv(EXPERIMENT_LOG_FILE)
    inference_stats = read_csv(INFERENCE_STATS_FILE)
    training_stats = read_csv(TRAINING_STATS_FILE)
    gpu_metrics = read_csv(GPU_METRICS_LOG_FILE)

    # Process and merge data for training
    processed_training_data = process_training_data(experiment_log, training_stats, gpu_metrics)

    # Process and merge data for inference
    processed_inference_data = process_inference_data(experiment_log, inference_stats, gpu_metrics)

    # Save processed data to new CSV files
    save_to_csv(processed_training_data, OUTPUT_TRAINING_FILE)
    save_to_csv(processed_inference_data, OUTPUT_INFERENCE_FILE)
    print(f"Processed training data saved to {OUTPUT_TRAINING_FILE}")
    print(f"Processed inference data saved to {OUTPUT_INFERENCE_FILE}")

if __name__ == "__main__":
    main()
