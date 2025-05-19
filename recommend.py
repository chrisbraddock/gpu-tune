import pandas as pd

# Load the data
training_stats = pd.read_csv('training_stats.csv')
inference_stats = pd.read_csv('inference_stats.csv')

# Convert the timestamp column to datetime
training_stats['timestamp'] = pd.to_datetime(training_stats['timestamp'])
inference_stats['timestamp'] = pd.to_datetime(inference_stats['timestamp'])

# Function to calculate summary statistics
def calculate_summary(data):
    numeric_data = data.select_dtypes(include='number')
    numeric_data['max_watt'] = data['max_watt']
    grouped = numeric_data.groupby('max_watt').mean()  # Use mean instead of median
    grouped['total_time_min'] = data.groupby('max_watt').apply(lambda x: (x['timestamp'].max() - x['timestamp'].min()).total_seconds() / 60.0)
    grouped['energy_consumption_watt_min'] = grouped['power_draw'] * grouped['total_time_min']
    return grouped

# Function to recommend sweet spot
def recommend_sweet_spot(summary, metric='energy_consumption_watt_min'):
    optimal_watt = None
    min_energy_consumption = float('inf')
    for watt in summary.index:
        energy_consumption = summary.loc[watt, 'energy_consumption_watt_min']
        if energy_consumption < min_energy_consumption:
            min_energy_consumption = energy_consumption
            optimal_watt = watt
    return optimal_watt

if __name__ == "__main__":
    # Generate summary tables
    training_summary = calculate_summary(training_stats)
    inference_summary = calculate_summary(inference_stats)

    # Recommend sweet spot for training and inference
    optimal_training_watt = recommend_sweet_spot(training_summary, 'energy_consumption_watt_min')
    optimal_inference_watt = recommend_sweet_spot(inference_summary, 'energy_consumption_watt_min')

    # Print summaries and recommendations
    print("Training Summary:")
    print(training_summary)
    print(f"Optimal Max Power for Training: {optimal_training_watt}W\n")

    print("Inference Summary:")
    print(inference_summary)
    print(f"Optimal Max Power for Inference: {optimal_inference_watt}W\n")

    # Save summaries to CSV
    training_summary.to_csv('training_summary.csv')
    inference_summary.to_csv('inference_summary.csv')
