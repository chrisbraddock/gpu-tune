import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.interpolate import make_interp_spline
import numpy as np
from datetime import datetime
from recommend import calculate_summary, recommend_sweet_spot

# Load the provided CSV files
training_stats = pd.read_csv('training_stats.csv')
inference_stats = pd.read_csv('inference_stats.csv')

# Convert the timestamp column to datetime
training_stats['timestamp'] = pd.to_datetime(training_stats['timestamp'])
inference_stats['timestamp'] = pd.to_datetime(inference_stats['timestamp'])

# Calculate total time for each row
inference_stats['time_diff'] = inference_stats['timestamp'].diff().dt.total_seconds().fillna(0)
training_stats['time_diff'] = training_stats['timestamp'].diff().dt.total_seconds().fillna(0)

# Function to remove outliers based on z-score
def remove_outliers(df, column):
    return df[(zscore(df[column]) < 3)]

# Remove outliers and then calculate mean for each max_power setting
# Inference Metrics
inference_cleaned = inference_stats.groupby('max_watt').apply(remove_outliers, column='tokens_per_sec').reset_index(drop=True)
inference_grouped = inference_cleaned.groupby('max_watt').agg({
    'tokens_per_sec': 'mean',
    'temperature': 'mean',
    'gpu_utilization': 'mean',
    'memory_utilization': 'mean',
    'time_diff': 'sum'
}).reset_index()

# Training Metrics
training_cleaned = training_stats.groupby('max_watt').apply(remove_outliers, column='tokens_per_sec').reset_index(drop=True)
training_grouped = training_cleaned.groupby('max_watt').agg({
    'tokens_per_sec': 'mean',
    'temperature': 'mean',
    'gpu_utilization': 'mean',
    'memory_utilization': 'mean',
    'loss': 'mean',
    'time_diff': 'sum'
}).reset_index()

# Generate summary tables
training_summary = calculate_summary(training_stats)
inference_summary = calculate_summary(inference_stats)

# Recommend sweet spot for training and inference
optimal_training_watt = recommend_sweet_spot(training_summary, 'energy_consumption_watt_min')
optimal_inference_watt = recommend_sweet_spot(inference_summary, 'energy_consumption_watt_min')

# Function to plot smooth curves
def plot_smooth_curve(ax, x, y, title, xlabel, ylabel, highlight_x=None):
    # Ensure x is sorted and strictly increasing
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]

    # Reduce the number of interpolation points for aesthetics
    x_new = np.linspace(x_unique.min(), x_unique.max(), 100)
    spl = make_interp_spline(x_unique, y_unique, k=3)
    y_smooth = spl(x_new)

    ax.plot(x_new, y_smooth, linestyle='-', color='blue')
    ax.scatter(x_unique, y_unique, color='red', edgecolors='k', zorder=5)

    # Highlight the recommended setting
    if highlight_x is not None:
        highlight_y = y[x == highlight_x].values[0]
        ax.scatter([highlight_x], [highlight_y], color='green', edgecolors='k', zorder=10, s=100, label='Recommended')
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

# Plotting the updated charts without outliers
fig, axs = plt.subplots(6, 2, figsize=(20, 30))

# Add header with current date and time
fig.suptitle(f'Performance Metrics and Recommendations\nGenerated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16)

# Inference Metrics
plot_smooth_curve(axs[0, 0], inference_grouped['max_watt'], inference_grouped['tokens_per_sec'],
                  'Max Power vs. Tokens per Second (Inference)', 'Max Power (W)', 'Tokens per Second', highlight_x=optimal_inference_watt)

plot_smooth_curve(axs[1, 0], inference_grouped['max_watt'], inference_grouped['temperature'],
                  'Max Power vs. Temperature (Inference)', 'Max Power (W)', 'Temperature (C)', highlight_x=optimal_inference_watt)

plot_smooth_curve(axs[2, 0], inference_grouped['max_watt'], inference_grouped['gpu_utilization'],
                  'Max Power vs. GPU Utilization (Inference)', 'Max Power (W)', 'GPU Utilization (%)', highlight_x=optimal_inference_watt)

plot_smooth_curve(axs[3, 0], inference_grouped['max_watt'], inference_grouped['memory_utilization'],
                  'Max Power vs. Memory Utilization (Inference)', 'Max Power (W)', 'Memory Utilization (%)', highlight_x=optimal_inference_watt)

plot_smooth_curve(axs[4, 0], inference_grouped['max_watt'], inference_grouped['time_diff'],
                  'Max Power vs. Total Time (Inference)', 'Max Power (W)', 'Total Time (seconds)', highlight_x=optimal_inference_watt)

# Training Metrics
plot_smooth_curve(axs[0, 1], training_grouped['max_watt'], training_grouped['tokens_per_sec'],
                  'Max Power vs. Tokens per Second (Training)', 'Max Power (W)', 'Tokens per Second', highlight_x=optimal_training_watt)

plot_smooth_curve(axs[1, 1], training_grouped['max_watt'], training_grouped['temperature'],
                  'Max Power vs. Temperature (Training)', 'Max Power (W)', 'Temperature (C)', highlight_x=optimal_training_watt)

plot_smooth_curve(axs[2, 1], training_grouped['max_watt'], training_grouped['gpu_utilization'],
                  'Max Power vs. GPU Utilization (Training)', 'Max Power (W)', 'GPU Utilization (%)', highlight_x=optimal_training_watt)

plot_smooth_curve(axs[3, 1], training_grouped['max_watt'], training_grouped['memory_utilization'],
                  'Max Power vs. Memory Utilization (Training)', 'Max Power (W)', 'Memory Utilization (%)', highlight_x=optimal_training_watt)

plot_smooth_curve(axs[4, 1], training_grouped['max_watt'], training_grouped['time_diff'],
                  'Max Power vs. Total Time (Training)', 'Max Power (W)', 'Total Time (seconds)', highlight_x=optimal_training_watt)

# Summary with recommended settings
summary_text = (
    f"Recommended Settings:\n"
    f"Optimal Max Power for Training: {optimal_training_watt}W\n"
    f"Optimal Max Power for Inference: {optimal_inference_watt}W\n\n"
    "Recommendations are based on the lowest energy consumption (Watt-min) for each scenario.\n"
    "Energy consumption is calculated as the product of power draw and total time taken."
)

axs[5, 0].axis('off')
axs[5, 1].text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12, wrap=True)
axs[5, 1].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('report.png')
plt.show()
