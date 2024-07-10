import pynvml
# import time
import logging
import subprocess

logger = logging.getLogger(__name__)

def initialize_nvml():
    pynvml.nvmlInit()

def shutdown_nvml():
    pynvml.nvmlShutdown()

def get_gpu_metrics():
    device_count = pynvml.nvmlDeviceGetCount()
    metrics = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

        # Additional metrics
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        memory_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
        # encoder_utilization = pynvml.nvmlDeviceGetEncoderUtilization(handle)[0]
        # decoder_utilization = pynvml.nvmlDeviceGetDecoderUtilization(handle)[0]
        # graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        # sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        # memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        # video_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_VIDEO)
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # Convert to watts
        # pci_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / (1024 ** 2)  # Convert to MB/s
        # pci_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / (1024 ** 2)  # Convert to MB/s

        metrics.append({
            'temperature': temperature,
            'fan_speed': fan_speed,
            'memory_used': memory_info.used / (1024 ** 2),  # Convert to MB
            'power_draw': power_draw,
            'gpu_utilization': gpu_utilization,
            'memory_utilization': memory_utilization,
            # 'encoder_utilization': encoder_utilization,
            # 'decoder_utilization': decoder_utilization,
            # 'graphics_clock': graphics_clock,
            # 'sm_clock': sm_clock,
            # 'memory_clock': memory_clock,
            # 'video_clock': video_clock,
            'power_limit': power_limit,
            # 'pci_tx': pci_tx,
            # 'pci_rx': pci_rx
        })
    return metrics

def set_gpu_power_limit(gpu_ids, power_limit):
    """Set the power limit for specified GPUs using nvidia-smi."""
    for gpu_id in gpu_ids:
        command = f"sudo nvidia-smi -i {gpu_id} -pl {power_limit}"
        try:
            subprocess.run(command, shell=True, check=True)
            logger.info(f"Set GPU {gpu_id} power limit to {power_limit} watts")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set power limit for GPU {gpu_id}: {e}")

def start_gpu_metrics_collector(interval=10, log_file='gpu_metrics_log.csv'):
    command = f"python gpu_metrics_collector.py --interval {interval} --log_file {log_file}"
    process = subprocess.Popen(command, shell=True)
    return process

def stop_gpu_metrics_collector(process):
    process.terminate()
    process.wait()
    logger.info("Terminated GPU metrics collector")
