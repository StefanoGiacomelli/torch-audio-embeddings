#############################################################################################################
#                                                                                                           #
#   Author      : Stefano Giacomelli                                                                        #
#   Affiliation : PhD candidate at University of L'Aquila (Italy)                                           #
#   Department  : DISIM - Department of Information Engineering, Computer Science and Mathematics           #
#   Description : Performance Benchmark functions for audio embeddings models.                              #
#   Last Update : 2024-12-15                                                                                #
#                                                                                                           #
#############################################################################################################
import os
import sys
import shutil
import platform
import glob
import subprocess
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import tracemalloc
from codecarbon import EmissionsTracker
import time
from tqdm import tqdm
import json
import csv
import numpy as np
from scipy.stats import iqr, skew, kurtosis
import torch

from audio_embeddings import load_model_config, package_install_and_import, model_init, pre_process_input, compute_embedding
from utils import set_seeds, test_input, HiddenPrints, debug_log


# Module variables
cpu_usage_samples = Queue()
cpu_monitoring = threading.Event()
gpu_usage_samples = Queue()
gpu_monitoring = threading.Event()



#############################################################################################################
# Static Profiling functions
#############################################################################################################
def your_gpu(verbose=True):
    """
    Verify NVIDIA GPU(s) availability and return corresponding PyTorch device string, optionally report device inspection into a logging file.

    :param verbose: Whether to include detailed logging (default: True).
    :type verbose: bool
    :return: PyTorch device string ('cuda:ID' or 'cpu') and details about GPUs.
    :rtype: tuple(str, dict)
    """

    def bytes_to_gb(bytes_val):
        """Convert Bytes to GigaBytes."""
        return bytes_val * 1e-9

    def fetch_gpu_info(gpu_id):
        """Retrieve information for a specific (<gpu_id>) GP-GPU"""
        gpu_info = {"id": gpu_id}
        device = f"cuda:{gpu_id}"
        gpu_info["name"] = torch.cuda.get_device_name(gpu_id)
        
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            gpu_info["total_memory_gb"] = bytes_to_gb(total_mem)
            gpu_info["free_memory_gb"] = bytes_to_gb(free_mem)
            debug_log(f"Retrieving memory info for GPU {gpu_id}: {gpu_info}.", level="success", verbose=verbose)
        except Exception as e:
            gpu_info["total_memory_gb"] = None
            gpu_info["free_memory_gb"] = None
            debug_log(f"Failed to retrieve memory info for GPU {gpu_id}: {e}.", level="error", verbose=verbose)
            debug_log(f"Retrieved partial memory info for GPU {gpu_id}: {gpu_info}.", level="warning", verbose=verbose)
        
        return gpu_info

    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            debug_log(f"{gpu_count}x NVIDIA GPU(s) Detected", level="success", verbose=verbose)
            
            # Collect GPU details (in parallel)
            gpu_details = {"count": gpu_count, "devices": []}
            with ThreadPoolExecutor() as executor:
                gpu_details["devices"] = list(executor.map(fetch_gpu_info, range(gpu_count)))

            # Log `nvidia-smi` CLI output
            if shutil.which("nvidia-smi"):
                debug_log(f"-------------------------------- Nvidia-SMI Report --------------------------------", level="info", verbose=verbose)
                try:
                    smi_output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
                    debug_log("NVIDIA-SMI output:\n" + smi_output, level="success", verbose=verbose)
                except Exception as e:
                    debug_log(f"Failed to retrieve NVIDIA-SMI output: {e}", level="error", verbose=verbose)
            else:
                debug_log("'nvidia-smi' command not found. Ensure NVIDIA drivers are installed and available in PATH.", level="warning", verbose=verbose)

            # Log `nvcc` CLI output
            if shutil.which("nvcc"):
                try:
                    nvcc_output = subprocess.check_output(["nvcc", "--version"], encoding="utf-8")
                    debug_log("NVCC output:\n" + nvcc_output, level="success", verbose=verbose)
                except Exception as e:
                    debug_log(f"Failed to retrieve NVCC version: {e}", level="error", verbose=verbose)
            else:
                debug_log("NVCC not found. Ensure CUDA toolkit is installed and available in PATH.", level="warning", verbose=verbose)
            
            debug_log(f'{torch.cuda.list_gpu_processes(f"cuda:{torch.cuda.current_device()}")}', level="debug", verbose=verbose)
            debug_log(f'{torch.cuda.memory_summary(device=f"cuda:{torch.cuda.current_device()}", abbreviated=False)}', level="info", verbose=verbose)
            debug_log(f"PyTorch Version: {torch.__version__}", level="success", verbose=verbose)
            
            return f"cuda:{torch.cuda.current_device()}", gpu_details
        else:
            debug_log("No NVIDIA GPU(s) detected. Using CPU.", level="warning", verbose=verbose)
            debug_log(f"PyTorch Version: {torch.__version__}", level="success", verbose=verbose)
            
            return "cpu", {"count": 0, "devices": []}
    except Exception as e:
        debug_log(f"An unexpected error occurred while detecting devices: {e}", level="critical", verbose=verbose)
        
        return "cpu", {"count": 0, "devices": []}


def your_hardware(verbose=True):
    """
    Inspect and log hardware details with cross-platform support.

    :param logger: Logger for reporting.
    :type logger: Logger
    :param verbose: Whether to include detailed logging (default: True).
    :type verbose: bool
    :return: Summary of hardware details.
    :rtype: dict
    """
    hardware_info = {}

    # CPU Info
    def get_cpu_info():
        """Retrieve CPU information based on the operating system."""
        if platform.system() == "Linux":
            try:
                output = subprocess.check_output(["cat", "/proc/cpuinfo"], encoding="utf-8")
                debug_log("CPU Info:\n" + output, level="success", verbose=verbose)
                
                return parse_cpu_info_linux(output)
            except Exception as e:
                debug_log(f"Failed to retrieve CPU info on Linux: {e}", level="error", verbose=verbose)
        elif platform.system() == "Windows":
            try:
                cpu_count = psutil.cpu_count(logical=True)
                cpu_freq = psutil.cpu_freq()
                
                return {"cpu_count": cpu_count, "max_frequency_mhz": cpu_freq.max, "min_frequency_mhz": cpu_freq.min}
            except Exception as e:
                debug_log(f"Failed to retrieve CPU info on Windows: {e}", level="error", verbose=verbose)
        else:
            debug_log("CPU info retrieval not available for this platform.", level="warning", verbose=verbose)

        return {}

    def parse_cpu_info_linux(output):
        """Parse /proc/cpuinfo output into a structured format."""
        cpu_details = []
        cpu_info = {}
        for line in output.splitlines():
            if not line.strip():
                if cpu_info:
                    cpu_details.append(cpu_info)
                    cpu_info = {}
                continue
            key, _, value = line.partition(":")
            cpu_info[key.strip()] = value.strip()
        
        return cpu_details

    hardware_info["cpu"] = get_cpu_info()

    # Disk Info
    def get_disk_info():
        """Retrieve disk information in a cross-platform way."""
        if platform.system() in ["Linux", "Darwin"]:
            if shutil.which("df"):
                try:
                    output = subprocess.check_output(["df", "-h"], encoding="utf-8")
                    debug_log(f"Disk Info Output:\n{output}", level="success", verbose=verbose)

                    return parse_disks_unix(output)
                except Exception as e:
                    debug_log(f"Failed to retrieve disk info using 'df': {e}", level="error", verbose=verbose)
            else:
                debug_log(f"'df' command not found. Skipping disk info retrieval.", level="warning", verbose=verbose)
        elif platform.system() == "Windows":
            try:
                disks = []
                for partition in psutil.disk_partitions():
                    usage = psutil.disk_usage(partition.mountpoint)
                    disks.append({"Device": partition.device,
                                  "Mountpoint": partition.mountpoint,
                                  "FileSystem": partition.fstype,
                                  "Total": f"{usage.total / 1e9:.2f} GB",
                                  "Used": f"{usage.used / 1e9:.2f} GB",
                                  "Free": f"{usage.free / 1e9:.2f} GB",
                                  "Use%": f"{usage.percent:.1f}%"})
                
                return disks
            except Exception as e:
                debug_log(f"Failed to retrieve disk info on Windows: {e}", level="error", verbose=verbose)
        else:
            debug_log("Disk info retrieval is not implemented for this platform.", level="critical", verbose=verbose)
        
        return []

    def parse_disks_unix(output):
        """Parse 'df -h' output into a structured format."""
        lines = output.splitlines()
        headers = lines[0].split()
        disks = [dict(zip(headers, line.split())) for line in lines[1:] if line.strip()]
        
        return disks

    hardware_info["disks"] = get_disk_info()

    # RAM Info
    def get_ram_info():
        """Retrieve RAM information based on the operating system."""
        if platform.system() == "Linux":
            try:
                output = subprocess.check_output(["cat", "/proc/meminfo"], encoding="utf-8")
                debug_log("RAM Info:\n" + output, level="success", verbose=verbose)

                return parse_meminfo_linux(output)
            except Exception as e:
                debug_log(f"Failed to retrieve RAM info on Linux: {e}", level="error", verbose=verbose)
        elif platform.system() == "Windows":
            try:
                virtual_mem = psutil.virtual_memory()
                
                return {"total_memory_gb": f"{virtual_mem.total / 1e9:.2f} GB",
                        "available_memory_gb": f"{virtual_mem.available / 1e9:.2f} GB",
                        "used_memory_gb": f"{virtual_mem.used / 1e9:.2f} GB",
                        "percent_used": f"{virtual_mem.percent:.1f}%"}
            except Exception as e:
                debug_log(f"Failed to retrieve RAM info on Windows: {e}", level="error", verbose=verbose)
        else:
            debug_log("RAM info retrieval is not implemented for this platform.", level="critical", verbose=verbose)
        
        return {}

    def parse_meminfo_linux(output):
        """Parse /proc/meminfo output into a structured format."""
        memory = {}
        for line in output.splitlines():
            key, _, value = line.partition(":")
            memory[key.strip()] = value.strip()
        return memory

    hardware_info["ram"] = get_ram_info()

    # Summary
    debug_log(f"Hardware Summary: {hardware_info}", level="success", verbose=verbose)

    return hardware_info


#############################################################################################################
# Units Monitoring Functions
#############################################################################################################
def monitor_cpu_usage():
    """
    Continuously monitor CPU resources usage and append utilization samples to a thread-safe queue.
    
    :global cpu_usage_samples: A thread-safe queue to store CPU usage percentages.
    :type cpu_usage_samples: Queue
    :global cpu_monitoring: A thread-safe event to control the monitoring loop.
    :type cpu_monitoring: threading.Event
    """
    if not cpu_monitoring.is_set():
        cpu_monitoring.set()

    while cpu_monitoring.is_set():
        cpu_usage_samples.put(psutil.cpu_percent(interval=0.1))


def monitor_gpu_usage():
    """
    Continuously monitor GPU utilization and append samples to a thread-safe queue.
    
    :global gpu_usage_samples: A thread-safe queue to store GPU usage percentages.
    :type gpu_usage_samples: Queue
    :global gpu_monitoring: A thread-safe event to control the monitoring loop.
    :type gpu_monitoring: threading.Event
    """
    if not gpu_monitoring.is_set():
        gpu_monitoring.set()

    while gpu_monitoring.is_set():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
            if result.returncode == 0:
                utilization = int(result.stdout.strip())
                gpu_usage_samples.put(utilization)
        except Exception as e:
            gpu_usage_samples.put(0)  # Assume 0% usage if query fails

        time.sleep(0.1)


#############################################################################################################
# Models Profiling Functions
#############################################################################################################
def cpu_model_profiler(model_name, iterations=100, verbose=True, debug=False):
    """
    CPU-accelerated Performance Profiler for audio embeddings models.

    :params model_name: str, name of the model to profile (backbones category included as path).
    :type model_name: str
    :params iterations: int, number of iterations to run for each benchmark. Defaults to 100.
    :type iterations: int
    :params verbose: bool, enable verbose mode.
    :type verbose: bool
    :params debug: bool, enable Audio embedding models logging report.
    :type debug: bool
    """
    category = model_name.split('/')[0]
    model_name = model_name.split('/')[-1]
    model_benchmark = {}
    cwd = os.getcwd()

    with HiddenPrints():
        # Install and import modules
        imported_modules = package_install_and_import(f'{category}/{model_name}', debug=debug)

        # Init model
        model_info = load_model_config(f'{category}/{model_name}', debug=debug)
        model_info = model_info[model_name.split('/')[-1]]
        model = model_init(f'{category}/{model_name}', imported_modules, debug=debug)

        # Create the results directory
        os.makedirs(f'{cwd}/{model_name}_profiling_results', exist_ok=True)


    ######################################### CPU Computations Benchmark ######################################
    # Inference/Evaluation mode
    with torch.inference_mode():
        if model_name in ['panns_Cnn14', 'panns_ResNet38', 'panns_Wavegram_Logmel_Cnn14']:
            model.model.to('cpu')
            model.model.eval()
        else:
            model.to('cpu')
            model.eval()
        debug_log('Model moved to CPU and set to eval mode', level='success', verbose=verbose)
        

        # MINIMUM INPUT SIZE PROFILING ------------------------------------------------------------------------
        max_dur = int(model_info['sample_rate'] * 10)
        low, high = 1, max_dur
        total_iterations = high - low + 1
        
        # Binary search (for boundaries limitation)
        with tqdm(total=total_iterations, desc="Minimum Input Size Binary Search") as pbar:
            i = 0
            while low < high and (high - low) > 1:
                mid = (high + low) // 2
                try:
                    set_seeds(seed=42)
                    x = test_input(model_name, batch_size=1, duration=mid, sample_rate=model_info['sample_rate'], device='cpu')
                    if model_name in ['vggish', 'yamnet']:
                        x = pre_process_input(f'{category}/{model_name}', x.float(), imported_modules, 'cpu', debug=debug)
                    y = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)

                    if y is not None:
                        high = mid - 1
                    i += 1
                except:
                    low = mid
                    i += 1
    
                completed_iterations = total_iterations - (high - low + 1)
                pbar.n = completed_iterations
                pbar.refresh()
        
        # Results processing
        duration = high
        model_benchmark['min_in_size'] = {'samps': int(duration),
                                          'sec': duration / model_info['sample_rate'],
                                          'binary_search_iterations': i}
        debug_log(f"Minimum Input Size processed by '{model_name}': {model_benchmark['min_in_size']}", level='success', verbose=verbose)
        
        
        # PROFILING EXPERIMENT INPUT --------------------------------------------------------------------------
        set_seeds(seed=42)
        debug_log('Seeds set to 42 for experiment repeatability', level='success', verbose=verbose)
        
        x = test_input(model_name, batch_size=1, duration=10., sample_rate=model_info['sample_rate'], device='cpu')
        debug_log(f'Synthesizing input (10sec. at {model_info['sample_rate']}Hz): {type(x)} w. shape {x.shape}', level='success', verbose=verbose)
        if model_name == 'yamnet':
            x = pre_process_input(f'{category}/{model_name}', x.float(), imported_modules, 'cpu', debug=debug)
        elif model_name == 'vggish':
            x = pre_process_input(f'{category}/{model_name}', x, imported_modules, 'cpu', debug=debug)
        debug_log(f'Pre-processing input: now {type(x)} w. shape {x.shape}', level='success', verbose=verbose)


        # OVERALL (sleep included) TIME PROFILING -------------------------------------------------------------
        cpu_overall_times = []
        for i in tqdm(range(iterations), desc='CPU Overall Times benchmarking'):
            start = time.perf_counter()
            _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
            cpu_overall_times.append(time.perf_counter() - start)

        # Results processing
        model_benchmark['cpu_overall_time'] = {"max": float(np.max(cpu_overall_times)),
                                               "min": float(np.min(cpu_overall_times)),
                                               "mean": float(np.mean(cpu_overall_times)),
                                               "std_dev": float(np.std(cpu_overall_times, ddof=1)),
                                               "median": float(np.median(cpu_overall_times)),
                                               "percentiles": {f"{p}th_perc": float(np.percentile(cpu_overall_times, p)) for p in [25, 33, 66, 75]},
                                               "iqr": float(iqr(cpu_overall_times)),
                                               "skewness": float(skew(cpu_overall_times)),
                                               "kurtosis": float(kurtosis(cpu_overall_times))}
        debug_log(f'CPU Overall Times: {model_benchmark["cpu_overall_time"]}', level='success', verbose=verbose)
        np.savez_compressed(f'{cwd}/{model_name}_profiling_results/cpu_overall_times.npz', 
                            values=np.array(cpu_overall_times), 
                            features=np.array(list(model_benchmark['cpu_overall_time'].items())))
        debug_log(f"Saving CPU Overall Times 'values' and 'features' in: {cwd}/{model_name}_profiling_results/", level='success', verbose=verbose)


        # PROCESS (System + User) TIME PROFILING --------------------------------------------------------------
        cpu_process_times = []
        for i in tqdm(range(iterations), desc='CPU Process Times benchmarking'):
            start = time.process_time()
            _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
            cpu_process_times.append(time.process_time() - start)

        # Results processing
        model_benchmark['cpu_process_time'] = {"max": float(np.max(cpu_process_times)),
                                            "min": float(np.min(cpu_process_times)),
                                            "mean": float(np.mean(cpu_process_times)),
                                            "std_dev": float(np.std(cpu_process_times, ddof=1)),
                                            "median": float(np.median(cpu_process_times)),
                                            "percentiles": {f"{p}th_perc": float(np.percentile(cpu_process_times, p)) for p in [25, 33, 66, 75]},
                                            "iqr": float(iqr(cpu_process_times)),
                                            "skewness": float(skew(cpu_process_times)),
                                            "kurtosis": float(kurtosis(cpu_process_times))}
        debug_log(f'CPU Process Times: {model_benchmark["cpu_process_time"]}', level='success', verbose=verbose)
        
        np.savez_compressed(f'{cwd}/{model_name}_profiling_results/cpu_process_times.npz', 
                            values=np.array(cpu_process_times), 
                            features=np.array(list(model_benchmark['cpu_process_time'].items())))
        debug_log(f"Saving CPU Process Times 'values' and 'features' in: {cwd}/{model_name}_profiling_results/", level='success', verbose=verbose)


        # MEMORY (RAM) USAGE PROFILING ------------------------------------------------------------------------
        tracemalloc.start()
        for i in tqdm(range(iterations), desc='RAM Usage benchmarking'):
            _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Results processing
        model_benchmark['ram_usage'] = {"peak":  peak}
        debug_log(f'RAM usage: {model_benchmark["ram_usage"]}', level='success', verbose=verbose)


        # CPU RESOURCES USAGE (%) PROFILING -------------------------------------------------------------------
        try:
            cpu_monitoring.set()
            monitor_thread = threading.Thread(target=monitor_cpu_usage)
            monitor_thread.start()
            for i in tqdm(range(iterations), desc='CPU Resources usage benchmarking'):
                _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
        except Exception as e:
            debug_log(f"During CPU usage monitoring: {e}", level='error', verbose=verbose)
        finally:
            cpu_monitoring.clear()
            monitor_thread.join()

        # Results processing
        cpu_perc_samples = []
        while not cpu_usage_samples.empty():
            cpu_perc_samples.append(cpu_usage_samples.get())
        
        avg_cpu_usage = sum(cpu_perc_samples) / len(cpu_perc_samples) if cpu_perc_samples else 0
        peak_cpu_usage = max(cpu_perc_samples) if cpu_perc_samples else 0
        model_benchmark['cpu_usage'] = {"avg": avg_cpu_usage,
                                        "peak": peak_cpu_usage}
        debug_log(f'CPU Resources usage: {model_benchmark["cpu_usage"]}', level='success', verbose=verbose)


        # ENERGY CONSUMPTION & CO2 EMISSIONS ESTIMATION -------------------------------------------------------
        # Custom logger
        import logging
        logger = logging.getLogger(f'{model_name}_energy_emissions')
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])
        formatter = logging.Formatter("%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s")
        file_handler = logging.FileHandler(f'{cwd}/{model_name}_profiling_results/{model_name}_energy_emissions' + '.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(formatter)
        consoleHandler.setLevel(logging.WARNING)
        logger.addHandler(consoleHandler)
        
        energy_profiler = EmissionsTracker(project_name=f'{model_name}_energy_emissions',
                                           tracking_mode='machine',
                                           save_to_file=True,
                                           save_to_logger=True,
                                           output_dir=f"{cwd}/{model_name}_profiling_results",
                                           output_file=f"{cwd}/{model_name}_profiling_results/energy_emissions.csv",
                                           logging_logger=logger,
                                           measure_power_secs=0.1)
        
        energy_profiler.start()
        for i in tqdm(range(iterations), desc='Energy/CO2 Emissions benchmarking'):
            energy_profiler.start_task(f'Run-{i+1}')
            _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
            energy_profiler.stop_task(f'Run-{i+1}')
        energy_profiler.stop()

        # Results processing
        csv_pattern = f"{cwd}/{model_name}_profiling_results/emissions_base_*.csv"
        csv_files = glob.glob(csv_pattern)
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found matching the pattern '{csv_pattern}'.")
        csv_file = csv_files[0]

        emissions_rate_values = []
        cpu_energy_values = []
        ram_energy_values = []
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)  # Read CSV with header as a dictionary
            for row in reader:
                emissions_rate_values.append(float(row['emissions_rate']))
                cpu_energy_values.append(float(row['cpu_energy']))
                ram_energy_values.append(float(row['ram_energy']))

        model_benchmark['energy_emissions'] = {'avg_emission_rate' : sum(emissions_rate_values) / len(emissions_rate_values) if emissions_rate_values else 0,
                                               'avg_cpu_energy' : sum(cpu_energy_values) / len(cpu_energy_values) if cpu_energy_values else 0,
                                               'avg_ram_energy' : sum(ram_energy_values) / len(ram_energy_values) if ram_energy_values else 0}
        debug_log(f'Energy Consumption & CO2 Emissions: {model_benchmark["energy_emissions"]}', level='success', verbose=verbose)


        # OUTPUT FORMATTING -----------------------------------------------------------------------------------
        with open(f"{cwd}/{model_name}_profiling_results/model_stats_cpu.json", "w") as json_file:
            json.dump(model_benchmark, json_file, indent=4)
        
        debug_log(f"CPU Profiling: result logs and assets saved in '{cwd}/{model_name}_profiling_results/'", level='success', verbose=verbose)


def gpu_model_profiler(model_name, iterations=100, batch_sizes=[1, 2, 5, 10], verbose=True, debug=False):
    """
    GPU-accelerated Performance Profiler for audio embeddings models.

    :params model_name: str, name of the model to profile (backbones category included as path).
    :type model_name: str
    :params iterations: int, number of iterations to run for each benchmark. Defaults to 100.
    :type iterations: int
    :params batch_sizes: list, list of batch sizes (in samples) for latency profiling. Defaults to [1, 2, 5, 10].
    :type batch_sizes: list[int]
    :params verbose: bool, enable verbose mode.
    :type verbose: bool
    :params debug: bool, enable Audio embedding models logging report.
    :type debug: bool
    """
    category = model_name.split('/')[0]
    model_name = model_name.split('/')[-1]
    if model_name == 'yamnet':
        raise NotImplementedError("YAMNet model is not supported for batches-GPU profiling.")
    model_benchmark = {}
    cwd = os.getcwd()
    device = f'cuda:{torch.cuda.current_device()}'

    with HiddenPrints():
        # Install and import modules
        imported_modules = package_install_and_import(f'{category}/{model_name}', debug=debug)

        # Init model
        model_info = load_model_config(f'{category}/{model_name}', debug=debug)
        model_info = model_info[model_name.split('/')[-1]]
        model = model_init(f'{category}/{model_name}', imported_modules, debug=debug)

        # Create the results directory
        os.makedirs(f'{cwd}/{model_name}_profiling_results', exist_ok=True)


    ######################################### GPU Computations Benchmark ######################################
    # Inference/Evaluation mode
    with torch.inference_mode():
        if model_name in ['panns_Cnn14', 'panns_ResNet38', 'panns_Wavegram_Logmel_Cnn14']:
            model.model.to(device)
            model.model.eval()
        else:
            model.to(device)
            model.eval()
        debug_log(f'Model moved to GPU ({device}) and set to eval mode', level='success', verbose=verbose)

        for batch_size in batch_sizes:      
            set_seeds(seed=42)
            debug_log('Seeds set to 42 for experiment repeatability', level='success', verbose=verbose)
            x = test_input(model_name, batch_size=batch_size, duration=10., sample_rate=model_info['sample_rate'], device=device)

            print('GPU-CUDA profiling for Input batch size:', batch_size)
            debug_log(f'GPU-CUDA profiling for Input batch size: {batch_size}', level='info', verbose=verbose)

            # CUDA DEVICE TIMING PROFILING --------------------------------------------------------------------
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            gpu_timings = []
            
            for _ in tqdm(range(iterations), desc=f"GPU Timing benchmarking"):
                start.record()
                # YAMNET exception handling w. reiterated inference and embeddings concatenation
                _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
                end.record()
                gpu_timings.append(start.elapsed_time(end) / 1000.)

            # Results processing
            model_benchmark[f'gpu_times_{batch_size}'] = {"max": float(np.max(gpu_timings)),
                                                          "min": float(np.min(gpu_timings)),
                                                          "mean": float(np.mean(gpu_timings)),
                                                          "std_dev": float(np.std(gpu_timings, ddof=1)),
                                                          "median": float(np.median(gpu_timings)),
                                                          "percentiles": {f"{p}th_perc": float(np.percentile(gpu_timings, p)) for p in [25, 33, 66, 75]},
                                                          "iqr": float(iqr(gpu_timings)),
                                                          "skewness": float(skew(gpu_timings)),
                                                          "kurtosis": float(kurtosis(gpu_timings))}
            debug_log(f'GPU Times: {model_benchmark[f'gpu_times_{batch_size}']}', level='success', verbose=verbose)
            np.savez_compressed(f'{cwd}/{model_name}_profiling_results/gpu_times_batch={batch_size}.npz', 
                                values=np.array(gpu_timings), 
                                features=np.array(list(model_benchmark[f'gpu_times_{batch_size}'].items())))
            debug_log(f"Saving GPU Times (batch={batch_size}) 'values' and 'features' in: {cwd}/{model_name}_profiling_results/", level='success', verbose=verbose)


            # CPU/CUDA E2E TIME -------------------------------------------------------------------------------
            e2e_times = []

            for _ in tqdm(range(iterations), desc=f"E2E Times benchmarking"):
                start = time.perf_counter()
                # YAMNET exception handling w. reiterated inference and embeddings concatenation
                _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
                torch.cuda.synchronize(device=device)
                e2e_times.append(time.perf_counter() - start)

            # Results processing
            model_benchmark[f'e2e_times_{batch_size}'] = {"max": float(np.max(e2e_times)),
                                                          "min": float(np.min(e2e_times)),
                                                          "mean": float(np.mean(e2e_times)),
                                                          "std_dev": float(np.std(e2e_times, ddof=1)),
                                                          "median": float(np.median(e2e_times)),
                                                          "percentiles": {f"{p}th_perc": float(np.percentile(e2e_times, p)) for p in [25, 33, 66, 75]},
                                                          "iqr": float(iqr(e2e_times)),
                                                          "skewness": float(skew(e2e_times)),
                                                          "kurtosis": float(kurtosis(e2e_times))}
            debug_log(f'GPU Times: {model_benchmark[f'e2e_times_{batch_size}']}', level='success', verbose=verbose)
            np.savez_compressed(f'{cwd}/{model_name}_profiling_results/e2e_times_batch={batch_size}.npz', 
                                values=np.array(e2e_times), 
                                features=np.array(list(model_benchmark[f'gpu_times_{e2e_times}'].items())))
            debug_log(f"Saving E2E Times (batch={batch_size}) 'values' and 'features' in: {cwd}/{model_name}_profiling_results/", level='success', verbose=verbose)


            # GPU MEMORY USAGE PROFILING ----------------------------------------------------------------------
            torch.cuda.reset_peak_memory_stats()
            for i in tqdm(range(iterations), desc='GPU Memory Usage benchmarking'):
                _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
                torch.cuda.synchronize(device=device)
            peak = torch.cuda.max_memory_allocated()
            
            # Results processing
            model_benchmark[f'gpu_memory_usage_{batch_size}'] = {"peak":  peak}
            debug_log(f"GPU memory usage: {model_benchmark[f'gpu_memory_usage_{batch_size}']}", level='success', verbose=verbose)


            # GPU RESOURCES USAGE (%) PROFILING ----------------------------------------------------------------
            try:
                gpu_monitoring.set()
                monitor_thread = threading.Thread(target=monitor_gpu_usage)
                monitor_thread.start()
                for _ in tqdm(range(iterations), desc="GPU Resources usage benchmarking"):
                    _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
            except Exception as e:
                debug_log(f"During GPU usage monitoring: {e}", level="error", verbose=verbose)
            finally:
                gpu_monitoring.clear()
                monitor_thread.join()
            
            # Results processing
            gpu_perc_samples = []
            while not gpu_usage_samples.empty():
                gpu_perc_samples.append(gpu_usage_samples.get())
            
            avg_gpu_usage = sum(gpu_perc_samples) / len(gpu_perc_samples) if gpu_perc_samples else 0
            peak_gpu_usage = max(gpu_perc_samples) if gpu_perc_samples else 0
            model_benchmark[f'gpu_usage_{batch_size}'] = {"avg": avg_gpu_usage,
                                                          "peak": peak_gpu_usage}
            debug_log(f"GPU Resources usage: {model_benchmark[f'gpu_usage_{batch_size}']}", level="success", verbose=verbose)


            # ENERGY CONSUMPTION & CO2 EMISSIONS ESTIMATION -------------------------------------------------------
            # Custom logger
            import logging
            logger = logging.getLogger(f'{model_name}_energy_emissions_{batch_size}')
            while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])
            formatter = logging.Formatter("%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s")
            file_handler = logging.FileHandler(f'{cwd}/{model_name}_profiling_results/{model_name}_energy_emissions_{batch_size}' + '.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setFormatter(formatter)
            consoleHandler.setLevel(logging.WARNING)
            logger.addHandler(consoleHandler)
            
            energy_profiler = EmissionsTracker(project_name=f'{model_name}_energy_emissions_{batch_size}',
                                               tracking_mode='machine',
                                               save_to_file=True,
                                               save_to_logger=True,
                                               output_dir=f"{cwd}/{model_name}_profiling_results",
                                               output_file=f"{cwd}/{model_name}_profiling_results/energy_emissions_{batch_size}.csv",
                                               logging_logger=logger,
                                               measure_power_secs=0.1)
            
            energy_profiler.start()
            for i in tqdm(range(iterations), desc='Energy/CO2 Emissions benchmarking'):
                energy_profiler.start_task(f'Run-{i+1}')
                _ = compute_embedding(f'{category}/{model_name}', imported_modules, model, x.float(), debug=debug)
                energy_profiler.stop_task(f'Run-{i+1}')
            energy_profiler.stop()

            # Results processing
            csv_pattern = f"{cwd}/{model_name}_profiling_results/emissions_base_*.csv"
            csv_files = glob.glob(csv_pattern)
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found matching the pattern '{csv_pattern}'.")
            csv_file = csv_files[0]

            emissions_rate_values = []
            cpu_energy_values = []
            ram_energy_values = []
            with open(csv_file, 'r') as file:
                reader = csv.DictReader(file)  # Read CSV with header as a dictionary
                for row in reader:
                    emissions_rate_values.append(float(row['emissions_rate']))
                    cpu_energy_values.append(float(row['cpu_energy']))
                    ram_energy_values.append(float(row['ram_energy']))

            model_benchmark[f'energy_emissions_{batch_size}'] = {'avg_emission_rate' : sum(emissions_rate_values) / len(emissions_rate_values) if emissions_rate_values else 0,
                                                                 'avg_cpu_energy' : sum(cpu_energy_values) / len(cpu_energy_values) if cpu_energy_values else 0,
                                                                 'avg_ram_energy' : sum(ram_energy_values) / len(ram_energy_values) if ram_energy_values else 0}
            debug_log(f'Energy Consumption & CO2 Emissions: {model_benchmark[f"energy_emissions_{batch_size}"]}', level='success', verbose=verbose)

            print('--------------------------------------------------------------------------------------------') 
            print('\n' * 4)

        # OUTPUT FORMATTING -----------------------------------------------------------------------------------
        with open(f"{cwd}/{model_name}_profiling_results/model_stats_gpu.json", "w") as json_file:
            json.dump(model_benchmark, json_file, indent=4)
        
        debug_log(f"GPU Profiling: result logs and assets saved in '{cwd}/{model_name}_profiling_results/'", level='success', verbose=verbose)
