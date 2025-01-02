#############################################################################################################
#                                                                                                           #
#   Author      : Stefano Giacomelli                                                                        #
#   Affiliation : PhD candidate at University of L'Aquila (Italy)                                           #
#   Department  : DISIM - Department of Information Engineering, Computer Science and Mathematics           #
#   Description : Core functions to install, import, and compute audio embeddings w. pre-trained models.    #
#   Last Update : 2024-12-07                                                                                #
#                                                                                                           #
#############################################################################################################
import os
import subprocess
import json
import numpy as np
import torch
import torchaudio
from utils import debug_log


#############################################################################################################
# Auxiliary functions
#############################################################################################################
def load_model_config(model_name, base_path="./backbones/", debug=False):
    """
    Load a model's configuration JSON file dynamically.

    :param model_name: 'category/model_name' of the model.
    :type model_name: str
    :param base_path: Base directory containing backbones JSON files.
    :type base_path: str
    :param debug: Enable logging to module-dedicated file.
    :type debug: bool
    :return: Model configuration dictionary.
    :rtype: dict
    """
    model_file_path = os.path.join(base_path, f"{model_name}.json")

    if not os.path.isfile(model_file_path):
        debug_log(f"Tried to import '{model_file_path}': NOT FOUND!", level="error", debug=debug)
        raise FileNotFoundError(f"Configuration file for '{model_name}' not found.")

    with open(model_file_path, "r") as f:
        try:
            data = json.load(f)
            debug_log(f"Loaded configuration for '{model_name}' from '{model_file_path}'.", level="success", debug=debug)
            return data
        except json.JSONDecodeError as e:
            debug_log(f"Parsing JSON file '{model_file_path}': {e}", level="error", debug=debug)
            raise


def execute_commands(commands, debug=False):
    """
    Execute a series of CLI shell commands.

    :param commands: List of commands to execute.
    :type commands: list
    :param debug: Enable logging to module-dedicated file.
    :type debug: bool
    :return: None
    """
    for command in commands:
        try:
            subprocess.run(command, shell=True, check=True)
            debug_log(f"Command executed: {command}", level="success", debug=debug)
        except subprocess.CalledProcessError as e:
            debug_log(f"Command failed: {command}", level="error", debug=debug)
            raise RuntimeError(f"Command failed: {e}")


#############################################################################################################
# Install and Import Model's packages and modules
#############################################################################################################
def package_install_and_import(model_name, debug=False):
    """
    Dynamically install a model's packages (handling pre-import contexts) and import required modules.

    :param model_name: 'category/model_name' of the model.
    :type model_name: str
    :param debug: Enable logging to module-dedicated file.
    :type debug: bool
    :return: Dictionary of user's Python Environment globals() w. required packages and imported modules.
    :rtype: dict
    """
    model_info = load_model_config(model_name, debug=debug)
    model_info = model_info[model_name.split('/')[-1]]
    imported_modules = {}   # Dictionary to store imported modules

    # Package installer
    if "install_cmd" in model_info:
        try:
            execute_commands(model_info["install_cmd"], debug=debug)
        except RuntimeError as e:
            debug_log(f"Installing package for '{model_name}': {e}", level="error", debug=debug)
            raise ImportError(f"Could not install package for '{model_name}'.")

    # Handle context switching for imports
    working_dir = os.getcwd()
    context_dir = model_info.get("context", None)

    try:
        if context_dir:
            debug_log(f"Switching imports context to: {context_dir}", level="info", debug=debug)
            os.chdir(context_dir)

        # Imports handling
        if "import" in model_info:
            for script in model_info["import"]:
                try:
                    exec(script, globals(), imported_modules)
                    debug_log(f"Executed import commands for '{model_name}': {script}", level="success", debug=debug)
                except Exception as e:
                    debug_log(f"Executing import commands for model '{model_name}': {e}", level="error", debug=debug)
                    raise ImportError(f"Could not execute import commands for '{model_name}': {e}")
    finally:
        if context_dir:
            os.chdir(working_dir)

    return imported_modules


#############################################################################################################
# Initialize Model
#############################################################################################################
def model_init(model_name, imported_modules, debug=False):
    """
    Dynamically initialize a model using its `model_init` configuration.

    :param model_name: 'category/model_name' of the model.
    :type model_name: str
    :param imported_modules: Dictionary of imported modules for required model.
    :type imported_modules: dict
    :param debug: Enable logging to module-dedicated file.
    :type debug: bool
    :return: The model instance (torch.nn.Module or object).
    :rtype: object
    """
    model_info = load_model_config(model_name, debug=debug)
    model_info = model_info[model_name.split('/')[-1]]

    if "model_init" not in model_info or not model_info["model_init"]:
        debug_log(f"No 'model_init' commands found for '{model_name}'.", level="error", debug=debug)
        raise KeyError(f"'{model_name}' is missing 'model_init' commands.")

    try:
        exec("\n".join(model_info["model_init"]), globals(), imported_modules)
        debug_log(f"Executed model initialization for '{model_name}': {' '.join(model_info['model_init'])}", level="success", debug=debug)
        model = imported_modules.get("model", None)

        # Model sanity check
        if model is None:
            debug_log(f"Model instance ('model') was not initialized for '{model_name}'.", level="error", debug=debug)
            raise ValueError(f"Model initialization failed for '{model_name}'.")

        return model

    except Exception as e:
        debug_log(f"Initializing '{model_name}': {e}", level="error", debug=debug)
        raise RuntimeError(f"Model initialization failed for '{model_name}': {str(e)}")


#############################################################################################################
# Compute Embeddings
#############################################################################################################
def pre_process_input(model_name, in_data, imported_modules, device="cpu", debug=False):
    """
    Dynamically pre-process input data for the target model.

    :param model_name: The name of the target model.
    :type model_name: str
    :param in_data: Input data (file path string, list, NumPy array, or Torch.Tensor).
    :type in_data: str | list | numpy.ndarray | torch.Tensor
    :param imported_modules: Dictionary of model-specific imported modules.
    :type imported_modules: dict
    :param device: Target device for audio processing ('cpu' or 'cuda:<id>').
    :type device: str
    :param debug: Enable logging to module-dedicated file.
    :type file_path: str
    :return: Pre-processed input tensor.
    :rtype: torch.Tensor 
    """
    model_info = load_model_config(model_name, debug=debug)
    model_info = model_info[model_name.split('/')[-1]]

    # File path handling
    if isinstance(in_data, str):
        if not os.path.isfile(in_data):
            debug_log(f"File path '{in_data}' does not exist.", level="error", debug=debug)
            raise FileNotFoundError(f"Audio file not found: {in_data}")
        
        # Load audio file
        waveform, source_sr = torchaudio.load(in_data)
        debug_log(f"Loading '{in_data}' with SR: {source_sr}Hz and shape {waveform.shape}.", level="success", debug=debug)

        # Mono Downmix
        if waveform.shape[0] > 1:
            n_chs = waveform.shape[0]
            for i in range(n_chs - 1):
                waveform[0] += waveform[i + 1]
            waveform = waveform[0] / n_chs
            debug_log(f"Downmixing audio --> NOW shape: {waveform.shape}.", level="success", debug=debug)

        # Resampling
        target_sr = model_info["sample_rate"]
        if source_sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)(waveform)
            debug_log(f"Resampling from {source_sr}Hz to {target_sr}Hz.", level="success", debug=debug)
        
        if waveform.shape[0] >= 1 and model_name.split('/')[-1] == 'vggish':
            waveform = waveform.squeeze()
        elif model_name.split('/')[-1].startswith('encodec'):
            waveform = waveform.unsqueeze(0)
        
        x = waveform

    # Numpy.NDArray or List handling     
    elif isinstance(in_data, (list, np.ndarray)):
        if model_name.split('/')[-1] == 'vggish':
            if type(in_data) == list:
                x = np.array(in_data)
            else:
                x = in_data
            debug_log(f"VGGish {type(in_data)} input, with shape: {x.shape}.", level="success", debug=debug)
        else:
            x = torch.tensor(in_data, dtype=torch.float32) if isinstance(in_data, list) else torch.from_numpy(in_data)
            debug_log(f"Converted from {type(in_data)} into torch.Tensor with shape: {x.shape}.", level="success", debug=debug)

    # torch.Tensor handling     
    elif isinstance(in_data, torch.Tensor):  # Torch Tensor
        x = in_data
        debug_log(f"Using provided torch.Tensor with shape: {x.shape}.", level="info", debug=debug)

    else:
        debug_log(f"Unsupported input type: {type(in_data)}.", level="error", debug=debug)
        raise TypeError("Input data must be a file path (str), list, numpy.ndarray, or torch.Tensor.")

    # Input shape sanity check
    if len(x.shape) != model_info["in_shape"]:
        debug_log(f"Input shape {x.shape} does not match '{model_name}' expected shape: {model_info['in_shape']}.", level="error", debug=debug)
        raise ValueError(f"Input shape mismatch for model '{model_name}'. Expected {model_info['in_shape']} dimensions.")

    # Pre-Processing (if defined)
    if "in_pre" in model_info and model_info["in_pre"]:
        try:
            namespace = {"x": x, "device": device, **globals(), **imported_modules}
            
            for command in model_info["in_pre"]:
                exec(command, namespace)
                debug_log(f"Pre-processing step executing command: {command}", level="success", debug=debug)
            
            x = namespace["x"]
            debug_log(f"Pre-processing completed for '{model_name}'. Final shape: {x.shape}", level="success", debug=debug)

        except Exception as e:
            debug_log(f"Pre-processing for '{model_name}' failed: {e}", level="error", debug=debug)
            raise RuntimeError(f"Preprocessing failed for '{model_name}': {str(e)}")

    return x.to(device)


def compute_embedding(model_name, imported_modules, model, x, debug=False):
    """
    Dynamically compute embeddings using the model's `embed_fwd` commands.

    :param model_name: The name of the model.
    :type model_name: str
    :param imported_modules: Dictionary of model-specific imported modules.
    :type imported_modules: dict
    :param model: Initialized model instance (for embedding).
    :type model: object or torch.nn.Module
    :param x: Input tensor or data to process.
    :type x: torch.Tensor or numpy.ndarray
    :param debug: Enable logging to module-dedicated file.
    :type debug: bool
    :return: Computed embeddings.
    :rtype: torch.Tensor or numpy.ndarray
    """
    model_info = load_model_config(model_name, debug=debug)
    model_info = model_info[model_name.split('/')[-1]]

    if "embed_fwd" not in model_info or len(model_info["embed_fwd"]) == 0:
        debug_log(f"No 'embed_fwd' commands found for '{model_name}'.", level="error", debug=debug)
        raise KeyError(f"'{model_name}' is missing 'embed_fwd' commands.")
    
    try:
        namespace = {**globals(), **imported_modules}
        
        # Overwrite existing variables w. desired parameters
        namespace["x"] = x
        namespace["model"] = model

        for command in model_info["embed_fwd"]:
            debug_log(f"Executing embedding command: {command}", level="info", debug=debug)
            exec(command, namespace)

        # Retrieve the resulting embedding
        y = namespace.get("y", None)

        if y is None:
            debug_log(f"Embedding computation failed for '{model_name}': y is None.", level="error", debug=debug)
            raise ValueError(f"Embedding computation for '{model_name}' returned None.")

        debug_log(f"Embedding computation for '{model_name}' completed. Result shape: {y.shape if hasattr(y, 'shape') else 'Unknown'}", level="success", debug=debug)
        return y

    except Exception as e:
        debug_log(f"Embedding computation for '{model_name}' failed: {e}", level="error", debug=debug)
        raise RuntimeError(f"Embedding computation failed for '{model_name}': {str(e)}")
