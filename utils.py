#############################################################################################################
#                                                                                                           #
#   Author      : Stefano Giacomelli                                                                        #
#   Affiliation : PhD candidate at University of L'Aquila (Italy)                                           #
#   Department  : DISIM - Department of Information Engineering, Computer Science and Mathematics           #
#   Description : Utility functions for the Torch-Audio-Embeddings package                                  #
#   Last Update : 2024-12-10                                                                                #
#                                                                                                           #
#############################################################################################################
import sys
import os
import inspect
from loguru import logger
import random
import numpy as np
import torch


#############################################################################################################
# Debugging Utilities
#############################################################################################################
def debug_log(message, level="info", verbose=True):
    """
    Utility wrapper for profiling logging, supporting function-specific log files.

    :param message: The log message to record.
    :type message: str
    :param level: Log level ('info', 'success', 'warning', 'error', etc.).
    :type level: str
    :param verbose: Enable logging to module-dedicated file.
    :type verbose: bool
    """
    if not verbose:
        return
    
    # Get the calling function's name
    caller_frame = inspect.stack()[1]
    function_name = caller_frame.function

    # Configure the log file path in the CWD
    cwd = os.getcwd()
    log_file = os.path.join(cwd, f"{function_name}_stats.log")

    # Check if the log file is already active
    if os.path.exists(log_file) is False:
        func_logger = logger.bind()
        func_logger.remove()
        func_logger.add(log_file,
                        format="{time: YYYY-MM-DD > HH:mm:ss} | {level} | {message}",
                        level="DEBUG",
                        rotation="10 MB",
                        enqueue=True)

    # Log the message with the specified level
    getattr(logger, level)(message)


#############################################################################################################
# Experiments Utilities
#############################################################################################################
class HiddenPrints:
    """
    A context manager that suppresses stdout prints for the block of code it wraps.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def set_seeds(seed=42):
    """
    Set seeds for Python `random`, NumPy, PyTorch (CPU and CUDA).
    
    :param seed: Seed value to set.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # PyTorch random seed (for CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ensure deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_input(model_name: str, batch_size: int, duration: float | int, sample_rate: int, device='cpu'):
    """
    Generate a test input tensor or np.ndarray based on the model name and input specifications.

    :param model_name: The name of the model for which to generate the test input (see torch-audio-embeddings backbones).
    :type model_name: str
    :param batch_size: The batch size for the input tensor.
    :type batch_size: int
    :param duration: The duration of the input in seconds (float) or number of samples (int).
    :type duration: float | int
    :param sample_rate: The sample rate to use for generating the input (used only for 'float durations').
    :type sample_rate: int
    :param device: The device on which to allocate the tensor ('cpu' or 'cuda:<id>').
    :type device: str
    :return: A generated test input tensor or array.
    :rtype: torch.Tensor or np.ndarray
    """
    if duration <= 0:
        raise ValueError("Duration must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero.")
   
    # Seconds handling
    if type(duration) == float:
        if model_name == 'vggish':
            test_input = np.random.uniform(-1, 1, int(duration * sample_rate)).astype(np.float32)
        elif 'encodec_24' in model_name:
            test_input = torch.rand((batch_size, 1, int(duration * sample_rate)), dtype=float, device=device) * 2. - 1.
        elif 'encodec_48' in model_name:
            test_input = torch.rand((batch_size, 2, int(duration * sample_rate)), dtype=float, device=device) * 2. - 1.
        else:
            test_input = torch.rand((batch_size, int(duration * sample_rate)), dtype=float, device=device) * 2. - 1.
    # Samples handling
    else:
        if model_name == 'vggish':
            test_input = np.random.uniform(-1, 1, duration).astype(np.float32)
        elif 'encodec_24' in model_name:
            test_input = torch.rand((batch_size, 1, duration), dtype=float, device=device) * 2. - 1.
        elif 'encodec_48' in model_name:
            test_input = torch.rand((batch_size, 2, duration), dtype=float, device=device) * 2. - 1.
        else:
            test_input = torch.rand((batch_size, duration), dtype=float, device=device) * 2. - 1.

    return test_input.to(device) if model_name != 'vggish' else test_input
