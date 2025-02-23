#############################################################################################################
#                                                                                                           #
#   Author      : Stefano Giacomelli                                                                        #
#   Affiliation : PhD candidate, University of L'Aquila (Italy)                                             #
#   Department  : DISIM - Department of Information Engineering, Computer Science and Mathematics           #
#   Description : Embedding models parsing functions for audio signals embedding.                           #
#   Last Update : 2025-02-23                                                                                #
#                                                                                                           #
#############################################################################################################
import os
import ast
import json
import subprocess


def load_model_dict(filepath):
    """
    Load a backbone JSON file from the given path and return its contents as a Python dictionary.
    
    :param filepath: Path to the backbone embedding JSON file.
    :type filepath: str
    :return: Parsed JSON as a dictionary.
    :rtype: dict
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found.")
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file '{filepath}': {e}")
    return data


def install_dependencies(model_dict):
    """
    Execute terminal commands from the backbone 'install_cmds' key to install required dependencies.

    :param model_dict: Dictionary containing model configuration.
    :type model_dict: dict
    :return: None
    """
    cmds = model_dict.get("install_cmds", [])
    original_dir = os.getcwd()

    try:
        for cmd in cmds:
            cmd = cmd.strip()
            if cmd.startswith("cd "):
                target_dir = cmd[3:].strip()
                os.chdir(target_dir)
                print(f"Changed directory to: {target_dir}")
            else:
                subprocess.run(cmd, shell=True, check=True)
                print(f"Successfully executed: {cmd}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        raise RuntimeError(f"Failed to execute command '{cmd}': {e}")
    finally:
        os.chdir(original_dir)
        print(f"Returned to original directory: {original_dir}")


def import_pkgs(model_dict):
    """
    Execute import commands specified in 'import_cmds'. If an 'import_context' key is present,
    change the working directory to that folder before executing, and then return to the original directory.
    
    This function attempts to determine if a command is valid Python code by trying to parse it
    using the ast module. Valid Python commands are executed using exec() (so commands like
    "sys.path.insert(0, os.getcwd())" will work), while non-Python commands are executed as shell commands.
    
    :param model_dict: Dictionary containing model configuration.
    :type model_dict: dict
    :return: A namespace dictionary with the results of the executed import commands.
    :rtype: dict
    """
    import_namespace = {}
    original_dir = os.getcwd()
    context_dir = model_dict.get("import_context", None)
    
    try:
        if context_dir:
            os.chdir(context_dir)
        cmds = model_dict.get("import_cmds", [])
        for cmd in cmds:
            cmd = cmd.strip()

            try:
                ast.parse(cmd)
                is_python = True
            except SyntaxError:
                is_python = False
            
            if is_python:
                try:
                    exec(cmd, globals(), import_namespace)
                    print(f"Executed Python command: {cmd}")
                except Exception as e:
                    raise ImportError(f"Error executing Python command '{cmd}': {e}")
            else:
                try:
                    subprocess.run(cmd, shell=True, check=True)
                    print(f"Executed shell command: {cmd}")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Shell command '{cmd}' failed: {e}")
    finally:
        if context_dir:
            os.chdir(original_dir)
    
    return import_namespace


def model_init(model_dict, import_namespace):
    """
    Initialize the model by executing the Python commands in the 'model_init' key.
    If an 'import_context' is provided, the working directory is temporarily changed to that directory.
    
    :param model_dict: Dictionary containing model configuration.
    :type model_dict: dict
    :param import_namespace: Namespace dictionary returned from import_pkgs.
    :type import_namespace: dict
    :return: The initialized model instance.
    :rtype: object
    """
    cmds = model_dict.get("model_init", [])
    if not cmds:
        raise KeyError("No 'model_init' commands found in the model configuration.")
    
    namespace = {}
    namespace.update(import_namespace)
    
    original_dir = os.getcwd()
    context_dir = model_dict.get("model_init_context", None)
    try:
        if context_dir:
            os.chdir(context_dir)
        for cmd in cmds:
            try:
                exec(cmd, globals(), namespace)
                print(f"Executed model initialization command: {cmd}")
            except Exception as e:
                raise RuntimeError(f"Error executing model initialization command '{cmd}': {e}")
    finally:
        if context_dir:
            os.chdir(original_dir)
    
    if "model" not in namespace:
        raise ValueError("Model initialization failed: 'model' not defined in namespace.")
    
    return namespace["model"]


def pre_proc(model_dict, x, import_namespace):
    """
    Pre-process the input data by executing the Python commands in the 'pre_proc' key.
    If an 'import_context' is provided, the working directory is temporarily changed to that directory.
    
    :param model_dict: Dictionary containing model configuration.
    :type model_dict: dict
    :param x: Input data to be pre-processed.
    :param import_namespace: Namespace dictionary returned from import_pkgs.
    :type import_namespace: dict
    :return: Pre-processed input data.
    """
    cmds = model_dict.get("pre_proc", None)
    if not cmds:
        print("No 'pre_proc' commands found in the model configuration. returning input 'x'")
        return x  # Return x unchanged if no pre-processing commands are provided.
    
    namespace = {"x": x}
    namespace.update(import_namespace)
    
    original_dir = os.getcwd()
    context_dir = model_dict.get("pre_proc_context", None)
    try:
        if context_dir:
            os.chdir(context_dir)
        for cmd in cmds:
            try:
                exec(cmd, globals(), namespace)
                print(f"Executed pre-processing command: {cmd}")
            except Exception as e:
                raise RuntimeError(f"Error executing pre-processing command '{cmd}': {e}")
    finally:
        if context_dir:
            os.chdir(original_dir)
    
    if "x" not in namespace:
        raise ValueError("Pre-processing failed: 'x' not defined after execution.")
    
    return namespace["x"]


def compute_embedding(model_dict, model, x, import_namespace):
    """
    Compute the embedding by executing the Python commands in the 'embed_fwd' key.
    
    :param model_dict: Dictionary containing model configuration.
    :type model_dict: dict
    :param model: The initialized model instance.
    :param x: Pre-processed input data.
    :param import_namespace: Namespace dictionary returned from import_pkgs.
    :type import_namespace: dict
    :return: Computed embedding.
    """
    cmds = model_dict.get("embed_fwd", [])
    if not cmds:
        raise KeyError("No 'embed_fwd' commands found in the model configuration.")
    
    namespace = {"model": model, "x": x}
    namespace.update(import_namespace)
    
    for cmd in cmds:
        try:
            exec(cmd, globals(), namespace)
            print(f"Executed embedding command: {cmd}")
        except Exception as e:
            raise RuntimeError(f"Error executing embedding command '{cmd}': {e}")
    
    if "y" not in namespace:
        raise ValueError("Embedding computation failed: 'y' not defined after execution.")
    return namespace["y"]
