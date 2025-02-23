import torch
import numpy as np
from deep_audio_embedding import load_model_dict, install_dependencies, import_pkgs, model_init, pre_proc, compute_embedding


# Parameters MISSING EFFICIENT_AT (LAST ONE)
model_name = 'yamnet'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    # Format the Path to the JSON configuration
    config_path = f"./backbones/general_purpose/{model_name}.json"

    # Load configuration dictionary
    model_dict = load_model_dict(config_path)
    
    # Install dependencies
    install_dependencies(model_dict)
    
    # Import required packages (handle any import context if present)
    imports = import_pkgs(model_dict)
    
    # Initialize the model
    model = model_init(model_dict, imports)
    
    # Synthetic input (e.g., random tensor of 3 batch, mono, 1 sec. each)
    x = torch.rand((3, int(model_dict['sample_rate'] * 1))) * 2 - 1
    #x = np.random.rand(int(model_dict['sample_rate'] * 1)) * 2 - 1
    
    # Pre-process the input
    x_proc = pre_proc(model_dict, x, imports)
    print("Pre-processed input shape:", x_proc.shape, x_proc.device)

    # Compute the embedding on CPU & GPU
    embedding = compute_embedding(model_dict, model.cpu(), x_proc.cpu(), imports)
    print("Embedding shape (CPU):", embedding.shape, embedding.device)

    if device == 'cuda:0':
        embedding = compute_embedding(model_dict, model.cuda(), x_proc.cuda(), imports)
    print("Embedding shape (GPU):", embedding.shape, embedding.device)
