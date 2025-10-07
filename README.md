# Py-Torch-(Deep)-Audio-Embeddings: A Lightweight API for Neural Audio Embedding
**Py-Torch-DAE** is a lightweight, zero-dependency Python interface that enables streamlined loading, initialization, and execution of pre-trained Deep Audio Embedding (DAE) models based on a flexible JSON configuration format.

It is designed for **benchmarking**, **modular interfacing**, and **experimental reproducibility** across multiple DAE architectures with varying runtime requirements.

---

## 🚀 Features
- Model-agnostic embedding pipeline via JSON templates description
- Fully customizable pre-processing, forward pass, and context configuration  
- *No third-party dependencies*: built entirely on standard Python (...for DAE usage only, for benchmarking see "requirements.txt")
- Very large-scale compatibility with PyTorch models implementation

---

## 📂 Repository Structure
```
Py-Torch-DAE/
├── deep_audio_embedding.py      # Core functions module
└── backbones/
    ├── general_purpose/
    ├── speech/
    ├── music/
    └── encoding/
```

Each folder under `backbones/` contains `.json` files specifying how to install, import, initialize, and run related DAE models.

---

## 🛠 How to Install

Clone the repository and make sure your system uses Python ≥ 3.8:

```bash
git clone https://github.com/your-org/Py-Torch-DAE.git
cd Py-Torch-DAE
```

No external packages are required. Tested on Debian GNU/Linux 12 (bookworm)

---

## ⚙️ Usage

Use the provided API to execute a full embedding pipeline. Here's an example:

```python
from deep_audio_embedding import (load_model_dict,
                                  install_dependencies,
                                  import_pkgs,
                                  model_init,
                                  pre_proc,
                                  compute_embedding)

# Load the model JSON template
model_dict = load_model_dict("general_purpose/[model_name].json")

# Install dependencies
install_dependencies(model_dict)

# Set up imports and environment
namespace = import_pkgs(model_dict)

# Init model
model = model_init(model_dict, namespace)                   # returns a Pytorch nn.Module, or an abstract Class containing it

# Pre-process input
x_pre = pre_proc(model_dict, x, namespace)                  # x should be a torch.Tensor (see TorchAudio.IO documentations)

# Compute embedding
y = compute_embedding(model_dict, model, x_pre, namespace)
```

---

## 📐 JSON Template Format

Each `.json` file must define the behavior of your embedding pipeline using the following keys:

| Key                  | Required | Description |
|----------------------|----------|-------------|
| `authors`            | ✅       | List of authors of the original work |
| `references`         | ✅       | Links to publications and code |
| `downstream_tasks`   | ✅       | Tags such as `"audio_representation"`, `"speech_recognition"` |
| `install_cmds`       | ❌       | Shell commands to set up the environment |
| `import_context`     | ❌       | Directory context for imports |
| `import_cmds`        | ✅       | Python or shell commands to import packages and files |
| `model_init`         | ✅       | Python code to initialize the model (must define `model`) |
| `model_init_context` | ❌       | Directory for executing `model_init` |
| `sample_rate`        | ✅       | Input sampling rate required by the model |
| `in_shape`           | ✅       | Input tensor shape (2 = [B, S], 3 = [B, C, S]) |
| `pre_proc`           | ❌       | Python code to preprocess input `x` |
| `pre_proc_context`   | ❌       | Directory for executing `pre_proc` |
| `embed_fwd`          | ✅       | Python code that assigns embedding to `y` |

---

## 🤝 Contributing

If you have a DAE model you'd like to integrate, please refer to following guide for template formatting instructions, and open a pull request with your `.json` file, or follow instructions on our ad-hoc submission site: **LINK**.

Thank you for contributing your DAE model to the PyTorch-DAE ecosystem!

To ensure your model is portable and compatible with our benchmarking framework, please follow the structure below when preparing your JSON configuration file.

---

## Required Keys

| Key               | Type      | Description |
|------------------|-----------|-------------|
| `authors`         | list[str] | Authors of the original paper or codebase |
| `references`      | list[str] | DOI, arXiv, or GitHub links |
| `downstream_tasks`| list[str] | Tasks the model was trained for (e.g. `"audio_representation"`) |
| `model_init`      | list[str] | Python code to initialize your model; must define `model` |
| `sample_rate`     | int       | Required input sampling rate |
| `in_shape`        | int       | Input tensor dimensionality: 2 = waveform, 3 = spectrogram batch |
| `embed_fwd`       | list[str] | Python code to extract embedding and assign it to variable `y` |

---

## Optional Keys

| Key                  | Type       | Description |
|----------------------|------------|-------------|
| `install_cmds`       | list[str]  | Shell commands for installing packages or downloading files |
| `import_context`     | str        | Directory to change into before `import_cmds` |
| `import_cmds`        | list[str]  | Python/shell code to prepare the runtime |
| `model_init_context` | str        | Directory to change into before `model_init` |
| `pre_proc`           | list[str]  | Python code to apply to the input `x` before forward pass |
| `pre_proc_context`   | str        | Directory to change into before running `pre_proc` |

---

## Example Structure

```json
{
  "authors": ["Jane Doe", "John Smith"],
  "references": ["https://arxiv.org/abs/1234.5678", "https://github.com/janedoe/audio-embed"],
  "downstream_tasks": ["audio_classification"],
  "install_cmds": ["git clone https://github.com/janedoe/audio-embed", "pip install torch"],
  "import_context": "./audio-embed/",
  "import_cmds": [
    "import sys",
    "import os",
    "sys.path.insert(0, os.getcwd())",
    "from model import MyDAE"
  ],
  "model_init": ["model = MyDAE(pretrained=True)"],
  "sample_rate": 16000,
  "in_shape": 2,
  "pre_proc": ["x = preprocess_audio(x)"],
  "embed_fwd": ["y = model(x)"]
}

---

## 📬 Contact

For inquiries or integration support, please open an issue or reach out at: stefano.giacomelli@graduate.univaq.it
