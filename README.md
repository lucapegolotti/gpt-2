Here's a README for your GPT-2 repository:

# GPT-2 Training and Evaluation

This repository provides code for training and evaluating GPT-2 models. It supports training on either the "Tiny Shakespeare" dataset or a sample of the "FineWeb-Edu" dataset. The project also includes functionality for evaluating the trained model on the HellaSwag benchmark.

**Acknowledgement:** This codebase is based on the accompanying materials for the YouTube video "[Let's build GPT: from scratch, in PyTorch](https://www.youtube.com/watch?v=l8pRSuU81PU&t=1787s)" by Andrej Karpathy. The `hellaswag.py` and `fineweb.py` files have been minimally edited from the original code provided in the video's context.

## Table of Contents

  - [Features](https://www.google.com/search?q=%23features)
  - [File Structure](https://www.google.com/search?q=%23file-structure)
  - [Setup](https://www.google.com/search?q=%23setup)
  - [Data Preparation](https://www.google.com/search?q=%23data-preparation)
  - [Training](https://www.google.com/search?q=%23training)
  - [Evaluation](https://www.google.com/search?q=%23evaluation)
  - [Configuration](https://www.google.com/search?q=%23configuration)

## Features

  * **GPT-2 Model Implementation:** A PyTorch implementation of the GPT-2 architecture, including `CausalSelfAttention` and `MLP` blocks.
  * **Flexible Data Loading:** Supports loading and tokenizing the "Tiny Shakespeare" dataset and a sample of the "FineWeb-Edu" dataset.
  * **Distributed Training Support:** Configured to run training across multiple GPUs using PyTorch's DistributedDataParallel (DDP).
  * **HellaSwag Benchmark Integration:** Includes a utility to download and evaluate the model's performance on the HellaSwag benchmark.
  * **Configurable Training Parameters:** Easily adjust hyperparameters such as batch size, learning rate, and training steps via `config.py`.
  * **Logging:** Training and evaluation metrics are logged to a file in the `log` directory.

## File Structure

```
.
├── .gitignore
├── config.py
├── create_venv.sh
├── data/
│   ├── dataloader.py
│   ├── fineweb.py
│   ├── hellaswag.py
│   └── tiny_shakespeare.txt
├── device_manager.py
├── log_manager.py
├── model.py
├── requirements.txt
├── run.sh
└── train_gpt2.py
```

  * `config.py`: Defines training hyperparameters and data-related configurations.
  * `create_venv.sh`: Script to set up a Python virtual environment.
  * `data/dataloader.py`: Handles data loading and batching for training.
  * `data/fineweb.py`: Script to download and preprocess the FineWeb-Edu dataset.
  * `data/hellaswag.py`: Contains utilities for downloading and evaluating on the HellaSwag benchmark.
  * `data/tiny_shakespeare.txt`: The Tiny Shakespeare dataset.
  * `device_manager.py`: Manages CUDA device and DistributedDataParallel (DDP) setup.
  * `log_manager.py`: Handles logging of training and evaluation metrics.
  * `model.py`: Defines the GPT-2 model architecture.
  * `requirements.txt`: Lists all Python dependencies.
  * `run.sh`: Example script to run distributed training.
  * `train_gpt2.py`: The main script for training the GPT-2 model.

## Setup

To set up the environment, follow these steps:

1.  **Create a Virtual Environment:**
    ```bash
    ./create_venv.sh
    ```
    This script will create a virtual environment named `gpt2-venv` in your repository root.
2.  **Activate the Virtual Environment:**
    ```bash
    . gpt2-venv/bin/activate
    ```
3.  **Install Dependencies:**
    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

This repository supports two datasets:

  * **Tiny Shakespeare:** Already included in `data/tiny_shakespeare.txt`. This dataset is suitable for single-GPU training.
  * **FineWeb-Edu:** For larger-scale training, you can use the `data/fineweb.py` script to download and tokenize a sample of the FineWeb-Edu dataset. This will create a directory named `edu_fineweb10B` with data shards.
    ```bash
    python data/fineweb.py
    ```
  * **HellaSwag:** The HellaSwag evaluation script (`data/hellaswag.py`) will automatically download the necessary validation data when run.

## Training

To train the GPT-2 model:

1.  **Configure Training Parameters:**
    Edit `config.py` to adjust hyperparameters like `batch_size`, `max_steps`, and `dataset_name`. Set `dataset_name` to `"tiny_shakespeare"` or `"edu_fineweb10B"` as desired.
    ```python
    # config.py
    self.dataset_name = "tiny_shakespeare"  # or "edu_fineweb10B"
    ```
2.  **Run Training:**
    You can run training on a single GPU or in a distributed manner.
      * **Single GPU (Example):**
        ```bash
        python train_gpt2.py
        ```
      * **Distributed Training (Example with 8 GPUs):**
        ```bash
        ./run.sh
        ```
        This script uses `torchrun` for distributed training.

During training, the model will periodically log loss, learning rate, and gradient norm. It will also generate text samples and, if `do_evaluate_benchmark` is set to `True` in `config.py`, perform HellaSwag evaluation. Model checkpoints will be saved in the `log` directory.

## Evaluation

To evaluate a pre-trained GPT-2 model (e.g., `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) on the HellaSwag benchmark:

```bash
python data/hellaswag.py --model_type gpt2 --device cuda
```

This script will report `acc_norm` and `acc` metrics for the specified model on the HellaSwag validation set.

## Configuration

The `config.py` file allows you to customize various aspects of the training process:

  * `batch_size`: Batch size per GPU.
  * `total_batch_size`: Total batch size across all GPUs, used to calculate gradient accumulation steps.
  * `max_lr`, `min_lr`: Maximum and minimum learning rates for cosine decay.
  * `warmup_steps`, `max_steps`: Learning rate schedule parameters.
  * `block_size`: Sequence length for the model.
  * `dataset_name`: Specifies which dataset to use (`"tiny_shakespeare"` or `"edu_fineweb10B"`).
  * `do_evaluate_benchmark`: Set to `True` to perform HellaSwag evaluation during training.