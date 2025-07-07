import torch
import argparse
import os
import tiktoken

from model import GPT, GPTConfig
from device_manager import DeviceManager
from config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Load a pre-trained GPT-2 model and sample text."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pt file) or a HuggingFace model type (e.g., gpt2, gpt2-medium).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, I'm a language model,",
        help="Initial text prompt for generation.",
    )
    parser.add_argument(
        "--response_length",
        type=int,
        default=32,
        help="Maximum length of the generated response.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of independent sequences to generate.",
    )

    args = parser.parse_args()

    # Device management
    dm = DeviceManager()  #

    # Load tiktoken encoder
    enc = tiktoken.get_encoding("gpt2")

    # Load the model
    model = None
    if os.path.exists(args.model) and args.model.endswith(".pt"):
        print(f"Loading model from checkpoint: {args.model}")
        checkpoint = torch.load(args.model, map_location=dm.device, weights_only=False)
        model_config = checkpoint["config"]
        model = GPT(model_config)  #
        model.load_state_dict(checkpoint["model"])
    elif args.model == "gpt2":
        print(f"Loading HuggingFace pre-trained model: {args.model}")
        model = GPT.from_pretrained(args.model)  #
    else:
        print(
            f"Error: Invalid model path or HuggingFace model type provided: {args.model}"
        )
        return

    model.to(dm.device)  #
    model.eval()  # Set model to evaluation mode

    # Sample text
    print("Generating text...")
    model.sample_sequence(
        args.prompt,
        dm,
        enc,
        args.num_return_sequences,
        args.response_length,
        top_priority=50,
    )  #

    dm.terminate()  #


if __name__ == "__main__":
    main()
