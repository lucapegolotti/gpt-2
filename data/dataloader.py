import os
import numpy as np
import torch


class DataLoaderLite:
    def __init__(self, config, device_manager, encoder, split):
        self.device_manager = device_manager
        self.B = config.batch_size
        self.T = config.block_size
        self.process_rank = device_manager.ddp_rank
        self.num_processes = device_manager.ddp_world_size
        self.split = split
        self.dataset_name = config.dataset_name
        assert self.split in {"train", "val"}
        assert self.dataset_name in {"edu_fineweb10B", "tiny_shakespeare"}

        if config.dataset_name == "edu_fineweb10B":
            self.initialize_edu_fineweb10B()
        elif config.dataset_name == "tiny_shakespeare":
            assert (
                self.num_processes == 1
            ), "tiny_shakespeare only supports single-GPU training for now"
            self.initialize_tiny_shakespeare(encoder)

    def load_tokens(self, filename):
        npt = np.load(filename)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def initialize_edu_fineweb10B(self):
        # get the shard filenames
        data_root = "data/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if self.split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {self.split}"
        if self.device_manager.master_process:
            print(f"[DataLoaderLite] Found {len(shards)} shards for split {self.split}")
        self.reset()

    def initialize_tiny_shakespeare(
        self, encoder, input_file="data/tiny_shakespeare.txt"
    ):
        with open(input_file, "r") as file:
            data = file.read()
        self.tokens = torch.tensor(encoder.encode(data), dtype=torch.long)
        ntokens = len(self.tokens)

        # we use a fix split of 0.95 for train and 0.05 for validation.
        # This of course should be optimized and moved to the configuration file.
        ntrain = int(ntokens * 0.95) // (self.B * self.T) * (self.B * self.T) + 1
        nval = int(ntokens * 0.05) // (self.B * self.T) * (self.B * self.T) + 1
        if self.split == "train":
            self.tokens = self.tokens[:ntrain]

        else:
            self.tokens = self.tokens[ntrain : ntrain + nval]

        self.current_position = 0

    def reset(self):
        if self.dataset_name == "edu_fineweb10B":
            # state, init at shard zero
            self.current_shard = 0
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        else:
            self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor

        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            if self.dataset_name == "edu_fineweb10B":
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y
