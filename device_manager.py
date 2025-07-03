import os
import torch
from torch.distributed import init_process_group, destroy_process_group


class DeviceManager:
    def __init__(self):
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            assert torch.cuda.is_available(), "DDP requires CUDA to be available"
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.master_process = True
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
        print("[DeviceManager] using device:", self.device)
