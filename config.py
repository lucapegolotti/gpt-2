class Config:
    def __init__(self, data_manager):
        self.batch_size = 4
        self.total_batch_size = 2**19
        self.max_lr = 6e-4
        self.min_lr = self.max_lr * 0.1
        self.warmup_steps = 715
        self.max_steps = 19073
        self.block_size = 1024
        self.log_step = 250
        self.model_output_step = 250
        self.num_return_sequences_sample_training = 4
        self.max_length_sample_training = 32
        self.dataset_name = "tiny_shakespeare"  # "edu_fineweb10B" or "tiny_shakespeare"
        self.evaluate_benchmark = False
        self.data_manager = data_manager

        self.check_data_consistency()

        self.grad_accum_steps = self.total_batch_size // (
            self.batch_size * self.block_size * self.data_manager.ddp_world_size
        )

        if self.data_manager.master_process:
            print("[Config] Computing gradient accumulation steps")
            print(f"\tTotal desired batch size: {self.total_batch_size}")
            print(f"\tCalculated gradient accumulation steps: {self.grad_accum_steps}")

    def check_data_consistency(self):
        assert (
            self.total_batch_size % (self.batch_size * self.block_size) == 0
        ), "total batch size must be divisible by B * T"
