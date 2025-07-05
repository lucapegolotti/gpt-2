import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from hellaswag import render_example, iterate_examples
from model import GPT, GPTConfig
from device_manager import DeviceManager
from config import Config
from log_manager import LogManager
from dataloader import DataLoaderLite

import tiktoken


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def get_lr(it, config):
    if it < config.warmup_steps:
        return config.max_lr * (it + 1) / config.warmup_steps
    if it > config.max_steps:
        return config.min_lr

    decay_ratio = (it - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <= 1, "decay ratio out of bounds: %f" % decay_ratio
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


def evaluate_benchmark(step, model, data_manager, log_manager):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        if i % data_manager.ddp_world_size != data_manager.ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(data_manager.device)
        mask = mask.to(data_manager.device)
        with torch.no_grad():
            with torch.autocast(device_type=data_manager.device, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    if data_manager.ddp:
        num_total = torch.tensor(
            num_total, dtype=torch.long, device=data_manager.device
        )
        num_correct_norm = torch.tensor(
            num_correct_norm, dtype=torch.long, device=data_manager.device
        )
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if data_manager.master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        log_manager.to_file(step, "benchmark", acc_norm)


def evaluate_validation(step, model, data_manager, log_manager):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(dm.device), y.to(dm.device)
            with torch.autocast(device_type=dm.device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss /= val_loss_steps
            val_loss_accum += loss.detach()
    if data_manager.ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if data_manager.master_process:
        print(f"validation loss at step {step}: {val_loss_accum.item():.4f}")
        log_manager.to_file(step, "val", val_loss_accum)


def save_model(raw_model, log_manager):
    checkpoint_path = os.path.join(log_manager.dir, f"model.pt")
    checkpoint = {
        "model": raw_model.state_dict(),
        "config": raw_model.config,
        "step": step,
    }
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    # we set TF32
    torch.set_float32_matmul_precision("high")

    dm = DeviceManager()

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # this is a trick to be able to use the same batch size as the original GPT-2
    # without having to load huge batches into memory
    config = Config(data_manager=dm)

    enc = tiktoken.get_encoding("gpt2")
    train_loader = DataLoaderLite(
        device_manager=dm,
        config=config,
        split="train",
        encoder=enc,
    )
    val_loader = DataLoaderLite(
        device_manager=dm,
        config=config,
        split="val",
        encoder=enc,
    )

    # we overwrite the vocab size (froom 50257 to 50304) to make the number "nice"
    # (it can be divided by many powers of 2)
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(dm.device)
    use_compile = False
    if torch.cuda.is_available() and use_compile:
        torch.compile(model)
    if dm.ddp:
        model = DDP(model, device_ids=[dm.ddp_local_rank])
    raw_model = model.module if dm.ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=3e-4, device=dm.device
    )

    # object use to print to file metrics (train, validation, benchmark performance)
    lm = LogManager(dir="log")

    # training loop
    for step in range(config.max_steps):
        t0 = time.time()
        last_step = step == config.max_steps - 1

        if (step % config.log_step == 0 or last_step) and (not use_compile):
            prompt = "Hello, I'm a language model,"
            model.sample_sequence(
                prompt,
                dm,
                enc,
                config.num_return_sequences_sample_training,
                config.max_length_sample_training,
                top_priority=50,
            )
            if config.evaluate_benchmark:
                evaluate_benchmark(step, model, dm, lm)
            evaluate_validation(step, model, dm, lm)

        if step > 0 and (step % config.model_output_step == 0 or last_step):
            save_model(raw_model, lm)

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(config.grad_accum_steps):
            x, y = train_loader.next_batch()
            # useful line to use float16 precision for training
            if torch.cuda.is_available():
                with torch.autocast(device_type=dm.device, dtype=torch.bfloat16):
                    logits, loss = model(x.to(dm.device), y.to(dm.device))
            # on MPS, autocast to bfloat16 is slower, so we use float32
            else:
                logits, loss = model(x.to(dm.device), y.to(dm.device))
            # we need to divide the loss by the number of steps to make sure that we get
            # the same loss as if we had not accumulated gradients
            loss /= config.grad_accum_steps
            loss_accum += loss.detach()  # accumulate loss for logging
            # this is to ensure that the communication of gradients only happens at the
            # very last step to avoid unnecessary communication overhead
            if dm.ddp:
                model.require_backward_grad_sync = (
                    micro_step == config.grad_accum_steps - 1
                )
            # here we accumulate the steps
            loss.backward()
        # this is to ensure that the loss is averaged across all processes
        if dm.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        loss_accum = loss_accum.item()  # convert to python float for logging
        # clip gradients to avoid exploding gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # optimizer step
        optimizer.step()

        # synchronize across GPUS
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # compute times
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_second = (
            (x.size(0) * x.size(1) * config.grad_accum_steps)
            / (dt / 1000.0)
            * dm.ddp_world_size
        )
        if dm.master_process:
            print(
                f"step {step} | loss: {loss_accum:.4f} | lr: {lr:.3e} | norm: {norm:.4f} | time: {dt:.2f} ms | tokens/sec: {tokens_per_second:.2f}"
            )
            lm.to_file(step, "train", loss_accum)

    dm.terminate()
