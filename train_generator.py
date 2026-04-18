# train_generator.py
"""
Fine-tune Stable Diffusion 1.5 with LoRA for aesthetic-conditioned generation.

The UNet learns to associate aesthetic quality descriptors in the text prompt
with the visual characteristics that correspond to high/low aesthetic scores.

Usage:
    python train_generator.py
    python train_generator.py --config configs/config.yaml --steps 3000
"""
import argparse
import os
import time
import torch
from pathlib import Path
from tqdm import tqdm

from utils.device import get_device, print_device_info
from utils.data_utils import load_config, set_seed, get_generator_dataloader
from models.aesthetic_generator import AestheticGenerator


def train_generator(config_path: str, steps_override: int = None):
    """Fine-tune Stable Diffusion with LoRA on aesthetic-scored fashion images."""
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    print_device_info()

    device = get_device()
    gen_cfg = config["generator"]
    max_steps = steps_override or gen_cfg["max_train_steps"]
    grad_accum = gen_cfg["gradient_accumulation_steps"]

    # ── Data ──────────────────────────────────────────────────────────────
    dataloader = get_generator_dataloader(config)

    # ── Model ─────────────────────────────────────────────────────────────
    generator = AestheticGenerator(config, device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        generator.get_trainable_params(),
        lr=gen_cfg["learning_rate"],
        weight_decay=1e-2,
    )

    warmup_steps = gen_cfg.get("lr_warmup_steps", 100)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──────────────────────────────────────────────────────────
    ckpt_dir = Path(config["paths"]["checkpoints_dir"]) / "generator_lora"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_steps = gen_cfg.get("save_steps", 500)

    print(f"\n{'='*60}")
    print(f"  Fine-tuning SD 1.5 with LoRA — {max_steps} steps")
    print(f"  LR: {gen_cfg['learning_rate']}, Batch: {gen_cfg['batch_size']}, "
          f"Grad Accum: {grad_accum}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    generator.unet.train()
    generator.text_encoder.eval()
    generator.vae.eval()

    global_step = 0
    running_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    pbar = tqdm(total=max_steps, desc="Training Generator")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            loss = generator.training_step(batch)
            loss = loss / grad_accum
            loss.backward()

            running_loss += loss.item()

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(generator.get_trainable_params(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            pbar.update(1)

            # Logging
            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}",
                                 elapsed=f"{elapsed:.0f}s")
                running_loss = 0.0

            # Save checkpoint
            if global_step % save_steps == 0:
                save_path = str(ckpt_dir / f"step_{global_step}")
                generator.save_lora(save_path)

    pbar.close()

    # Save final checkpoint
    final_path = str(ckpt_dir / "final")
    generator.save_lora(final_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Generator Training Complete!")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Final LoRA saved to: {final_path}")
    print(f"{'='*60}")

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SD 1.5 with LoRA")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    train_generator(args.config, args.steps)
