# models/aesthetic_generator.py
"""
Aesthetic-Conditioned Design Generator
Architecture: Stable Diffusion 1.5 + LoRA fine-tuning

Uses PEFT LoRA to fine-tune the UNet on fashion images, conditioned on
aesthetic scores embedded in the text prompt. This allows generating
novel designs at a target aesthetic quality level.
"""
import torch
import os
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model


class AestheticGenerator:
    """
    Stable Diffusion 1.5 with LoRA fine-tuning for aesthetic-conditioned generation.
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.model_id = config["generator"]["model_id"]
        gen_cfg = config["generator"]

        print(f"Loading Stable Diffusion pipeline: {self.model_id}")
        dtype = torch.float32  # MPS requires float32

        # Load components individually for fine-tuning control
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=dtype
        ).to(device)
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=dtype
        ).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", torch_dtype=dtype
        ).to(device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

        # Freeze everything except UNet LoRA
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # Apply LoRA to UNet
        lora_config = LoraConfig(
            r=gen_cfg["lora_rank"],
            lora_alpha=gen_cfg["lora_alpha"],
            target_modules=gen_cfg["lora_target_modules"],
            lora_dropout=0.05,
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()

        # VAE scaling factor
        self.vae_scale = self.vae.config.scaling_factor

    def get_trainable_params(self):
        """Return trainable (LoRA) parameters for the optimizer."""
        return [p for p in self.unet.parameters() if p.requires_grad]

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel images to latent space."""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample() * self.vae_scale
        return latents

    def encode_prompt(self, prompts: list) -> torch.Tensor:
        """Encode text prompts to embeddings."""
        tok = self.tokenizer(
            prompts, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            emb = self.text_encoder(tok.input_ids)[0]
        return emb

    def training_step(self, batch: dict) -> torch.Tensor:
        """Single training step — returns the loss."""
        images = batch["image"].to(self.device)
        captions = batch["caption"]

        # Encode images to latents
        latents = self.encode_images(images)

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bs,), device=self.device,
        ).long()

        # Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Encode prompts
        encoder_hidden_states = self.encode_prompt(captions)

        # Predict noise
        noise_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states,
        ).sample

        # MSE loss (noise prediction objective)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        return loss

    def save_lora(self, save_dir: str):
        """Save LoRA adapter weights."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(save_dir)
        print(f"Saved LoRA weights to {save_dir}")

    def load_lora(self, load_dir: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        self.unet = PeftModel.from_pretrained(self.unet.base_model.model, load_dir).to(self.device)
        print(f"Loaded LoRA weights from {load_dir}")

    @torch.no_grad()
    def generate(
        self, prompt, negative_prompt="", num_images=1,
        guidance_scale=7.5, num_steps=30, seed=None,
    ):
        """Generate images using the fine-tuned pipeline."""
        # Build a pipeline from our components for inference
        pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(self.device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images
