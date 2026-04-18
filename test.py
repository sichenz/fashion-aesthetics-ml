import torch
from pathlib import Path
from utils.device import get_device
from utils.data_utils import load_config, get_predictor_transforms, _score_to_label
from models.aesthetic_predictor import AestheticPredictor
from models.aesthetic_generator import AestheticGenerator
from utils.evaluation_utils import plot_generated_gallery

def run_test():
    device = get_device()
    config = load_config()
    
    # 1. Load the Aesthetic Predictor
    print("Loading Predictor...")
    pred_cfg = config["predictor"]
    predictor = AestheticPredictor(
        clip_model_name=pred_cfg["clip_model"],
        hidden_dims=pred_cfg["hidden_dims"],
        dropout=pred_cfg["dropout"],
        freeze_backbone=True
    ).to(device)
    predictor.eval()
    
    # 2. Load the Aesthetic Generator
    print("Loading Generator...")
    generator = AestheticGenerator(config, device)
    
    # 3. Generate two test designs
    print("Generating images...")
    prompt = f"a high quality fashion photograph, {_score_to_label(8.0)} aesthetic quality"
    images = generator.generate(
        prompt=prompt,
        num_images=2,
        num_steps=20,
        guidance_scale=7.5
    )
    
    # 4. Score the generated images
    print("Scoring images...")
    _, val_tf = get_predictor_transforms(224)
    scores = []
    for img in images:
        tensor = val_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            score = predictor(tensor).item()
        scores.append(score)
        
    # 5. Save the outputs
    out_dir = Path("outputs/generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        img.save(out_dir / f"test_design_{i+1}.png")
        print(f"Design {i+1} Score: {scores[i]:.2f}")
        
    plot_generated_gallery(images, scores, str(out_dir / "test_gallery.png"))
    print("Success! Outputs saved to outputs/generated/")

if __name__ == "__main__":
    run_test()
