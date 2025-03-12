# Fast Fashion Stable Diffusion Framework

This project implements a machine learning approach to augment the fashion design process

## Overview

This system combines three key components:
1. **Encoder**: A deep learning model that transforms fashion images into a compact embedding space
2. **Predictor**: A model that predicts aesthetic ratings from image embeddings
3. **Generator**: A fine-tuned Stable Diffusion model that generates new fashion designs with desired aesthetic properties

```
fashion-aesthetics-ml/
├── data/
│   ├── raw/                 # Raw downloaded datasets
│   ├── processed/           # Processed images and metadata
│   └── embeddings/          # Stored embeddings
├── models/
│   ├── encoder.py           # Encoder model
│   ├── predictor.py         # Aesthetic predictor model
│   └── generator.py         # SD-based generator
├── utils/
│   ├── data_utils.py        # Data loading and processing
│   ├── training_utils.py    # Training helpers 
│   └── evaluation_utils.py  # Evaluation metrics
├── configs/
│   └── config.yaml          # Configuration parameters
├── scripts/
│   ├── download_data.sh     # Data downloading script
│   ├── preprocess_data.py   # Data preprocessing
│   └── train_*.sh           # Training scripts
├── main.py                  # Main entry point
├── train_encoder.py         # Encoder training
├── train_predictor.py       # Predictor training
├── train_generator.py       # Generator fine-tuning
└── inference.py             # Generation and prediction scripts
```

## Setup

### Environment Setup

1. Create and set up the environment:
```bash
conda create -n fashion_env python=3.10
conda activate fashion_env
pip install -r requirements.txt
```

### Data Preparation
1. Download datasets:
   ```bash
   sbatch sbatch_download_data.sh
   ```
2. Preprocess datasets:
   ```bash
   sbatch sbatch_preprocess_data.sh
   ```

### Training
Train each component separately:
```bash
sbatch sbatch_train_encoder.sh  # Train encoder
sbatch sbatch_train_predictor.sh  # Train predictor
sbatch sbatch_train_generator.sh  # Train generator
```

### Evaluation and Inference
Evaluate model performance and generate new fashion designs:
```bash
sbatch sbatch_evaluate.sh  # Evaluate models
sbatch sbatch_inference.sh  # Generate new designs
```

## Advanced Usage

### Customizing Generation
You can generate custom designs by modifying the prompt:
```bash
python inference.py --num_samples 5 --target_rating 4.5 --style casual --color blue
```

### Model Tuning
To fine-tune the models, modify `configs/config.yaml` or pass parameters directly to the training scripts:
```bash
python main.py --mode train_encoder --batch_size 64 --epochs 30 --lr 2e-4 --embedding_dim 1024
```

## Citation
If you use this code, please cite the original paper:
```bibtex
@article{burnap2022product,
  title={Product Aesthetic Design: A Machine Learning Augmentation},
  author={Burnap, Alex and Hauser, John R and Timoshenko, Artem},
  journal={Marketing Science},
  year={2022}
}
```

## Running on NYU HPC Cluster

To run this project on the NYU HPC cluster, please refer to the CLUSTER.md

## Technologies Used
- **ConvNeXT-Large** for the encoder backbone
- **Stable Diffusion 3** with LoRA fine-tuning for generation
- **Progressive training and adversarial components** for enhanced performance
- **SAM (Segment Anything Model)** for image mask generation

