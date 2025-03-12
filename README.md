# Fashion Aesthetic Design ML

This project implements a machine learning approach to augment the fashion design process, inspired by "Product Aesthetic Design: A Machine Learning Augmentation" (Burnap, Hauser, and Timoshenko, 2022) but adapted for the fashion domain.

## Overview

This system combines three key components:
1. **Encoder**: A deep learning model that transforms fashion images into a compact embedding space
2. **Predictor**: A model that predicts aesthetic ratings from image embeddings
3. **Generator**: A fine-tuned Stable Diffusion model that generates new fashion designs with desired aesthetic properties

## Setup

### Environment Setup

1. Create the overlay file and set up the environment:
```bash
sbatch sbatch_setup_env.sh
