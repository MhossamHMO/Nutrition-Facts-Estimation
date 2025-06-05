# Nutrition5K Calorie and Nutrient Estimator Using BLIP

This repository contains code for a deep learning model that estimates the nutritional values—calories, protein, fat, and carbohydrates—from food images and their corresponding textual descriptions. The model leverages the BLIP (Bootstrapping Language-Image Pre-training) architecture by Salesforce for multi-modal feature extraction and regression.

---

## Project Description

The project aims to predict the nutritional content of food items by combining visual and textual data. Using the **Nutrition5K** dataset, which contains food images alongside cleaned captions and nutrition labels, we fine-tune the BLIP model to perform multi-output regression. The approach fuses image embeddings and text embeddings to capture both modalities for accurate nutritional value prediction.

This model can support applications in personalized diet tracking, food journaling apps, and nutrition monitoring without manual calorie counting.

---

## Features

- Multi-modal input (image + text caption)
- Normalization and denormalization of nutritional values for better regression performance
- Fine-tuning of BLIP model's image-text embeddings
- Modular and reusable PyTorch implementation
- Easy-to-run training and evaluation scripts with detailed loss reporting
- Batch processing with DataLoader and data splitting for train/test evaluation

---

## Getting Started

### Requirements

- Python 3.7+
- PyTorch
- Transformers (HuggingFace)
- torchvision
- pandas
- tqdm
- PIL (Pillow)

### Installation

Install the dependencies via pip:

```bash
pip install torch torchvision transformers pandas tqdm pillow
