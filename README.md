# 🍽️ BLIP-Nutrition Regression: Vision-Language Model for Estimating Food Calories and Macronutrients

This project implements a **multi-modal regression model** based on **BLIP (Bootstrapped Language-Image Pre-training)** to estimate four key nutritional values—**Calories (kcal), Protein, Fat, and Carbohydrates**—from food images and descriptive captions.

---

## 1. Project Overview 🏗️

The project pipeline involves three main stages: **Data Ingestion**, **Preprocessing**, and **Model Training**.

| Stage | Input | Process | Output |
| :--- | :--- | :--- | :--- |
| **Data Ingestion** | 🖼️ Images (zipped) + 📄 Captions CSV | Extract images, load captions | Raw image folders + `dish_manual_captions.csv` |
| **Preprocessing (Phase 1)** | Raw data | Extract nutritional values (kcal, protein, fat, carbs) via regex, merge with image paths | `nutrition5k_regression_data.csv` |
| **Preprocessing (Phase 2)** | `nutrition5k_regression_data.csv` | Clean captions by removing numeric nutrition info | `cleaned_nutrition5k.csv` |
| **Modeling** | `cleaned_nutrition5k.csv` + Image folders | Train BLIP Regressor using MSE Loss | `blip_regressor.pth` |

---

## 2. Dataset Overview 📊

The dataset links **food images** and **manual captions** via `dish_id` (3,490 unique dishes).

### A. Required Data 📂

1. **Image Data**  
   - Download and extract images folder: [Google Drive Link]([https://drive.google.com/file/d/1VnLx3Odmla4rGRI0Bk-9DCjotncFX07O/view?usp=sharing](https://drive.google.com/file/d/18TqLtLXWZc-dnhH7jyVUZ85F7TmGmZMD/view?usp=drive_link))  
   - Folder contains subfolders (e.g., `/realsense_overhead/`) with final images (`rgb.png`).  

2. **Caption Data (CSV)**  
   - `dish_manual_captions.csv` mapping `dish_id` → `caption`.

> ⚠️ **Note:** Notebook was developed on Google Colab using Google Drive mount. For local use, no mount is needed—ensure images are extracted and paths in the CSV match your folder structure.

### B. CSV Files

All CSV files used and generated in the pipeline will be uploaded to the repo:

- **Raw captions:** `dish_manual_captions.csv`  
- **Merged data with nutrition labels:** `nutrition5k_regression_data.csv`  
- **Cleaned captions for training:** `cleaned_nutrition5k.csv`  

> Images are **not included in the repo** due to size; download from the Drive link above.

---

## 3. Data Preprocessing 🧹

1. **Ground Truth Extraction & Merging**  
   - Extract **kcal, protein, fat, carbs** from captions using regex.  
   - Merge with corresponding image paths → `nutrition5k_regression_data.csv`.  

2. **Caption Cleaning**  
   - Remove numeric values and descriptive phrases (“It provides approximately…”)  
   - Result: `cleaned_nutrition5k.csv` (used for model training).  

3. **Label Normalization**  
   - For the optimized model, regression targets are **z-score normalized** (mean=0, std=1) to improve training stability.  

---

## 4. Model Implementation 🤖

This project uses a custom **BLIPRegressor** model, built on the BLIP Vision-Language architecture, for predicting four continuous nutritional values: **Calories, Protein, Fat, and Carbohydrates**. The model processes both food images and cleaned text captions, fusing their embeddings before passing them through a regression head.

### Model Architecture

- **Base Model:** `BlipModel` from `"Salesforce/blip-image-captioning-base"`.
- **Input Processing:** Images and text captions are encoded by BLIP, producing separate embeddings.
- **Fusion:** Image and text embeddings are concatenated into a single vector.
- **Regression Head:** An MLP maps the fused embedding (size 1024) to the four nutritional outputs. It consists of:
  - Linear layers with ReLU activations
  - Dropout layers (0.3) for regularization
  - Final linear layer outputs the four nutritional values

### Training Versions

Two versions were tested, differing in how the target labels were handled:

1. **Direct (Raw) Version:**  
   - Uses raw nutritional values as targets.  
   - High MSE due to large numerical ranges of the targets.  
   
2. **Normalized Version:**  
   - Targets are standardized using the dataset mean and standard deviation.  
   - Training is more stable, resulting in a significantly lower MSE.  
   - Predictions must be **denormalized** to convert back to real nutritional values.

### Normalization Details

- Each nutrient is standardized:  
  \[
  L_{\text{norm}} = \frac{L - \mu}{\sigma}
  \]  
  where μ and σ are the mean and standard deviation calculated from the full dataset.  
- Denormalization is required to interpret the model’s outputs in real-world units.
