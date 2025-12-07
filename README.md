# Amazon ML Challenge - Product Price Prediction 🏷️

This repository contains a machine learning solution for predicting product prices based on embedding features. The solution utilizes a **Weighted Ensemble** approach combining a custom **PyTorch Transformer**, **LightGBM**, and **XGBoost**.

## 🚀 Project Overview

The goal of this project is to predict the price of items given their embedding features. The pipeline involves loading reduced embeddings, preprocessing data, training a Deep Learning model, and ensembling it with gradient boosting models.

**Key Performance Metric:**
* **Validation SMAPE:** 55.918%

## 🧠 Model Architecture

The core of this solution uses a multi-modal approach:

### 1. Custom Multi-Modal Transformer (PyTorch)
A neural network built using `torch.nn.TransformerEncoder`. It treats the input embeddings as a sequence to capture complex relationships.
* **Input Dimension:** 898 features
* **Embedding Dimension:** 512
* **Attention Heads:** 4
* **Transformer Layers:** 2
* **Feedforward Dimension:** 1024
* **Optimization:** AdamW with ReduceLROnPlateau scheduler.

### 2. Gradient Boosting Models
The project leverages pre-trained models loaded from disk:
* **LightGBM** (`lgb_model_full_lowlevel.joblib`)
* **XGBoost** (`xgb_model_full_lowlevel.json`)

### 3. Ensemble Strategy
The final prediction is a weighted average of the three models to improve robustness:
$$P_{final} = 0.5 \times P_{Transformer} + 0.25 \times P_{LGBM} + 0.25 \times P_{XGBoost}$$


## 📂 Data Requirements

The notebook expects the following files to be present (or mounted via Google Drive):
* `X_train_reduced.npy` / `X_test_reduced.npy` (Input embeddings)
* `y_train.npy` (Target prices)
* `sample_ids.npy` (For submission mapping)
* `lgb_model_full_lowlevel.joblib` (Pre-trained LightGBM)
* `xgb_model_full_lowlevel.json` (Pre-trained XGBoost)

## 🛠️ Dependencies

The project runs on Python and requires **GPU acceleration** (CUDA).

```bash
pip install torch torchvision transformers lightgbm xgboost tqdm joblib pandas numpy scikit-learn
