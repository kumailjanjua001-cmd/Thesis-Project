# Malware vs Benign Syscall-Based Detection (Lightweight ML Model)

## Overview

This project implements a lightweight machine learning pipeline to classify **malware vs benign processes** using **system call behavior extracted via `strace` logs**.

Instead of relying on heavy static analysis or deep learning models, this approach focuses on:

- syscall frequency patterns  
- behavioral system-level features  
- lightweight classical ML models (RandomForest, SVM, Logistic Regression)

The goal is to evaluate:

> How well a lightweight ML model performs compared to industrial-grade malware detection systems under limited dataset conditions.

---

## Dataset Structure

The dataset is generated from syscall traces and stored in the following format:

```
cleaned/
│
├── benign/
│   ├── benign1.txt
│   ├── benign2.txt
│   └── ...
│
├── mirai/
│   ├── mirai1.txt
│   └── ...
│
├── gafygt/
│   ├── gafygt1.txt
│   └── ...
```

### Labels

- `benign` → normal system behavior  
- `mirai + gafygt` → combined as `malware`

---

## Features Extracted

Each sample is converted into a fixed feature vector based on syscall frequency and system behavior.

### Core Features

- Individual syscall counts (e.g., `open`, `read`, `write`, `execve`, `socket`, etc.)
- System behavior ratios:
  - `network_ratio`
  - `file_ratio`
  - `process_ratio`
  - `memory_ratio`
- Aggregated metrics:
  - `total_syscalls`
  - `unique_syscalls`

---

## Project Files

```
train_model.py          → Training + evaluation pipeline
feature_extractor.py    → Converts strace logs into feature vectors
dataset.csv             → Final extracted dataset
cleaned/                → Raw syscall logs (benign + malware)
requirements.txt        → Required Python packages
README.md               → Documentation
```

---

## Installation

### Option 1: System Installation (Ubuntu/Debian)

```bash
sudo apt install python3-pandas python3-sklearn python3-matplotlib
```

---

### Option 2: pip (if allowed)

```bash
pip install -r requirements.txt
```

If pip shows **externally-managed-environment**, use system packages instead.

---

## How to Run

### Step 1: Extract Features

```bash
python3 feature_extractor.py
```

This will:

- Read `cleaned/` folder  
- Process syscall logs  
- Generate `dataset.csv`

---

### Step 2: Train Model

```bash
python3 train_model.py
```

This will:

- Load `dataset.csv`  
- Split data into train/test (randomized each run)  
- Train multiple ML models  
- Evaluate performance  
- Save best model as:

```
malware_detector.pkl
```

---

## Model Pipeline

```
Raw strace logs
        ↓
Feature Extraction (syscall parsing)
        ↓
dataset.csv (structured features)
        ↓
Train/Test Split (randomized)
        ↓
ML Models (RF, SVM, LR)
        ↓
Evaluation (Accuracy, Precision, Recall, F1)
        ↓
Best Model Saved
```

---

## Evaluation Metrics

- **Accuracy** → Overall correctness of predictions  
- **Precision** → How many predicted malware are actually malware  
- **Recall** → How many actual malware were detected  
- **F1-score** → Balance between precision and recall  
- **Confusion Matrix** → Breakdown of correct vs incorrect predictions  

---