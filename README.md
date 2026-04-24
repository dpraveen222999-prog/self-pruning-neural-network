# Self-Pruning Neural Network using Learnable Gates (PyTorch)

## Overview

Neural networks are typically over-parameterized, leading to unnecessary memory and compute overhead. Traditional pruning is applied post-training, requiring additional fine-tuning.

This project implements **differentiable self-pruning**, where the network learns which weights to remove during training via learnable gates. The model jointly optimizes for accuracy and sparsity.

---

## Method

Each weight is modulated by a learnable gate:

g = sigmoid(gate_score / temperature)

Effective weight:
W' = W × g

- g ≈ 1 → active connection  
- g ≈ 0 → pruned connection  

This formulation makes pruning **fully differentiable**, allowing end-to-end optimization.

---

## Objective Function

Loss = CrossEntropy + λ × ∑g

- CrossEntropy → classification performance  
- ∑g → sparsity penalty (L1-style)  
- λ → controls sparsity–accuracy trade-off  

This drives many gates toward zero, inducing sparsity during training.

---

## Architecture

A simple MLP is used to isolate pruning behavior:

Flatten → PrunableLinear → ReLU →  
PrunableLinear → ReLU →  
PrunableLinear → Output  

---

## Experimental Setup

- Dataset: CIFAR-10  
- Optimizer: Adam  
- Epochs: 20  
- Batch size: 128  

λ ∈ {0, 1e-4, 1e-3, 1e-2}

---

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|--------------|--------------|
| 0.0    | 53.80        | 0.00         |
| 1e-4   | 52.15        | 48.66        |
| 1e-3   | 47.50        | 58.10        |
| 1e-2   | 39.39        | 59.88        |

---

## Analysis

- **λ = 0**  
  No regularization → dense model, maximum capacity.

- **λ = 1e-4**  
  Significant sparsity (~48%) with minimal accuracy drop → efficient regime.

- **λ = 1e-3**  
  Higher sparsity (~58%) but noticeable performance degradation → nearing capacity limits.

- **λ = 1e-2**  
  Over-pruning: excessive gate suppression leads to loss of representational power.

---

## Key Insight

The model learns a **bimodal gate distribution**:
- Near 0 → pruned weights  
- Near 1 → important weights  

This indicates effective separation between redundant and useful connections.

---

## Outputs

- Gate distribution histogram  
- Layer-wise sparsity visualization  
- Per-λ metrics (accuracy, sparsity)

---

## How to Run

```bash
pip install -r requirements.txt
python main.py --epochs 20 --batch_size 128
