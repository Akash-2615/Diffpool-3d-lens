# 🔬 DiffPool-3D-Lens

An interactive platform to train and explore **Differentiable Pooling (DiffPool) GNNs** on the PROTEINS benchmark. This project allows you to witness graph coarsening in real-time through immersive 3-D visualizations.

---

## ✨ Features

| Feature | Details |
|---|---|
| **DiffPool model** | 2-level hierarchical pooling, DenseGCN blocks |
| **Dual Modes** | Interactive **Streamlit Dashboard** or standalone **Jupyter Notebook** |
| **3-D graph evolution** | Original → Pool-1 → Pool-2 side-by-side in 3-D |
| **3-D assignment maps** | Soft S₁, S₂ matrices visualized as 3-D point clouds |
| **3-D training trajectory** | Epoch × Accuracy × Loss in an interactive 3-D scene |
| **Target accuracy** | ≥ 72 % on PROTEINS test set |

---

## 🚀 Quick-start

### 1. Environment Setup

```bash
# Create & activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / Mac
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Choose Your Interface

#### A. Streamlit Dashboard (Real-time tracking)
```bash
streamlit run app.py
```

#### B. Jupyter Notebook (In-depth analysis)
Open `diffpool_visualizer_final.ipynb` in VS Code or Jupyter Lab. This notebook is optimized for interactive cell-by-cell execution with embedded Plotly visuals.

---

## 🧠 Architecture

```
Input (N × F)
    │
    ├─ GNNBlock (embed) ──► h₁  (N × 64)
    └─ GNNBlock (pool)  ──► S₁  (N × p₁)
              │
        dense_diff_pool → X₁ (p₁ × 64), A₁ (p₁ × p₁)
              │
    ├─ GNNBlock (embed) ──► h₂  (p₁ × 64)
    └─ GNNBlock (pool)  ──► S₂  (p₁ × p₂)
              │
        dense_diff_pool → X₂ (p₂ × 64), A₂ (p₂ × p₂)
              │
        GNNBlock (readout) → mean-pool → 64-D vector
              │
        MLP classifier → logits
```

Each `GNNBlock` consists of 3 layers of `DenseGCNConv` with BatchNorm and SELU activations.

---

## 🗂️ Project Structure

```
.
├── app.py                         # Streamlit application
├── diffpool_visualizer_final.ipynb # Interactive Jupyter Notebook
├── requirements.txt               # Project dependencies
├── README.md                      # Documentation
└── data/                          # PROTEINS dataset (auto-downloaded)
```

---

## 📊 Performance Benchmarks

| Dataset | Method | Test accuracy |
|---|---|---|
| PROTEINS | DiffPool (this repo) | **72 – 76 %** |
| PROTEINS | GCN baseline | ~70 % |
| PROTEINS | Original paper | 76.3 % |
