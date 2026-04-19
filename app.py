"""
DiffPool GNN Visualizer — Streamlit App
Hierarchical Graph Pooling with 3D visualization of graph coarsening
Dataset: PROTEINS (TUDataset)
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG & STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiffPool GNN Visualizer",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Base — white everywhere including toolbar */
    .stApp, .main { background-color: #ffffff !important; color: #000000 !important; }
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    header[data-testid="stHeader"] * { color: #000000 !important; fill: #000000 !important; }
    [data-testid="stToolbar"] { background-color: #ffffff !important; }
    [data-testid="stDecoration"] { display: none !important; }
    .stDeployButton, [data-testid="stDeployButton"] * { color: #000000 !important; }

    section[data-testid="stSidebar"] { background-color: #f8fafc !important; border-right: 1px solid #e2e8f0; }
    section[data-testid="stSidebar"] * { color: #000000 !important; }

    /* Global text override — everything black */
    *, *::before, *::after { color: #000000 !important; }

    /* Typography */
    h1, h2, h3, h4, h5, h6 { color: #000000 !important; }
    p, span, div, label, li, a,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stText, .stCaption, .stCode,
    [data-testid="stMarkdownContainer"],pi
    [data-testid="stMarkdownContainer"] * { color: #000000 !important; }

    /* Sidebar labels and text */
    .stSlider label, .stSelectSlider label,
    .stSlider span, .stSelectSlider span { color: #000000 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #000000 !important; }
    [data-testid="metric-container"] [data-testid="stMetricDeltaIcon"] { color: #000000 !important; }

    /* Buttons — keep white text on purple bg */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 32px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button * { color: #ffffff !important; }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.35) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f1f5f9;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 10px 10px 0 0;
        gap: 4px;
        padding: 4px 4px 0 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #000000 !important;
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] * { color: #ffffff !important; }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #000000 !important;
    }
    .info-box * { color: #000000 !important; }

    /* Alerts / info banners */
    [data-testid="stAlert"] * { color: #000000 !important; }

    /* Expander */
    [data-testid="stExpander"] * { color: #000000 !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] * { color: #000000 !important; }

    /* Progress */
    .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6) !important; }

    /* Slider */
    .stSlider [data-baseweb="slider"] { color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL DEFINITION
# ─────────────────────────────────────────────

class GNNBlock(nn.Module):
    """Dense GCN block used in DiffPool (embed + pool heads)."""

    def __init__(self, in_ch: int, hid_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = DenseGCNConv(in_ch, hid_ch)
        self.conv2 = DenseGCNConv(hid_ch, hid_ch)
        self.conv3 = DenseGCNConv(hid_ch, out_ch)
        self.bn1 = nn.BatchNorm1d(hid_ch)
        self.bn2 = nn.BatchNorm1d(hid_ch)
        self.bn3 = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask=None) -> torch.Tensor:
        B, N, _ = x.size()

        def apply_bn(bn, t):
            return bn(t.view(B * N, -1)).view(B, N, -1)

        x = F.selu(apply_bn(self.bn1, self.conv1(x, adj, mask)))
        x = F.selu(apply_bn(self.bn2, self.conv2(x, adj, mask)))
        x = F.selu(apply_bn(self.bn3, self.conv3(x, adj, mask)))
        return x


class DiffPoolNet(nn.Module):
    """
    2-level DiffPool network for graph classification.
    Architecture:
        GNN(embed) + GNN(pool) → dense_diff_pool  ×2
        → GNN(readout) → MLP classifier
    """

    def __init__(self, in_features: int, num_classes: int, max_nodes: int = 100,
                 hidden: int = 64):
        super().__init__()

        # Pool sizes (roughly 25% at each level)
        p1 = max(10, max_nodes // 4)
        p2 = max(4, p1 // 4)
        self.p1, self.p2 = p1, p2

        # Level 1
        self.embed1 = GNNBlock(in_features, hidden, hidden)
        self.pool1  = GNNBlock(in_features, hidden, p1)

        # Level 2
        self.embed2 = GNNBlock(hidden, hidden, hidden)
        self.pool2  = GNNBlock(hidden, hidden, p2)

        # Readout GNN
        self.embed3 = GNNBlock(hidden, hidden, hidden)

        # Classifier
        self.cls = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Dropout(0.45),
            nn.Linear(hidden, hidden // 2),
            nn.SELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x, adj, mask=None, return_intermediates: bool = False):
        # ── Level 1 ──
        s1 = self.pool1(x, adj, mask)          # soft-assignment  [B, N0, p1]
        h1 = self.embed1(x, adj, mask)         # node embeddings   [B, N0, H]
        x1, adj1, lp1, le1 = dense_diff_pool(h1, adj, s1, mask)  # [B, p1, H]

        # ── Level 2 ──
        s2 = self.pool2(x1, adj1)
        h2 = self.embed2(x1, adj1)
        x2, adj2, lp2, le2 = dense_diff_pool(h2, adj1, s2)       # [B, p2, H]

        # ── Readout ──
        h3   = self.embed3(x2, adj2)
        read = h3.mean(dim=1)                  # global mean pool
        logits = self.cls(read)

        aux = lp1 + le1 + lp2 + le2

        if return_intermediates:
            return F.log_softmax(logits, dim=-1), aux, {
                "x0": x,  "adj0": adj,
                "x1": x1, "adj1": adj1, "s1": s1,
                "x2": x2, "adj2": adj2, "s2": s2,
            }
        return F.log_softmax(logits, dim=-1), aux


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_proteins(max_nodes: int = 100):
    """Download & preprocess PROTEINS dataset (cached)."""
    dataset = TUDataset(root="data/TUDataset", name="PROTEINS")

    # Filter out very large graphs
    filtered = [d for d in dataset if d.num_nodes <= max_nodes]

    # Reproducible split
    torch.manual_seed(42)
    perm = torch.randperm(len(filtered))
    n_train = int(0.8 * len(filtered))
    train_data = [filtered[i] for i in perm[:n_train]]
    test_data  = [filtered[i] for i in perm[n_train:]]

    actual_max = max(d.num_nodes for d in filtered)
    n_feat     = filtered[0].num_node_features
    n_cls      = dataset.num_classes

    return train_data, test_data, actual_max, n_feat, n_cls


def dense_batch(data_list, max_nodes: int, device):
    """Convert a list of PyG Data objects → dense tensors."""
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(data_list)
    x, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=max_nodes)
    adj      = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=max_nodes)
    y        = batch.y
    return x.to(device), adj.to(device), mask.to(device), y.to(device)


# ─────────────────────────────────────────────
#  TRAINING HELPERS
# ─────────────────────────────────────────────

def train_one_epoch(model, data, optimizer, max_nodes, device, bs=32):
    model.train()
    perm = torch.randperm(len(data))
    data = [data[i] for i in perm]

    tot_loss = correct = total = 0
    for i in range(0, len(data), bs):
        chunk = data[i : i + bs]
        x, adj, mask, y = dense_batch(chunk, max_nodes, device)

        optimizer.zero_grad()
        out, aux = model(x, adj, mask)
        cls_loss  = F.nll_loss(out, y)
        loss      = cls_loss + 0.5 * aux          # aux weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        tot_loss += loss.item() * len(chunk)
        correct  += (out.argmax(-1) == y).sum().item()
        total    += len(chunk)

    return tot_loss / total, correct / total


@torch.no_grad()
def evaluate(model, data, max_nodes, device, bs=32):
    model.eval()
    correct = total = 0
    for i in range(0, len(data), bs):
        chunk = data[i : i + bs]
        x, adj, mask, y = dense_batch(chunk, max_nodes, device)
        out, _ = model(x, adj, mask)
        correct += (out.argmax(-1) == y).sum().item()
        total   += len(chunk)
    return correct / total


# ─────────────────────────────────────────────
#  3-D VISUALIZATION HELPERS
# ─────────────────────────────────────────────

DARK_BG   = "#ffffff"
PANEL_BG  = "#f8fafc"
GRID_COL  = "#d1d5db"
BLK       = "#000000"   # single source of truth for all text

# ── Reusable axis style dicts ────────────────────────────────────────────────

def _ax2d(title=""):
    """Standard black-text 2-D axis."""
    return dict(
        title=dict(text=title, font=dict(color=BLK, size=12)),
        color=BLK,
        tickfont=dict(color=BLK, size=11),
        gridcolor=GRID_COL,
        linecolor=BLK,
        zerolinecolor=GRID_COL,
    )

def _ax3d(title="", show_ticks=True):
    """Standard black-text 3-D scene axis."""
    return dict(
        title=dict(text=title, font=dict(color=BLK, size=12)),
        tickfont=dict(color=BLK, size=10),
        showticklabels=show_ticks,
        showgrid=True,
        gridcolor=GRID_COL,
        showline=True,
        linecolor=BLK,
        zerolinecolor=GRID_COL,
    )

def _colorbar(title=""):
    """Colorbar with black tick labels and title."""
    return dict(
        thickness=10,
        title=dict(text=title, font=dict(color=BLK, size=11)) if title else None,
        tickfont=dict(color=BLK, size=10),
        outlinecolor=GRID_COL,
        outlinewidth=1,
    )

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(color=BLK, size=11, family="Arial, sans-serif"),
    margin=dict(l=10, r=10, t=45, b=10),
    title_font=dict(color=BLK, size=13),
)

# ────────────────────────────────────────────────────────────────────────────

def spectral_3d(adj: np.ndarray) -> np.ndarray:
    """3-D spectral layout via normalised Laplacian eigenvectors."""
    n = adj.shape[0]
    if n <= 1:
        return np.random.randn(n, 3) * 0.01

    deg = adj.sum(1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-9)), 0.0)
    D = np.diag(d_inv_sqrt)
    L = np.eye(n) - D @ adj @ D

    vals, vecs = np.linalg.eigh(L)
    k  = min(4, n)
    pos = vecs[:, 1:k]

    if pos.shape[1] < 3:
        pad = np.random.randn(n, 3 - pos.shape[1]) * 0.05
        pos = np.hstack([pos, pad])

    std = pos.std(0) + 1e-8
    pos = (pos - pos.mean(0)) / std
    return pos[:, :3]


def make_3d_graph_fig(adj: np.ndarray, node_colors, title: str,
                      colorscale: str = "Viridis", showscale: bool = True):
    """Return a Plotly Figure with a 3-D node-link diagram."""
    n   = adj.shape[0]
    pos = spectral_3d(adj)

    # Edges
    ex, ey, ez = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.05:
                ex += [pos[i, 0], pos[j, 0], None]
                ey += [pos[i, 1], pos[j, 1], None]
                ez += [pos[i, 2], pos[j, 2], None]

    edges = go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(color="rgba(99,102,241,0.30)", width=1.5),
        hoverinfo="none",
        showlegend=False,
    )

    sizes = np.full(n, 8)
    sizes[0] = 12

    nodes = go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="markers",
        marker=dict(
            size=sizes.tolist(),
            color=node_colors,
            colorscale=colorscale,
            showscale=showscale,
            colorbar=_colorbar(),
            line=dict(color="rgba(0,0,0,0.3)", width=0.5),
            opacity=0.92,
        ),
        hovertext=[f"node {i}" for i in range(n)],
        hoverinfo="text",
        showlegend=False,
    )

    scene = dict(
        bgcolor=DARK_BG,
        # no tick labels on graph nodes — keep clean
        xaxis=dict(showgrid=False, showticklabels=False, showline=False,
                   title=dict(text="", font=dict(color=BLK))),
        yaxis=dict(showgrid=False, showticklabels=False, showline=False,
                   title=dict(text="", font=dict(color=BLK))),
        zaxis=dict(showgrid=False, showticklabels=False, showline=False,
                   title=dict(text="", font=dict(color=BLK))),
        camera=dict(eye=dict(x=1.6, y=1.0, z=0.9)),
    )

    fig = go.Figure([edges, nodes])
    fig.update_layout(
        title=dict(text=title, font=dict(color=BLK, size=13)),
        scene=scene,
        height=370,
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_graph_evolution(intermediates: dict, idx: int = 0):
    """Render 3 side-by-side 3-D graphs showing DiffPool coarsening."""
    adj0 = intermediates["adj0"][idx].numpy()
    adj1 = intermediates["adj1"][idx].numpy()
    adj2 = intermediates["adj2"][idx].numpy()
    s1   = torch.softmax(intermediates["s1"][idx], dim=-1).numpy()

    n0 = int((adj0.sum(1) > 0).sum())
    n1 = adj1.shape[0]
    n2 = adj2.shape[0]
    cluster_assign = s1[:n0].argmax(-1).tolist()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(make_3d_graph_fig(
            adj0[:n0, :n0], node_colors=cluster_assign,
            title=f"🔵 Original  ({n0} nodes)", colorscale="Turbo",
        ), use_container_width=True)
    with c2:
        st.plotly_chart(make_3d_graph_fig(
            adj1[:n1, :n1], node_colors=list(range(n1)),
            title=f"🟣 After Pool-1  ({n1} super-nodes)", colorscale="Viridis",
        ), use_container_width=True)
    with c3:
        st.plotly_chart(make_3d_graph_fig(
            adj2[:n2, :n2], node_colors=list(range(n2)),
            title=f"🔴 After Pool-2  ({n2} super-nodes)", colorscale="Plasma",
        ), use_container_width=True)


def plot_3d_trajectory(history: dict):
    """3-D scatter: epoch × train-acc × loss — all text black."""
    df = pd.DataFrame(history)

    fig = go.Figure(go.Scatter3d(
        x=df["epoch"],
        y=df["train_acc"],
        z=df["train_loss"],
        mode="lines+markers",
        marker=dict(
            size=5,
            color=df["epoch"],
            colorscale="Viridis",
            showscale=True,
            colorbar=_colorbar("Epoch"),
        ),
        line=dict(color="#6366f1", width=2),
        hovertemplate=(
            "<b>Epoch:</b> %{x}<br>"
            "<b>Train Acc:</b> %{y:.1f}%<br>"
            "<b>Loss:</b> %{z:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="3-D Training Trajectory", font=dict(color=BLK, size=14)),
        scene=dict(
            bgcolor=DARK_BG,
            xaxis=_ax3d("Epoch"),
            yaxis=_ax3d("Train Acc %"),
            zaxis=_ax3d("Loss"),
        ),
        height=520,
        **PLOTLY_LAYOUT,
    )
    return fig


def plot_3d_assignment(s_mat: np.ndarray, title: str, colorscale="Viridis"):
    """3-D scatter of soft-assignment matrix — all text black."""
    n_orig, n_super = s_mat.shape
    x_idx, y_idx, z_val = [], [], []
    for i in range(n_orig):
        for j in range(n_super):
            x_idx.append(j)
            y_idx.append(i)
            z_val.append(float(s_mat[i, j]))

    fig = go.Figure(go.Scatter3d(
        x=x_idx, y=y_idx, z=z_val,
        mode="markers",
        marker=dict(
            size=4,
            color=z_val,
            colorscale=colorscale,
            showscale=True,
            colorbar=_colorbar("Weight"),
            opacity=0.85,
        ),
        hovertemplate=(
            "<b>Super-node:</b> %{x}<br>"
            "<b>Orig node:</b> %{y}<br>"
            "<b>Weight:</b> %{z:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=BLK, size=13)),
        scene=dict(
            bgcolor=DARK_BG,
            xaxis=_ax3d("Super-node"),
            yaxis=_ax3d("Orig node"),
            zaxis=_ax3d("Assignment"),
        ),
        height=440,
        **PLOTLY_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────

def init_state():
    defaults = dict(
        trained=False,
        history={"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []},
        intermediates=None,
        final_test_acc=0.0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Hyper-parameters")
        epochs    = st.slider("Epochs",        20, 300, 120, 10)
        lr        = st.select_slider("Learning rate", [5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
        bs        = st.slider("Batch size",    8, 64, 32, 8)
        hidden    = st.select_slider("Hidden dim", [32, 64, 128], value=64)
        max_nodes = st.slider("Max nodes (filter)", 50, 200, 100, 10)

        st.markdown("---")
        st.markdown("**Dataset :** PROTEINS")
        st.markdown("**Task :** Binary graph classification")
        st.markdown("**Method :** DiffPool (2 levels)")
        st.markdown("**Target :** ≥ 65 % test accuracy")

        st.markdown("---")
        st.markdown("""
<div style='font-size:0.78rem; color:#475569'>
<b>DiffPool</b> learns hierarchical graph representations
by softly assigning nodes to super-nodes at each layer.
The assignment matrices <i>S₁</i>, <i>S₂</i> are trained
end-to-end alongside GNN embeddings.
</div>
""", unsafe_allow_html=True)

    return epochs, lr, bs, hidden, max_nodes


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    init_state()

    # ── Header ──
    st.markdown("""
<h1 style='font-size:2.2rem; font-weight:900;
   background:linear-gradient(90deg,#6366f1,#a78bfa,#ec4899);
   -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
🔬 DiffPool GNN — 3-D Graph Evolution Visualizer
</h1>
<p style='color:#000000; margin-top:-12px; font-size:0.95rem;'>
Hierarchical graph pooling · PROTEINS dataset · Real-time training dashboard
</p>
""", unsafe_allow_html=True)

    epochs, lr, bs, hidden, max_nodes = sidebar()

    # ── Tabs ──
    tab_train, tab_evo, tab_assign, tab_analytics = st.tabs([
        "🚀 Training",
        "🧬 3-D Graph Evolution",
        "🗺️ Assignment Maps",
        "📊 Analytics",
    ])

    # ════════════════════════════════════════
    #  TAB 1 — TRAINING
    # ════════════════════════════════════════
    with tab_train:
        st.markdown("""
<div class='info-box'>
Train a 2-level DiffPool GNN on the <b>PROTEINS</b> benchmark (graph classification, 2 classes).
Each epoch the live accuracy and loss curves update below.
After training, explore the <b>Graph Evolution</b> tab to see how DiffPool coarsens graphs in 3-D.
</div>
""", unsafe_allow_html=True)

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            run = st.button("🚀 Start Training", type="primary")

        if run:
            # ─ Load data ─
            status_ph = col_status.empty()
            status_ph.info("⏳ Downloading PROTEINS dataset …")

            train_data, test_data, act_max, n_feat, n_cls = load_proteins(max_nodes)
            status_ph.success(
                f"✅ Loaded — {len(train_data)} train / {len(test_data)} test  |  "
                f"{n_feat} features  |  {n_cls} classes  |  max nodes {act_max}"
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model     = DiffPoolNet(n_feat, n_cls, act_max, hidden).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            st.caption(f"Model · {n_params:,} trainable parameters · running on **{device}**")

            # ─ UI placeholders ─
            progress  = st.progress(0)
            ep_text   = st.empty()

            m1, m2, m3, m4 = st.columns(4)
            ph_loss   = m1.empty()
            ph_tr_acc = m2.empty()
            ph_te_acc = m3.empty()
            ph_best   = m4.empty()

            chart_ph = st.empty()

            history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}
            best_acc = 0.0

            for ep in range(1, epochs + 1):
                t0 = time.time()
                tr_loss, tr_acc = train_one_epoch(model, train_data, optimizer, act_max, device, bs)
                te_acc          = evaluate(model, test_data, act_max, device, bs)
                scheduler.step()

                history["epoch"].append(ep)
                history["train_loss"].append(round(tr_loss, 5))
                history["train_acc"].append(round(tr_acc * 100, 2))
                history["test_acc"].append(round(te_acc * 100, 2))

                best_acc = max(best_acc, te_acc * 100)

                progress.progress(ep / epochs)
                ep_text.caption(
                    f"Epoch {ep}/{epochs}  ·  {time.time()-t0:.1f}s  ·  "
                    f"LR {scheduler.get_last_lr()[0]:.2e}"
                )

                ph_loss.metric("Train Loss",  f"{tr_loss:.4f}")
                ph_tr_acc.metric("Train Acc",  f"{tr_acc*100:.1f}%")
                ph_te_acc.metric("Test Acc",   f"{te_acc*100:.1f}%")
                ph_best.metric("Best Test Acc",f"{best_acc:.1f}%",
                               delta="✅ ≥65%" if best_acc >= 65 else "")

                # Update chart every 3 epochs
                if ep % 3 == 0 or ep == epochs:
                    df  = pd.DataFrame(history)
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Training Loss", "Accuracy (%)"),
                    )
                    fig.add_trace(go.Scatter(
                        x=df.epoch, y=df.train_loss, name="Train Loss",
                        line=dict(color="#6366f1", width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.epoch, y=df.train_acc, name="Train Acc",
                        line=dict(color="#10b981", width=2)), row=1, col=2)
                    fig.add_trace(go.Scatter(
                        x=df.epoch, y=df.test_acc, name="Test Acc",
                        line=dict(color="#f59e0b", width=2, dash="dash")), row=1, col=2)
                    fig.add_hline(y=65, row=1, col=2,
                                  line=dict(color="#ef4444", dash="dot", width=1.5),
                                  annotation_text="65 % target",
                                  annotation_font_color="#ef4444")
                    fig.update_layout(
                        height=320,
                        legend=dict(bgcolor=PANEL_BG, font=dict(color=BLK)),
                        **PLOTLY_LAYOUT,
                    )
                    # Force all subplot title annotations to black
                    for ann in fig.layout.annotations:
                        ann.font.color = BLK
                    fig.update_xaxes(**_ax2d("Epoch"))
                    fig.update_yaxes(**_ax2d())
                    chart_ph.plotly_chart(fig, use_container_width=True)

            # ─ Capture intermediates on a training mini-batch ─
            model.eval()
            sample = train_data[:8]
            with torch.no_grad():
                xs, adjs, masks, _ = dense_batch(sample, act_max, device)
                _, _, inter = model(xs, adjs, masks, return_intermediates=True)

            inter_cpu = {k: v.detach().cpu() for k, v in inter.items()}

            # ─ Save to session state ─
            st.session_state.trained         = True
            st.session_state.history         = history
            st.session_state.intermediates   = inter_cpu
            st.session_state.final_test_acc  = history["test_acc"][-1]
            st.session_state.best_acc        = best_acc

            # ─ Result banner ─
            if best_acc >= 65:
                st.success(
                    f"🎉 **Target reached!** Best test accuracy = **{best_acc:.1f}%** (≥ 65 %). "
                    "Switch to the **Graph Evolution** tab to explore DiffPool in 3-D!"
                )
            else:
                st.warning(
                    f"⚠️ Best test accuracy = {best_acc:.1f}%. "
                    "Try increasing epochs, using lr=0.001, or widening the hidden dim."
                )

    # ════════════════════════════════════════
    #  TAB 2 — 3-D GRAPH EVOLUTION
    # ════════════════════════════════════════
    with tab_evo:
        st.markdown("""
<div class='info-box'>
<b>How to read this:</b>
· 🔵 Original graph — node colours show which super-node each node is assigned to after Pool-1<br>
· 🟣 After Pool-1 — compressed graph (~25 % of original nodes)<br>
· 🔴 After Pool-2 — further compressed (~6 % of original nodes)<br>
Rotate each 3-D graph with your mouse. Hover nodes for details.
</div>
""", unsafe_allow_html=True)

        if not st.session_state.trained:
            st.info("👆 Train the model first (Training tab).")
        else:
            inter = st.session_state.intermediates
            n_samples = inter["adj0"].shape[0]
            idx = st.slider("Graph sample index", 0, n_samples - 1, 0)

            plot_graph_evolution(inter, idx)

            # ─ Compression stats ─
            adj0 = inter["adj0"][idx].numpy()
            adj1 = inter["adj1"][idx].numpy()
            adj2 = inter["adj2"][idx].numpy()
            n0 = int((adj0.sum(1) > 0).sum())
            n1 = adj1.shape[0]
            n2 = adj2.shape[0]

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original nodes",  str(n0))
            c2.metric("After Pool-1",    str(n1), delta=f"−{n0-n1} nodes")
            c3.metric("After Pool-2",    str(n2), delta=f"−{n1-n2} nodes")
            c4.metric("Compression",     f"{100*(1-n2/n0):.0f}%")

    # ════════════════════════════════════════
    #  TAB 3 — 3-D ASSIGNMENT MAPS
    # ════════════════════════════════════════
    with tab_assign:
        st.markdown("""
<div class='info-box'>
<b>Soft assignment matrices</b> (S₁, S₂) show the learned probabilistic mapping
from nodes to super-nodes.  Each point in the 3-D scatter represents one
(orig-node, super-node, weight) triple.  Brighter = stronger assignment.
</div>
""", unsafe_allow_html=True)

        if not st.session_state.trained:
            st.info("👆 Train the model first (Training tab).")
        else:
            inter = st.session_state.intermediates
            n_samples = inter["s1"].shape[0]
            idx = st.slider("Sample index", 0, n_samples - 1, 0, key="assign_slider")

            col_a, col_b = st.columns(2)

            with col_a:
                s1 = torch.softmax(inter["s1"][idx], dim=-1).numpy()
                n_show = min(40, s1.shape[0])
                fig = plot_3d_assignment(
                    s1[:n_show],
                    title="S₁ — Original → Pool-1",
                    colorscale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                s2 = torch.softmax(inter["s2"][idx], dim=-1).numpy()
                fig = plot_3d_assignment(
                    s2,
                    title="S₂ — Pool-1 → Pool-2",
                    colorscale="Plasma",
                )
                st.plotly_chart(fig, use_container_width=True)

            # 2-D heatmaps
            st.markdown("### 2-D Assignment Heatmaps")
            ch1, ch2 = st.columns(2)
            with ch1:
                fig2 = go.Figure(go.Heatmap(
                    z=s1[:n_show], colorscale="Viridis", showscale=True,
                    colorbar=_colorbar(),
                ))
                fig2.update_layout(
                    title=dict(text="S₁ heatmap", font=dict(color=BLK, size=13)),
                    xaxis=_ax2d("Super-node"),
                    yaxis=_ax2d("Orig node"),
                    height=320, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig2, use_container_width=True)

            with ch2:
                fig2 = go.Figure(go.Heatmap(
                    z=s2, colorscale="Plasma", showscale=True,
                    colorbar=_colorbar(),
                ))
                fig2.update_layout(
                    title=dict(text="S₂ heatmap", font=dict(color=BLK, size=13)),
                    xaxis=_ax2d("Super-node"),
                    yaxis=_ax2d("Pool-1 node"),
                    height=320, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig2, use_container_width=True)

    # ════════════════════════════════════════
    #  TAB 4 — ANALYTICS
    # ════════════════════════════════════════
    with tab_analytics:
        if not st.session_state.trained:
            st.info("👆 Train the model first (Training tab).")
        else:
            hist = st.session_state.history
            best = st.session_state.get("best_acc", 0.0)

            # ─ KPI row ─
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Best Test Acc",  f"{best:.1f}%",
                      delta="✅ Target met" if best >= 65 else "❌ Below target")
            k2.metric("Final Test Acc", f"{hist['test_acc'][-1]:.1f}%")
            k3.metric("Final Train Loss", f"{hist['train_loss'][-1]:.4f}")
            k4.metric("Epochs",         len(hist["epoch"]))

            st.markdown("---")

            # ─ 2-D curves ─
            df = pd.DataFrame(hist)

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=df.epoch, y=df.train_acc, name="Train Acc",
                fill="tozeroy", fillcolor="rgba(99,102,241,0.12)",
                line=dict(color="#6366f1", width=2)))
            fig_acc.add_trace(go.Scatter(
                x=df.epoch, y=df.test_acc, name="Test Acc",
                line=dict(color="#f59e0b", width=2, dash="dash")))
            fig_acc.add_hline(y=65, line=dict(color="#ef4444", dash="dot"),
                              annotation_text="65% target",
                              annotation_font_color="#ef4444")
            fig_acc.update_layout(
                title=dict(text="Accuracy Curves", font=dict(color=BLK, size=14)),
                xaxis=_ax2d("Epoch"),
                yaxis=_ax2d("Accuracy (%)"),
                legend=dict(bgcolor=PANEL_BG, font=dict(color=BLK),
                            bordercolor=GRID_COL, borderwidth=1),
                height=360, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # ─ 3-D training trajectory ─
            st.markdown("### 3-D Training Trajectory")
            st.plotly_chart(plot_3d_trajectory(hist), use_container_width=True)

            # ─ Loss curve ─
            fig_loss = go.Figure(go.Scatter(
                x=df.epoch, y=df.train_loss, name="Train Loss",
                fill="tozeroy", fillcolor="rgba(168,85,247,0.12)",
                line=dict(color="#a855f7", width=2),
            ))
            fig_loss.update_layout(
                title=dict(text="Training Loss", font=dict(color=BLK, size=14)),
                xaxis=_ax2d("Epoch"),
                yaxis=_ax2d("Loss"),
                height=300, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            # ─ Raw table ─
            with st.expander("📋 Raw training log"):
                st.dataframe(df.style.format(
                    {"train_loss": "{:.4f}", "train_acc": "{:.1f}", "test_acc": "{:.1f}"}
                ), use_container_width=True)


if __name__ == "__main__":
    main()