"""Generate draw.io / Mermaid-style architectural pipeline Graphical Abstract for CIBM.

Replicates the exact human engineering style of the user-provided reference:
- Authentic draw.io color palette:
    Tan #FFE6CC (Input / CXR data)
    Yellow #FFF2CC (Feature Backbone)
    Red/Pink #F8CECC (Bayes Shift & Failure Mode)
    Green #D5E8D4 (Analytic Mitigation Loop)
    Blue #DAE8FC (Selective Decision)
    Purple #E1D5E7 (Prospective Evaluation Layer)
- Crisp rectangular nodes with 1px borders and centered text.
- Large system containers (Hospital Edge Deployment, Recalibration Loop, Evaluation Layer).
- Orthogonal (right-angled) and straight connector lines with inline rectangular edge badges.
- Top-level feedback / validation loops (dashed lines).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configure standard fonts
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

fig = plt.figure(figsize=(16, 8.0), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

# ==============================================================================
# Header (Clean academic journal title)
# ==============================================================================
plt.figtext(
    0.5, 0.958,
    "Loss Weighting, Probability Offset, and Selective Prediction in Chest Radiography",
    ha='center', va='center',
    fontsize=15.5, fontweight='bold', color='#0F172A'
)
plt.figtext(
    0.5, 0.925,
    "Disentangling Algorithmic Loss Bias from Temporal Distribution Shift in Deep Learning Models",
    ha='center', va='center',
    fontsize=10.8, style='italic', color='#475569'
)

# Main coordinate canvas (0-100 x 0-100)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# ==============================================================================
# Color Palette Constants (Exact draw.io standard)
# ==============================================================================
COLOR_TAN_FILL     = '#FFE6CC'
COLOR_TAN_BORDER   = '#D79B00'
COLOR_YEL_FILL     = '#FFF2CC'
COLOR_YEL_BORDER   = '#D6B656'
COLOR_BLU_FILL     = '#DAE8FC'
COLOR_BLU_BORDER   = '#6C8EBF'
COLOR_GRN_FILL     = '#D5E8D4'
COLOR_GRN_BORDER   = '#82B366'
COLOR_RED_FILL     = '#F8CECC'
COLOR_RED_BORDER   = '#B85450'
COLOR_PUR_FILL     = '#E1D5E7'
COLOR_PUR_BORDER   = '#9673A6'

# Containers
CONTAINER_BLU_BG   = '#F5F9FF'
CONTAINER_BLU_BD   = '#6C8EBF'
CONTAINER_GRN_BG   = '#F2FAF2'
CONTAINER_GRN_BD   = '#82B366'
CONTAINER_PUR_BG   = '#F8F5FA'
CONTAINER_PUR_BD   = '#9673A6'

TEXT_DARK          = '#0F172A'
TEXT_MUTED         = '#334155'

# ==============================================================================
# Drawing Helper Functions
# ==============================================================================
def draw_container(x, y, w, h, title, bg, border, dashed=False, lw=1.2):
    """Draws a draw.io style grouping container with a title on top."""
    style = '--' if dashed else '-'
    box = patches.Rectangle(
        (x, y), w, h,
        edgecolor=border, facecolor=bg, linewidth=lw, linestyle=style, zorder=1
    )
    ax.add_patch(box)
    
    # Title badge at top-center
    ax.text(
        x + w/2.0, y + h - 2.0, f"■  {title}",
        ha='center', va='center', fontsize=8.6, fontweight='bold', color=border, zorder=2
    )

def draw_node(x, y, w, h, title, subtitle_lines=[], bg='#FFFFFF', border='#333333',
              title_bold=True, lw=1.1, radius=0.0):
    """Draws a crisp rectangular node with centered text."""
    if radius > 0:
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={radius}",
            edgecolor=border, facecolor=bg, linewidth=lw, zorder=3
        )
    else:
        box = patches.Rectangle(
            (x, y), w, h,
            edgecolor=border, facecolor=bg, linewidth=lw, zorder=3
        )
    ax.add_patch(box)
    
    if not subtitle_lines:
        ax.text(x + w/2.0, y + h/2.0, title, ha='center', va='center',
                fontsize=8.5, fontweight='bold' if title_bold else 'normal',
                color=TEXT_DARK, zorder=4)
    else:
        total_items = 1 + len(subtitle_lines)
        y_center = y + h/2.0
        line_height = min(2.3, (h - 2.0) / total_items)
        y_top = y_center + (total_items - 1) * (line_height / 2.0)
        
        ax.text(x + w/2.0, y_top, title, ha='center', va='center',
                fontsize=8.5, fontweight='bold' if title_bold else 'normal',
                color=TEXT_DARK, zorder=4)
        
        for idx, line in enumerate(subtitle_lines):
            curr_y = y_top - (idx + 1) * line_height
            ax.text(x + w/2.0, curr_y, line, ha='center', va='center',
                    fontsize=7.5, color=TEXT_MUTED, zorder=4)

def draw_edge_label(mx, my, label, border='#64748B'):
    """Draws draw.io style edge label badge."""
    ax.text(
        mx, my, label, ha='center', va='center', fontsize=7.2,
        color=TEXT_DARK, fontweight='medium', zorder=6,
        bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF",
                  edgecolor=border, linewidth=0.7)
    )

def draw_straight_edge(x1, y1, x2, y2, label=None, border='#475569', lw=1.1, dashed=False):
    """Draws a straight directed arrow."""
    style = '--' if dashed else '-'
    arrow = patches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=10, lw=lw, color=border,
        linestyle=style, zorder=5
    )
    ax.add_patch(arrow)
    if label:
        draw_edge_label((x1 + x2)/2.0, (y1 + y2)/2.0, label, border=border)

def draw_orthogonal_edge(pts, label=None, label_idx=None, border='#475569', lw=1.1, dashed=False):
    """Draws right-angled connector through multiple points [ (x1, y1), (x2, y2), ... ]."""
    style = '--' if dashed else '-'
    n = len(pts)
    for i in range(n - 1):
        p_start = pts[i]
        p_end = pts[i+1]
        is_last = (i == n - 2)
        if is_last:
            arrow = patches.FancyArrowPatch(
                p_start, p_end,
                arrowstyle='-|>', mutation_scale=10, lw=lw, color=border,
                linestyle=style, zorder=5
            )
            ax.add_patch(arrow)
        else:
            line = plt.Line2D(
                [p_start[0], p_end[0]], [p_start[1], p_end[1]],
                color=border, linewidth=lw, linestyle=style, zorder=5
            )
            ax.add_artist(line)
        
        target_idx = (n // 2 - 1) if label_idx is None else label_idx
        if label and i == target_idx:
            mx = (p_start[0] + p_end[0]) / 2.0
            my = (p_start[1] + p_end[1]) / 2.0
            draw_edge_label(mx, my, label, border=border)

# ==============================================================================
# 1. MAIN CONTAINER: Hospital Edge Deployment Pipeline (Blue Container)
# ==============================================================================
# Bounds: x = 3.0 to 73.0, y = 9.0 to 82.0 (height 73.0)
draw_container(
    3.0, 9.0, 70.0, 73.0,
    "Clinical AI Deployment Pipeline (Hospital Edge Device)",
    CONTAINER_BLU_BG, CONTAINER_BLU_BD, dashed=False, lw=1.3
)

# Node 1: Input CXR Data (Tan)
draw_node(
    5.0, 43.0, 10.5, 15.0,
    "Chest Radiograph\n(CXR Input)",
    ["BRAX Dataset", r"$X \in \mathbb{R}^{H \times W}$", r"Prevalence $\pi \approx 4.4\%$"],
    bg=COLOR_TAN_FILL, border=COLOR_TAN_BORDER
)

# Edge 1 -> 2
draw_straight_edge(15.5, 50.5, 19.5, 50.5, label="Image X")

# Node 2: Deep Feature Backbone (Yellow)
draw_node(
    19.5, 43.0, 11.5, 15.0,
    "Deep CNN Backbone",
    ["DenseNet-121 / ResNet", r"Weighted BCE Loss", r"$w = (1-\pi)/\pi > 1$"],
    bg=COLOR_YEL_FILL, border=COLOR_YEL_BORDER
)

# Edge 2 -> 3
draw_straight_edge(31.0, 50.5, 34.5, 50.5, label="Logits z")

# Node 3: Bayes-Optimal Bias Analysis (Red / Pink)
draw_node(
    34.5, 42.0, 12.5, 17.0,
    "Bayes-Optimal Shift",
    [
        r"$z^*(x) = \text{logit}(p) + \mathbf{\log w}$",
        r"$\Rightarrow$ Additive Intercept Bias",
        r"Effusion ($w=21.51$):",
        r"$\mathbf{+\log w = +3.07}$ Offset",
        "Severe over-confidence"
    ],
    bg=COLOR_RED_FILL, border=COLOR_RED_BORDER
)

# ------------------------------------------------------------------------------
# Fork into Two Paths:
# Path A (Upper): Conventional Temperature Scaling (Orange / Red)
# Path B (Lower): Proposed Closed-Form Offset (Green Subgraph Loop)
# ------------------------------------------------------------------------------

# Fork Upper: Node 3 -> Node 4A (Temperature Scaling)
draw_orthogonal_edge(
    [(47.0, 52.0), (49.5, 52.0), (49.5, 66.0), (51.5, 66.0)],
    label="Tuning", label_idx=1, border='#EA580C'
)

draw_node(
    51.5, 59.0, 12.0, 14.0,
    "Temperature Scaling",
    [
        r"$\hat{p}_{\mathrm{temp}} = \sigma(z / T)$",
        r"Slope scaling only",
        r"Optimizer: $T^* \approx 1.00$",
        "Fails: Intercept uncorrected"
    ],
    bg='#FFF7ED', border='#EA580C'
)

# Fork Lower: Inner Container -> Analytic Recalibration Loop (Green)
draw_container(
    49.0, 12.0, 22.5, 43.0,
    "Proposed Mitigation & Selective Decision Loop",
    CONTAINER_GRN_BG, CONTAINER_GRN_BD, dashed=True, lw=1.1
)

# Fork Lower: Node 3 -> Node 4B (Analytic Probability Offset)
draw_orthogonal_edge(
    [(47.0, 48.0), (48.2, 48.0), (48.2, 37.0), (50.5, 37.0)],
    label="Closed-Form", label_idx=1, border=COLOR_GRN_BORDER
)

# Node 4B: Analytic Probability Offset (Green)
draw_node(
    50.5, 30.0, 19.5, 14.0,
    "Analytic Probability Offset (Ours)",
    [
        r"$\hat{p}_{\mathrm{analytic}} = \sigma\left(z - \mathbf{\log w}\right)$",
        r"• Exact algebraic neutralization of $+\log w$",
        r"• Zero validation data or re-optimization",
        r"• Fully preserves discriminative AUC ranking"
    ],
    bg=COLOR_GRN_FILL, border=COLOR_GRN_BORDER
)

# Edge 4B -> Node 5 (Selective Prediction)
draw_straight_edge(60.25, 30.0, 60.25, 26.0, label="Calibrated p", border=COLOR_BLU_BORDER)

# Node 5: Selective Prediction (Light Blue)
draw_node(
    50.5, 15.0, 19.5, 11.0,
    "Selective Prediction Deployment",
    [
        r"Risk-coverage thresholding: $\hat{p}_{\mathrm{cal}} \gtrless \tau$",
        "Autonomous triage vs. Radiologist referral",
        "Guarantees safe risk coverage under drift"
    ],
    bg=COLOR_BLU_FILL, border=COLOR_BLU_BORDER
)

# ==============================================================================
# 2. EXTERNAL CONTAINER: Prospective Evaluation Layer (Purple Container)
# ==============================================================================
# Bounds: x = 77.5 to 97.0, y = 9.0 to 82.0 (height 73.0)
draw_container(
    77.5, 9.0, 19.5, 73.0,
    "Prospective Evaluation (BRAX T3, 2017)",
    CONTAINER_PUR_BG, CONTAINER_PUR_BD, dashed=False, lw=1.3
)

# Node 6A: Prospective Empirical Benchmark (White / Purple)
draw_node(
    79.0, 48.0, 16.5, 24.0,
    "Prospective Calibration\nBenchmark (T3)",
    [
        r"$N = 2,436$ radiographs (2017)",
        "Brier Score (Lower is Better):",
        "",
        r"Raw Weighted:      0.1065",
        r"Temp Scaling:      0.1059",
        r"Analytic Offset:   0.0280",
        r"Unweighted (Ref):  0.0269",
        "",
        r"$\mathbf{-74\%}$ Error Reduction",
        r"(Matches Unweighted Baseline)"
    ],
    bg='#FFFFFF', border=COLOR_PUR_BORDER
)

# Node 6B: Clinical Finding Summary (Green / Blue)
draw_node(
    79.0, 13.5, 16.5, 31.0,
    "Principal Finding",
    [
        "1. Loss-Induced Overconfidence:",
        "Positive loss weighting",
        "creates illusory temporal",
        "distribution shift.",
        "",
        "2. Exact Algebraic Remedy:",
        "Subtracting log(w) recovers",
        "baseline prospective",
        "calibration without retraining.",
        "",
        "3. Clinical Safety:",
        "Selective prediction preserves",
        "coverage guarantees under shift."
    ],
    bg=COLOR_GRN_FILL, border=COLOR_GRN_BORDER
)

# ==============================================================================
# 3. Inter-Container Connectors (Dashed lines, matching reference)
# ==============================================================================

# Connector 1: Temperature Scaling -> Prospective Benchmark
draw_straight_edge(
    63.5, 66.0, 79.0, 66.0,
    label="Upload Model", border='#EA580C', dashed=True
)

# Connector 2: Selective Prediction -> Prospective Benchmark
draw_orthogonal_edge(
    [(70.0, 20.5), (74.8, 20.5), (74.8, 54.0), (79.0, 54.0)],
    label="Deploy Model", label_idx=2, border=COLOR_BLU_BORDER, dashed=True
)

# Top Global Return Loop: Prospective Evaluation -> Training / Mitigation
# Long dashed line looping ABOVE the container boxes (at y = 85.5)
draw_orthogonal_edge(
    [(87.25, 72.0), (87.25, 86.0), (25.25, 86.0), (25.25, 58.0)],
    label="Disentangle Loss Bias from Genuine Temporal Shift",
    label_idx=1, border='#475569', lw=1.1, dashed=True
)

# ==============================================================================
# Footer
# ==============================================================================
plt.figtext(
    0.5, 0.030,
    "BRAX dataset: 41,208 patient-clustered frontal CXRs (T1/T2: 2008-2016 development cohort; T3: 2017 prospective evaluation cohort).",
    ha='center', va='center', fontsize=8.5, color='#64748B'
)

plt.savefig("manuscript_cibm/graphical_abstract.png", dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
print("Successfully generated draw.io style graphical abstract at 300 DPI.")
