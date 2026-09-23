"""Publication Graphical Abstract for CIBM / Elsevier (Final Masterpiece).
Conforms to Elsevier Graphical Abstract specifications:
- Readable at approximately 5 x 13 cm at standard screen/print sizes.
- Aspect ratio ~ 2.6 : 1 (13 cm x 5 cm).
- High-resolution 300 DPI (3900 x 1500 pixels).
- Story at a glance:
  [ Weighted BCE ]  -->  +log w  -->  [ Probability Inflation ]  -->  z - log w  -->  [ Calibration Restored ]
  Key values: Brier: .108 -> .028 (-74%)
  Small BRAX -> MIMIC validation arrow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Typography
plt.rcParams['mathtext.fontset'] = 'stixsans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']

# Figure setup: exactly 13.0 x 5.0 inches @ 300 DPI (13 cm x 5 cm aspect ratio: 2.6 : 1)
fig = plt.figure(figsize=(13.0, 5.0), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Outer clean card
bg_border = patches.FancyBboxPatch(
    (0.6, 1.0), 98.8, 98.0,
    boxstyle="round,pad=0.0,rounding_size=1.2",
    facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5, zorder=0
)
ax.add_patch(bg_border)

# Geometry
box_y = 22.0
box_h = 73.5
box_w = 25.2
gap = 10.0

c1_x = 2.2
c2_x = c1_x + box_w + gap  # 37.4
c3_x = c2_x + box_w + gap  # 72.6

# Helper to draw rounded cards
def draw_card(x, y, w, h, bg, border, lw=2.2, radius=1.2):
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=bg, edgecolor=border, linewidth=lw, zorder=2
    )
    ax.add_patch(card)
    return card

# Helper to draw step pills
def draw_step_pill(x, y, text, bg, fg, border):
    ax.text(
        x, y, text,
        ha='center', va='center', fontsize=9.2, fontweight='bold',
        color=fg, zorder=4,
        bbox=dict(boxstyle="round,pad=0.36,rounding_size=0.4",
                  facecolor=bg, edgecolor=border, linewidth=1.1)
    )

# ==============================================================================
# CARD 1: Weighted BCE (Subtle Slate / Steel)
# ==============================================================================
draw_card(c1_x, box_y, box_w, box_h, bg="#FFFFFF", border="#314D68", lw=1.8)

# Step pill
draw_step_pill(c1_x + box_w/2.0, box_y + box_h - 5.0, "STEP 1 • MODEL TRAINING", bg="#F0F4F8", fg="#314D68", border="#B0BAC4")

# Main Title
ax.text(c1_x + box_w/2.0, box_y + box_h - 15.0, "Weighted BCE",
        ha='center', va='center', fontsize=23.5, fontweight='bold', color="#102A43", zorder=3)

# Subtitle
ax.text(c1_x + box_w/2.0, box_y + box_h - 22.5, "Class Imbalance Training",
        ha='center', va='center', fontsize=12.2, fontweight='bold', color="#486581", zorder=3)

# Inner Math Container
math_c1 = patches.FancyBboxPatch(
    (c1_x + 1.6, box_y + 25.5), box_w - 3.2, 21.0,
    boxstyle="round,pad=0.0,rounding_size=0.8",
    facecolor="#F0F4F8", edgecolor="#B0BAC4", linewidth=1.2, zorder=3
)
ax.add_patch(math_c1)

ax.text(c1_x + box_w/2.0, box_y + 40.0, r"$\mathcal{L}_w = -w \cdot y \log p - (1-y)\log(1-p)$",
        ha='center', va='center', fontsize=11.0, color="#102A43", zorder=4)
ax.text(c1_x + box_w/2.0, box_y + 31.5, r"Positive Weight: $w = \frac{1-\pi}{\pi} > 1$",
        ha='center', va='center', fontsize=11.2, fontweight='bold', color="#0D47A1", zorder=4)

# 2 Large, readable bullets
ax.text(c1_x + 2.2, box_y + 16.0, r"• Severe class imbalance ($\pi \approx 4.4\%$)",
        ha='left', va='center', fontsize=10.2, color="#314D68", zorder=3)
ax.text(c1_x + 2.2, box_y + 8.5, "• Penalizes false negatives on pathology",
        ha='left', va='center', fontsize=10.2, color="#314D68", zorder=3)


# ==============================================================================
# TRANSITION 1: +log w (Subtle Cool Blue)
# ==============================================================================
t1_mid_x = c1_x + box_w + gap/2.0  # 32.4
t1_mid_y = box_y + box_h/2.0 + 1.0

# Inflow line (Box 1 -> Badge)
ax.plot([c1_x + box_w + 0.4, t1_mid_x - 3.1], [t1_mid_y, t1_mid_y],
        color="#1976D2", lw=2.4, zorder=3)

# Outflow arrow (Badge -> Box 2)
arr1 = patches.FancyArrowPatch(
    (t1_mid_x + 3.1, t1_mid_y), (c2_x - 0.6, t1_mid_y),
    arrowstyle='-|>', mutation_scale=16, lw=2.4, color="#1976D2", zorder=3
)
ax.add_patch(arr1)

# Badge 1 (+log w)
b1 = patches.FancyBboxPatch(
    (t1_mid_x - 3.1, t1_mid_y - 9.2), 6.2, 18.4,
    boxstyle="round,pad=0.0,rounding_size=0.8",
    facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=1.5, zorder=5
)
ax.add_patch(b1)

ax.text(t1_mid_x, t1_mid_y + 4.2, r"$\mathbf{+\log w}$",
        ha='center', va='center', fontsize=16.0, fontweight='bold', color="#0D47A1", zorder=6)
ax.text(t1_mid_x, t1_mid_y - 4.2, "Logit\nShift",
        ha='center', va='center', fontsize=9.2, fontweight='bold', color="#0D47A1", zorder=6)


# ==============================================================================
# CARD 2: Probability Inflation (Subtle Slate / Steel)
# ==============================================================================
draw_card(c2_x, box_y, box_w, box_h, bg="#FFFFFF", border="#314D68", lw=1.8)

# Step pill
draw_step_pill(c2_x + box_w/2.0, box_y + box_h - 5.0, "STEP 2 • FAILURE MODE", bg="#F0F4F8", fg="#314D68", border="#B0BAC4")

# Main Title
ax.text(c2_x + box_w/2.0, box_y + box_h - 15.0, "Probability Inflation",
        ha='center', va='center', fontsize=22.5, fontweight='bold', color="#102A43", zorder=3)

# Subtitle
ax.text(c2_x + box_w/2.0, box_y + box_h - 22.5, "Illusory Temporal Calibration Drift",
        ha='center', va='center', fontsize=12.2, fontweight='bold', color="#486581", zorder=3)

# Inner Math Container
math_c2 = patches.FancyBboxPatch(
    (c2_x + 1.6, box_y + 25.5), box_w - 3.2, 21.0,
    boxstyle="round,pad=0.0,rounding_size=0.8",
    facecolor="#F0F4F8", edgecolor="#B0BAC4", linewidth=1.2, zorder=3
)
ax.add_patch(math_c2)

ax.text(c2_x + box_w/2.0, box_y + 40.0, r"$z^*(x) = \mathrm{logit}(p) + \mathbf{\log w}$",
        ha='center', va='center', fontsize=12.8, fontweight='bold', color="#102A43", zorder=4)
ax.text(c2_x + box_w/2.0, box_y + 31.5, r"Effusion ($w = 21.5$): $\mathbf{\Delta z \approx +3.07}$ Bias",
        ha='center', va='center', fontsize=11.0, fontweight='bold', color="#0D47A1", zorder=4)

# 2 Large, readable bullets
ax.text(c2_x + 2.2, box_y + 16.0, "• Severe clinical overconfidence in risks",
        ha='left', va='center', fontsize=10.2, color="#314D68", zorder=3)
ax.text(c2_x + 2.2, box_y + 8.5, r"• Temp scaling fails ($T^* \approx 1$, no shift)",
        ha='left', va='center', fontsize=10.2, fontweight='bold', color="#314D68", zorder=3)


# ==============================================================================
# TRANSITION 2: z - log w (Subtle Cool Blue)
# ==============================================================================
t2_mid_x = c2_x + box_w + gap/2.0  # 67.6
t2_mid_y = box_y + box_h/2.0 + 1.0

# Inflow line (Box 2 -> Badge)
ax.plot([c2_x + box_w + 0.4, t2_mid_x - 3.3], [t2_mid_y, t2_mid_y],
        color="#1976D2", lw=2.4, zorder=3)

# Outflow arrow (Badge -> Box 3)
arr2 = patches.FancyArrowPatch(
    (t2_mid_x + 3.3, t2_mid_y), (c3_x - 0.6, t2_mid_y),
    arrowstyle='-|>', mutation_scale=16, lw=2.4, color="#1976D2", zorder=3
)
ax.add_patch(arr2)

# Badge 2 (z - log w)
b2 = patches.FancyBboxPatch(
    (t2_mid_x - 3.3, t2_mid_y - 9.2), 6.6, 18.4,
    boxstyle="round,pad=0.0,rounding_size=0.8",
    facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=1.5, zorder=5
)
ax.add_patch(b2)

ax.text(t2_mid_x, t2_mid_y + 4.2, r"$\mathbf{z - \log w}$",
        ha='center', va='center', fontsize=15.5, fontweight='bold', color="#0D47A1", zorder=6)
ax.text(t2_mid_x, t2_mid_y - 4.2, "Analytic\nOffset",
        ha='center', va='center', fontsize=9.2, fontweight='bold', color="#0D47A1", zorder=6)


# ==============================================================================
# CARD 3: Calibration Restored (Subtle Slate / Cohesive Light Box)
# ==============================================================================
draw_card(c3_x, box_y, box_w, box_h, bg="#FFFFFF", border="#314D68", lw=1.8)

# Step pill
draw_step_pill(c3_x + box_w/2.0, box_y + box_h - 5.0, "STEP 3 • EXACT RECOVERY", bg="#F0F4F8", fg="#314D68", border="#B0BAC4")

# Main Title
ax.text(c3_x + box_w/2.0, box_y + box_h - 15.0, "Calibration Restored",
        ha='center', va='center', fontsize=22.5, fontweight='bold', color="#102A43", zorder=3)

# Subtitle
ax.text(c3_x + box_w/2.0, box_y + box_h - 22.5, "Closed-Form Bayes Recalibration",
        ha='center', va='center', fontsize=12.2, fontweight='bold', color="#486581", zorder=3)

# Highlight Metric Box: Clean, Subtle Slate Box matching graphical_abstract1.png
metric_box = patches.FancyBboxPatch(
    (c3_x + 1.6, box_y + 24.5), box_w - 3.2, 23.0,
    boxstyle="round,pad=0.0,rounding_size=0.8",
    facecolor="#F0F4F8", edgecolor="#314D68", linewidth=1.4, zorder=3
)
ax.add_patch(metric_box)

ax.text(c3_x + box_w/2.0, box_y + 42.0, "HELD-OUT STRATUM (T3)",
        ha='center', va='center', fontsize=10.0, fontweight='bold', color="#486581", zorder=4)
ax.text(c3_x + box_w/2.0, box_y + 34.0, r"$\mathbf{Brier:\ .108 \rightarrow .028}$",
        ha='center', va='center', fontsize=18.5, fontweight='bold', color="#102A43", zorder=4)
ax.text(c3_x + box_w/2.0, box_y + 27.5, r"$\mathbf{-74\%}$ Calibration Error Reduction",
        ha='center', va='center', fontsize=11.2, fontweight='bold', color="#0D47A1", zorder=4)

# 2 Large, readable bullets
ax.text(c3_x + 2.2, box_y + 16.0, "• Zero-parameter closed-form recovery",
        ha='left', va='center', fontsize=10.2, color="#314D68", zorder=3)
ax.text(c3_x + 2.2, box_y + 8.5, "• Discriminative AUROC 100% preserved",
        ha='left', va='center', fontsize=10.2, fontweight='bold', color="#102A43", zorder=3)


# ==============================================================================
# BOTTOM VALIDATION BAR: BRAX -> MIMIC
# ==============================================================================
val_y = 2.4
val_h = 16.5
val_x = 2.2
val_w = 95.6

val_box = patches.FancyBboxPatch(
    (val_x, val_y), val_w, val_h,
    boxstyle="round,pad=0.0,rounding_size=0.9",
    facecolor="#F0F4F8", edgecolor="#B0BAC4", linewidth=1.4, zorder=2
)
ax.add_patch(val_box)

# Validation Tag
ax.text(
    val_x + 8.5, val_y + val_h/2.0, "EXTERNAL\nVALIDATION",
    ha='center', va='center', fontsize=9.4, fontweight='bold', color="#0D47A1", zorder=4,
    bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.4",
              facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=1.1)
)

# BRAX Cohort Pill
brax_pill = patches.FancyBboxPatch(
    (val_x + 18.0, val_y + 2.2), 22.5, val_h - 4.4,
    boxstyle="round,pad=0.0,rounding_size=0.6",
    facecolor="#FFFFFF", edgecolor="#314D68", linewidth=1.2, zorder=3
)
ax.add_patch(brax_pill)
ax.text(val_x + 29.25, val_y + val_h/2.0 + 2.4, "BRAX Cohort (Brazil)",
        ha='center', va='center', fontsize=12.0, fontweight='bold', color="#102A43", zorder=4)
ax.text(val_x + 29.25, val_y + val_h/2.0 - 2.8, r"N = 40,523 • Multi-Year $T_1 \to T_3$",
        ha='center', va='center', fontsize=9.8, color="#486581", zorder=4)

# Validation Arrow: BRAX -> MIMIC
val_arrow = patches.FancyArrowPatch(
    (val_x + 41.5, val_y + val_h/2.0), (val_x + 51.5, val_y + val_h/2.0),
    arrowstyle='-|>', mutation_scale=20, lw=2.4, color="#1976D2", zorder=5
)
ax.add_patch(val_arrow)
ax.text(val_x + 46.5, val_y + val_h/2.0 + 4.2, "Validation",
        ha='center', va='center', fontsize=9.8, fontweight='bold', color="#1976D2", zorder=6)

# MIMIC Cohort Pill
mimic_pill = patches.FancyBboxPatch(
    (val_x + 52.5, val_y + 2.2), 22.5, val_h - 4.4,
    boxstyle="round,pad=0.0,rounding_size=0.6",
    facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=1.2, zorder=3
)
ax.add_patch(mimic_pill)
ax.text(val_x + 63.75, val_y + val_h/2.0 + 2.4, "MIMIC-CXR (USA)",
        ha='center', va='center', fontsize=12.0, fontweight='bold', color="#0D47A1", zorder=4)
ax.text(val_x + 63.75, val_y + val_h/2.0 - 2.8, "N = 3,403 • Cross-Site Test",
        ha='center', va='center', fontsize=9.8, color="#1565C0", zorder=4)

# Cross-Site Finding Badge
ax.text(val_x + 85.8, val_y + val_h/2.0 + 2.4, "Multi-Site Generalization",
        ha='center', va='center', fontsize=11.2, fontweight='bold', color="#102A43", zorder=4)
ax.text(val_x + 85.8, val_y + val_h/2.0 - 2.8, "Zero-parameter correction replicated",
        ha='center', va='center', fontsize=9.6, style='italic', color="#486581", zorder=4)

# Save high-res 300 DPI image
plt.savefig("manuscript_cibm/graphical_abstract.png", dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='#FFFFFF')
print("Successfully generated manuscript_cibm/graphical_abstract.png at 300 DPI.")
