import matplotlib
matplotlib.rcdefaults()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use("default")
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "text.color":        "black",
})

N_BLOCKS = 6
N_FOLDS  = 5
BW       = 1.0
BH       = 0.52
GAP_Y    = 0.22
LABEL_X  = -0.4

COLORS = {
    "train":  "#2166ac",
    "val":    "#d73027",
    "future": "#d9d9d9",
}

fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor="white")
ax.set_facecolor("white")
ax.axis("off")

for fold in range(N_FOLDS):
    y = (N_FOLDS - 1 - fold) * (BH + GAP_Y)

    ax.text(LABEL_X, y + BH / 2, f"Fold {fold + 1}",
            ha="right", va="center", fontsize=8.5, color="black")

    for b in range(N_BLOCKS):
        x = b * BW
        if b <= fold:
            color = COLORS["train"]
            txt   = "Training"
        elif b == fold + 1:
            color = COLORS["val"]
            txt   = "Validation"
        else:
            color = COLORS["future"]
            txt   = "Unused"

        rect = mpatches.FancyBboxPatch(
            (x + 0.04, y + 0.04), BW - 0.08, BH - 0.08,
            boxstyle="round,pad=0.02",
            linewidth=0, facecolor=color, zorder=2,
        )
        ax.add_patch(rect)

        text_color = "white" if color != COLORS["future"] else "#888888"
        ax.text(x + BW / 2, y + BH / 2, txt,
                ha="center", va="center",
                fontsize=7.5, color=text_color,
                fontweight="bold", zorder=3)

# ── Column headers ────────────────────────────────────────────────────────────
header_y = N_FOLDS * (BH + GAP_Y) - GAP_Y + 0.12
for b in range(N_BLOCKS):
    ax.text(b * BW + BW / 2, header_y,
            f"Block {b + 1}",
            ha="center", va="bottom", fontsize=7.5, color="#444444")

# ── Time axis with week tick marks ────────────────────────────────────────────
arrow_y   = -0.28
tick_h    = 0.08
# Block boundaries: week 1, 17, 33, 49, 65, 81, 96
week_ticks = [1, 17, 33, 49, 65, 81, 96]
# Map week number to x position: week 1 → x=0, week 96 → x=6
def week_to_x(w):
    return (w - 1) / (96 - 1) * N_BLOCKS

ax.annotate("",
            xy=(N_BLOCKS * BW + 0.1, arrow_y),
            xytext=(0, arrow_y),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9))

for w in week_ticks:
    x = week_to_x(w)
    # tick mark
    ax.plot([x, x], [arrow_y, arrow_y - tick_h],
            color="black", lw=0.8)
    # label
    ax.text(x, arrow_y - tick_h - 0.06,
            f"Wk {w}",
            ha="center", va="top", fontsize=7, color="#333333")

# Time label
ax.text(N_BLOCKS * BW / 2, arrow_y - tick_h - 0.28,
        "Time (weeks)", ha="center", va="top", fontsize=8.5)

ax.set_xlim(-0.8, N_BLOCKS * BW + 0.4)
ax.set_ylim(arrow_y - tick_h - 0.55, header_y + 0.5)

plt.tight_layout(pad=0.3)
plt.savefig("output/figures/cv_diagram.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
plt.show()
print("Saved: cv_diagram.png")