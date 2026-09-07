import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CSV file
csv_path = "test_results.csv"

# Directory containing CSV
csv_dir = os.path.dirname(os.path.abspath(csv_path))

# Load CSV
df = pd.read_csv(csv_path)

# Clean headers and text
df.columns = df.columns.str.strip()
df["stage"] = df["stage"].str.strip()

# Clean speedup column
df["speedup"] = (
    df["speedup"]
    .astype(str)
    .str.strip()
    .str.rstrip("x")
    .astype(float)
)

# --------------------------------------------------
# Automatically detect K values and stages
# --------------------------------------------------

k_values = sorted(df["K"].unique())
stages = list(df["stage"].unique())

print("K values found:", k_values)
print("Stages found:", stages)

# --------------------------------------------------
# Create one graph for each K
# --------------------------------------------------

for k in k_values:

    # Select rows for this K
    k_df = df[df["K"] == k]

    # Pivot:
    # rows    -> matrix size
    # columns -> stage
    # values  -> speedup
    pivot_df = k_df.pivot(
        index="size",
        columns="stage",
        values="speedup"
    )

    # Keep stages in the order they appear in CSV
    available_stages = [
        stage for stage in stages
        if stage in pivot_df.columns
    ]

    pivot_df = pivot_df[available_stages]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 5),
        facecolor="black"
    )

    ax.set_facecolor("black")

    x = np.arange(len(pivot_df.index))

    # Automatically calculate bar width
    num_stages = len(available_stages)
    width = 0.8 / num_stages

    # --------------------------------------------------
    # Plot every detected stage
    # --------------------------------------------------

    for i, stage in enumerate(available_stages):

        ax.bar(
            x + i * width,
            pivot_df[stage],
            width,
            label=stage.capitalize()
        )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_ylabel(
        "Speedup (x)",
        fontsize=12,
        color="white"
    )

    ax.set_xlabel(
        "Matrix size",
        fontsize=12,
        color="white",
        labelpad=10
    )

    # Center the group of bars around each matrix size
    ax.set_xticks(
        x + width * (num_stages - 1) / 2
    )

    ax.set_xticklabels(
        [f"size {s}" for s in pivot_df.index],
        color="white",
        fontsize=10
    )

    ax.tick_params(
        colors="white",
        labelsize=10
    )

    ax.grid(
        axis="y",
        color="#444444",
        linestyle="-",
        linewidth=0.8
    )

    ax.set_axisbelow(True)

    # Outer frame
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.2)

    # Title
    ax.set_title(
        f"Speedup Comparison (K = {k})",
        fontsize=14,
        color="white"
    )

    # Legend
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        fontsize=11,
        labelcolor="white"
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Save graph
    # --------------------------------------------------

    output_image_path = os.path.join(
        csv_dir,
        f"speedup_K{k}.png"
    )

    plt.savefig(
        output_image_path,
        bbox_inches="tight",
        dpi=300,
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print(f"Saved: {output_image_path}")