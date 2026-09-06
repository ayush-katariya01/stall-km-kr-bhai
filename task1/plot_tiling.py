import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("tile_dif_matrix.csv", skipinitialspace=True)
print(df["H"].unique())

# Clean column names
df.columns = df.columns.str.strip()

# Convert columns to numeric
df["H"] = pd.to_numeric(df["H"], errors="coerce")
df["b"] = pd.to_numeric(df["b"], errors="coerce")

# Convert "1.128x" -> 1.128
df["speedup"] = (
    df["speedup"]
    .astype(str)
    .str.strip()
    .str.replace("x", "", regex=False)
)
df["speedup"] = pd.to_numeric(df["speedup"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["H", "b", "speedup"])

# Sort properly
df = df.sort_values(["b", "H"])

print(df[["H", "b", "speedup"]])

# -----------------------------
# Speedup vs Matrix Size
# -----------------------------
plt.figure(figsize=(10, 6))

for b, group in df.groupby("b"):
    group = group.sort_values("H")

    plt.plot(
        group["H"],
        group["speedup"],
        marker="o",
        label=f"b={int(b)}"
    )

# Baseline: speedup = 1
plt.axhline(y=1.0, linestyle="--")

plt.xlabel("Matrix Size")
plt.ylabel("Speedup")
plt.title("Speedup vs. Matrix Size")

# Matrix sizes are powers of 2
plt.xscale("log", base=2)

plt.xticks(
    [512, 1024, 2048, 4096, 8192, 16384],
    ["512", "1024", "2048", "4096", "8192", "16384"]
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("speedup_vs_matrix_size.png", dpi=300)