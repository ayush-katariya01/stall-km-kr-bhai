import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# Create output directory
# =========================================================

os.makedirs("plots", exist_ok=True)


# =========================================================
# Data
# =========================================================

matrix_sizes = np.array([512, 1024, 2048, 4096, 8192])


# -------------------------
# K = 3
# -------------------------

naive_256_k3 = np.array([
    1.958, 6.690, 29.041, 114.384, 443.129
])

simd256_k3 = np.array([
    0.257, 1.047, 4.743, 18.343, 66.101
])

naive_128_k3 = np.array([
    1.690, 6.745, 27.395, 113.940, 444.469
])

simd128_k3 = np.array([
    0.378, 1.423, 6.527, 21.372, 94.031
])

speedup256_k3 = np.array([
    7.83, 6.47, 6.21, 6.37, 6.72
])

speedup128_k3 = np.array([
    4.54, 4.79, 4.36, 5.33, 4.75
])


# -------------------------
# K = 5
# -------------------------

naive_256_k5 = np.array([
    4.001, 14.841, 62.716, 239.914, 952.394
])

simd256_k5 = np.array([
    0.615, 2.319, 14.436, 40.296, 156.276
])

naive_128_k5 = np.array([
    3.887, 15.009, 62.383, 237.904, 950.128
])

simd128_k5 = np.array([
    1.019, 3.698, 17.573, 57.698, 231.197
])

speedup256_k5 = np.array([
    6.54, 6.40, 4.67, 6.07, 6.13
])

speedup128_k5 = np.array([
    3.91, 4.07, 3.61, 4.12, 4.11
])


# =========================================================
# Helper function
# =========================================================

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join("plots", filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# =========================================================
# 1. Naive vs SIMD256 - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, naive_256_k3, marker="o", label="Naive")
plt.plot(matrix_sizes, simd256_k3, marker="o", label="SIMD256")

plt.xlabel("Matrix Size")
plt.ylabel("Average Time (ms)")
plt.title("Naive vs SIMD256 Performance (K=3)")
plt.legend()
plt.grid(True)

save_plot("01_naive_vs_simd256_k3.png")


# =========================================================
# 2. Naive vs SIMD256 - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, naive_256_k5, marker="o", label="Naive")
plt.plot(matrix_sizes, simd256_k5, marker="o", label="SIMD256")

plt.xlabel("Matrix Size")
plt.ylabel("Average Time (ms)")
plt.title("Naive vs SIMD256 Performance (K=5)")
plt.legend()
plt.grid(True)

save_plot("02_naive_vs_simd256_k5.png")


# =========================================================
# 3. Naive vs SIMD128 - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, naive_128_k3, marker="o", label="Naive")
plt.plot(matrix_sizes, simd128_k3, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Average Time (ms)")
plt.title("Naive vs SIMD128 Performance (K=3)")
plt.legend()
plt.grid(True)

save_plot("03_naive_vs_simd128_k3.png")


# =========================================================
# 4. Naive vs SIMD128 - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, naive_128_k5, marker="o", label="Naive")
plt.plot(matrix_sizes, simd128_k5, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Average Time (ms)")
plt.title("Naive vs SIMD128 Performance (K=5)")
plt.legend()
plt.grid(True)

save_plot("04_naive_vs_simd128_k5.png")


# =========================================================
# 5. SIMD256 Speedup - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, speedup256_k3, marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("Speedup (×)")
plt.title("SIMD256 Speedup vs Matrix Size (K=3)")
plt.grid(True)

save_plot("05_simd256_speedup_k3.png")


# =========================================================
# 6. SIMD256 Speedup - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, speedup256_k5, marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("Speedup (×)")
plt.title("SIMD256 Speedup vs Matrix Size (K=5)")
plt.grid(True)

save_plot("06_simd256_speedup_k5.png")


# =========================================================
# 7. SIMD128 Speedup - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, speedup128_k3, marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("Speedup (×)")
plt.title("SIMD128 Speedup vs Matrix Size (K=3)")
plt.grid(True)

save_plot("07_simd128_speedup_k3.png")


# =========================================================
# 8. SIMD128 Speedup - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, speedup128_k5, marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("Speedup (×)")
plt.title("SIMD128 Speedup vs Matrix Size (K=5)")
plt.grid(True)

save_plot("08_simd128_speedup_k5.png")


# =========================================================
# 9. SIMD256 vs SIMD128 Time - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, simd256_k3, marker="o", label="SIMD256")
plt.plot(matrix_sizes, simd128_k3, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Average Time (ms)")
plt.title("SIMD256 vs SIMD128 Performance (K=3)")
plt.legend()
plt.grid(True)

save_plot("09_simd256_vs_simd128_k3.png")


# =========================================================
# 10. SIMD256 vs SIMD128 Time - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, simd256_k5, marker="o", label="SIMD256")
plt.plot(matrix_sizes, simd128_k5, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Average Time (ms)")
plt.title("SIMD256 vs SIMD128 Performance (K=5)")
plt.legend()
plt.grid(True)

save_plot("10_simd256_vs_simd128_k5.png")


# =========================================================
# 11. Speedup Comparison - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, speedup256_k3, marker="o", label="SIMD256")
plt.plot(matrix_sizes, speedup128_k3, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Speedup (×)")
plt.title("SIMD256 vs SIMD128 Speedup Comparison (K=3)")
plt.legend()
plt.grid(True)

save_plot("11_speedup_comparison_k3.png")


# =========================================================
# 12. Speedup Comparison - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, speedup256_k5, marker="o", label="SIMD256")
plt.plot(matrix_sizes, speedup128_k5, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Speedup (×)")
plt.title("SIMD256 vs SIMD128 Speedup Comparison (K=5)")
plt.legend()
plt.grid(True)

save_plot("12_speedup_comparison_k5.png")


# =========================================================
# 13. Complete Scaling - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, naive_256_k3, marker="o", label="Naive")
plt.plot(matrix_sizes, simd256_k3, marker="o", label="SIMD256")
plt.plot(matrix_sizes, simd128_k3, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Average Time (ms)")
plt.title("Performance Scaling with Matrix Size (K=3)")
plt.legend()
plt.grid(True)

save_plot("13_complete_scaling_k3.png")


# =========================================================
# 14. Complete Scaling - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, naive_256_k5, marker="o", label="Naive")
plt.plot(matrix_sizes, simd256_k5, marker="o", label="SIMD256")
plt.plot(matrix_sizes, simd128_k5, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Average Time (ms)")
plt.title("Performance Scaling with Matrix Size (K=5)")
plt.legend()
plt.grid(True)

save_plot("14_complete_scaling_k5.png")


# =========================================================
# 15. Matrix Size vs SIMD256 Time - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, simd256_k3, marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("SIMD256 Time (ms)")
plt.title("Matrix Size vs SIMD256 Execution Time (K=3)")
plt.grid(True)

save_plot("15_matrix_vs_simd256_time_k3.png")


# =========================================================
# 16. Matrix Size vs SIMD256 Time - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, simd256_k5, marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("SIMD256 Time (ms)")
plt.title("Matrix Size vs SIMD256 Execution Time (K=5)")
plt.grid(True)

save_plot("16_matrix_vs_simd256_time_k5.png")


# =========================================================
# 17. Matrix Size vs SIMD128 Time - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, simd128_k3, marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("SIMD128 Time (ms)")
plt.title("Matrix Size vs SIMD128 Execution Time (K=3)")
plt.grid(True)

save_plot("17_matrix_vs_simd128_time_k3.png")


# =========================================================
# 18. Matrix Size vs SIMD128 Time - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, simd128_k5, marker="o")

plt.xlabel("Matrix Size")
plt.ylabel("SIMD128 Time (ms)")
plt.title("Matrix Size vs SIMD128 Execution Time (K=5)")
plt.grid(True)

save_plot("18_matrix_vs_simd128_time_k5.png")


# =========================================================
# 19. Matrix Size vs SIMD Speedup - K=3
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, speedup256_k3, marker="o", label="SIMD256")
plt.plot(matrix_sizes, speedup128_k3, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Speedup (×)")
plt.title("Matrix Size vs SIMD Speedup (K=3)")
plt.legend()
plt.grid(True)

save_plot("19_matrix_vs_speedup_k3.png")


# =========================================================
# 20. Matrix Size vs SIMD Speedup - K=5
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(matrix_sizes, speedup256_k5, marker="o", label="SIMD256")
plt.plot(matrix_sizes, speedup128_k5, marker="o", label="SIMD128")

plt.xlabel("Matrix Size")
plt.ylabel("Speedup (×)")
plt.title("Matrix Size vs SIMD Speedup (K=5)")
plt.legend()
plt.grid(True)

save_plot("20_matrix_vs_speedup_k5.png")


# =========================================================
# Done
# =========================================================

print("All plots generated successfully.")
print("Plots are saved in: ./plots/")
print("Number of plots generated: 20")