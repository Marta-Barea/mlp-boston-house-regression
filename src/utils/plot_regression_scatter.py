import os
import matplotlib.pyplot as plt


def plot_regression_scatter(y_true, y_pred, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
             color='red', linestyle='--', label='Ideal')
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "regression_scatter.png"))
    plt.close()
