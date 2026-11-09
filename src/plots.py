import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr_mat(corr_matrix):
    # Create the heatmap plot
    plt.figure(figsize=(14, 11))

    # We turn off 'annot' because 141x141 numbers will overlap and look messy
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',     # Red for positive correlation, Blue for negative
        center=0,            # 0 correlation will be neutral gray/white
        xticklabels=False,   # Turn off feature names on axes if they crowd the plot
        yticklabels=False,   # (Set to True if you want to see specific labels)
        cbar_kws={'label': 'Pearson Correlation Coefficient'}
    )

    plt.title("Correlation Matrix Heatmap (141 matminer Features)", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_val_loss(histories, title):
    plt.figure(figsize=(8, 4.5))
    for name, h in histories.items():
        plt.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_comparative_loss_curves(
    histories_large,
    histories_small,
    colors=None,
    title_large="4x256 Architecture (Deeper / Wider)",
    title_small="2x128 Architecture (Smaller)",
    suptitle="Comparative Loss Curves Across Feature Reductions",
    figsize=(16, 5)
):
    """
    Plots side-by-side comparative training/validation loss curves for two network architectures.

    Parameters:
    -----------
    histories_large : dict
        Loss histories dictionary for the larger architecture (e.g., histories_256x4).
    histories_small : dict
        Loss histories dictionary for the smaller architecture (e.g., histories_128x2).
    colors : dict, optional
        Mapping of feature set names to Matplotlib colors. Defaults to preset palette.
    title_large : str, optional
        Title for the left subplot panel.
    title_small : str, optional
        Title for the right subplot panel.
    suptitle : str, optional
        Overall figure title.
    figsize : tuple, optional
        Figure size dimensions (width, height).
    """
    if colors is None:
        colors = {
            "68 Features (All Uncorrelated)": "teal",
            "40 Features (Top RF Selected)": "chocolate",
            "20 Features (Top RF Selected)": "forestgreen"
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Helper to eliminate redundant plotting code for both panels
    def _draw_panel(ax, histories, title, show_ylabel=False):
        for name, hist in histories.items():
            color = colors.get(name, "gray")
            epochs_run = range(1, hist["epochs_run"] + 1)

            # Loss lines
            ax.plot(epochs_run, hist["train_loss"], label=f"{name} - Train", color=color, linestyle="-", alpha=0.8)
            ax.plot(epochs_run, hist["val_loss"], label=f"{name} - Val", color=color, linestyle="--", alpha=0.9)

            # Early stopping checkpoint
            ax.scatter(hist["best_epoch"], hist["val_loss"][hist["best_epoch"] - 1],
                       color=color, edgecolors='black', s=40, zorder=5)

        ax.set_xlabel("Epoch", fontsize=11)
        if show_ylabel:
            ax.set_ylabel("Loss (MSE)", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    # Render panels
    _draw_panel(ax1, histories_large, title_large, show_ylabel=True)
    _draw_panel(ax2, histories_small, title_small, show_ylabel=False)

    plt.suptitle(suptitle, fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_band_gap_predictions(targets, predictions, path):
    """Save one predicted-versus-actual panel for each benchmark."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(12, 4), sharex=True, sharey=True)
    lower = min(0, *(values.min() for values in predictions.values()))
    upper = max(targets.max(), *(values.max() for values in predictions.values()))
    for ax, (label, values) in zip(axes, predictions.items()):
        ax.scatter(targets, values, s=10, alpha=0.35)
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        ax.set(xlabel="Actual band gap (eV)", title=label)
    axes[0].set_ylabel("Predicted band gap (eV)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_uncertainty(binned, path):
    """Save the relationship between ensemble spread and test error."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(binned["sigma"], binned["abs_error"], "o-", color="darkblue")
    ax.set(
        xlabel="Mean ensemble standard deviation (eV)",
        ylabel="Mean absolute residual (eV)",
        title="Ensemble spread and test error",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
