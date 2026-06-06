"""Plot training loss, learning rate, and per-timestep losses from a checkpoint file.

Usage:
    python plot_loss.py                                          # uses latest checkpoint in ./results/bark_run/
    python plot_loss.py --checkpoint results/bark_run/model-6.pt
    python plot_loss.py --results_folder results/bark_60k_p2     # auto-finds latest checkpoint
"""
from fire import Fire
import torch
import os
import glob


def main(checkpoint: str = None, results_folder: str = None, save_loss_every: int = 100):
    # Find checkpoint
    if checkpoint is None:
        folder = results_folder or './results/bark_run'
        pts = sorted(glob.glob(os.path.join(folder, 'model-*.pt')),
                     key=lambda x: int(x.split('model-')[-1].split('.pt')[0]))
        if not pts:
            print(f"No checkpoints found in {folder}")
            return
        checkpoint = pts[-1]
        print(f"Using latest checkpoint: {checkpoint}")

    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    losses = data['loss']
    lrs = data['lr']
    step = data['step']
    t_bucket_losses = data.get('t_bucket_losses', None)

    print(f"Step: {step}")
    print(f"Loss entries: {len(losses)} (every {save_loss_every} steps)")
    print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
    print(f"Final loss (last 10 avg): {sum(losses[-10:])/10:.4f}")

    has_buckets = t_bucket_losses is not None and len(t_bucket_losses.get(0, [])) > 0

    try:
        import matplotlib.pyplot as plt
        import math

        steps = [i * save_loss_every for i in range(len(losses))]

        n_plots = 3 if has_buckets else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)

        # Plot 1: Overall loss
        ax1 = axes[0]
        ax1.plot(steps, losses, linewidth=0.5, alpha=0.4, color='blue', label='raw')
        if len(losses) > 20:
            window = 20
            smoothed = [sum(losses[max(0,i-window):i+1])/len(losses[max(0,i-window):i+1])
                        for i in range(len(losses))]
            ax1.plot(steps, smoothed, linewidth=2, color='red', label='smoothed (w=20)')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Loss (step {step})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learning rate
        ax2 = axes[1]
        ax2.plot(steps[:len(lrs)], lrs, linewidth=1.5, color='green')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Per-timestep bucket losses
        if has_buckets:
            ax3 = axes[2]
            bucket_labels = ['t=0-200 (hard)', 't=200-400', 't=400-600', 't=600-800', 't=800-1000 (easy)']
            colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

            for i in range(5):
                bucket_data = t_bucket_losses[i]
                bucket_steps = steps[:len(bucket_data)]
                # Filter NaN values
                valid = [(s, v) for s, v in zip(bucket_steps, bucket_data) if v == v]
                if not valid:
                    continue
                vs, vv = zip(*valid)

                # Smoothed version
                if len(vv) > 10:
                    w = 10
                    smoothed_v = [sum(vv[max(0,j-w):j+1])/len(vv[max(0,j-w):j+1])
                                  for j in range(len(vv))]
                    ax3.plot(vs, smoothed_v, linewidth=2, color=colors[i], label=bucket_labels[i])
                else:
                    ax3.plot(vs, vv, linewidth=1.5, color=colors[i], label=bucket_labels[i])

            ax3.set_ylabel('Unweighted MSE Loss')
            ax3.set_xlabel('Step')
            ax3.set_title('Loss by Timestep Range (unweighted, per-bucket)')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
        else:
            axes[-1].set_xlabel('Step')
            print("No per-timestep loss data in checkpoint (old format).")

        plt.tight_layout()
        save_path = checkpoint.replace('.pt', '_loss.png')
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
        plt.show()

    except ImportError:
        print("\nmatplotlib not installed. Install with: pip install matplotlib")
        print("Printing loss values instead:\n")
        for i, (l, lr) in enumerate(zip(losses, lrs)):
            print(f"Step {i*save_loss_every:6d}: loss={l:.6f}, lr={lr:.2e}")


if __name__ == '__main__':
    Fire(main)
