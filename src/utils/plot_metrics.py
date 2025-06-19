import os
import matplotlib.pyplot as plt


def plot_metrics(results, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(results.history['loss'], label='train_loss')
    if 'val_loss' in results.history:
        plt.plot(results.history['val_loss'], label='val_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()

    if 'mean_absolute_error' in results.history:
        plt.figure()
        plt.plot(results.history['mean_absolute_error'], label='train_mae')
        if 'val_mean_absolute_error' in results.history:
            plt.plot(
                results.history['val_mean_absolute_error'], label='val_mae')
        plt.title('Training MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'mae.png'))
        plt.close()
