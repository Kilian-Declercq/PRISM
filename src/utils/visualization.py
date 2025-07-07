
import torch
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import List, Union, Tuple
import cv2
import os
from pathlib import Path


def show_abundances(model: torch.nn.Module, input_data: torch.Tensor, pruned_layers: List[int]) -> None:
    """
    Display abundance maps from the model.
    
    Args:
        model: The autoencoder model
        input_data: Input tensor
        pruned_layers: List of indices of pruned layers
    """
    model.eval()
    with torch.no_grad():
        _, abundance_maps = model(input_data)
    shape = abundance_maps[0, 0, :, :].cpu().shape

    num_maps = abundance_maps.size(1)
    grid_size = math.ceil(math.sqrt(num_maps))
    fig, axs = plt.subplots(grid_size - 1, grid_size, figsize=(20, 18))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]


    for i in range(num_maps):
        ax = axs[i]
        norm = torch.norm(abundance_maps[0, i, :, :]).data
        if i in pruned_layers:
            
            base_img = 0.1*np.ones(shape)
            im = ax.imshow(base_img, cmap='viridis', vmin=0, vmax=1,alpha = 0.5)
    
            ax.set_title(f'Map {i+1}\nNorm: {norm:.2f}', color='red', pad=20,fontsize = 15)
            

            ax.patch.set_facecolor("#d6272880")
            ax.patch.set_linewidth(12)
            ax.patch.set_edgecolor("#d6272890")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        else:
            base_img = abundance_maps[0, i, :, :].clone().cpu()
        
            im = ax.imshow(base_img, 
                         cmap='viridis', vmin=0, vmax=1,
                      )
            
            ax.patch.set_linewidth(12)
            ax.patch.set_edgecolor("#2ca02c90")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.set_title(f'Map {i+1}\nNorm: {norm:.2f}', color='green', pad=20,fontsize = 15)

        #ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Remove unused subplots
    for i in range(num_maps, len(axs)):
        fig.delaxes(axs[i])

    

    plt.tight_layout()
    plt.savefig(f'results/visualizations_tmp/gif_final_abundances_{len(pruned_layers)}.png')
    plt.show()
    plt.close()

    # Plot decoder weights
    plot = model.decoder.weight.clone()
    indices_to_keep = [i for i in range(num_maps)]
    data = plot[:, indices_to_keep, 0, 0].cpu().detach().numpy()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#7A82AB', '#001F3F']

    plt.figure(figsize=(12, 6))
    for i in range(data.shape[1]):
        plt.plot(data[:, i], color=colors[i], label=f'Weight {indices_to_keep[i]+1}')

    plt.title('Weights Visualization')
    plt.xlabel('Index')
    plt.ylabel('Weight Value')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_image_with_circles(image: np.ndarray, 
                          flat_indices: List[int], 
                          radius: int = 10, 
                          color: tuple = (255, 0, 0), 
                          thickness: int = 2) -> np.ndarray:
    """
    Plot image with circles at specified indices.
    
    Args:
        image: Input image
        flat_indices: List of flattened indices where to draw circles
        radius: Circle radius
        color: Circle color in BGR format
        thickness: Circle line thickness
    
    Returns:
        Image with drawn circles
    """
    image_with_circles = image.copy()
    if len(image_with_circles.shape) == 2:
        image_with_circles = cv2.cvtColor(image_with_circles, cv2.COLOR_GRAY2RGB)

    height, width = image.shape[:2]

    for i, idx in enumerate(flat_indices):
        y = idx // width
        x = idx % width
        cv2.circle(image_with_circles, (x, y), radius, color, thickness)

        text_x = x - 10
        text_y = y - radius - 10
        text = str(i+1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, text_thickness)

        padding = 5
        cv2.rectangle(
            image_with_circles,
            (text_x - padding, text_y - text_height - padding),
            (text_x + text_width + padding, text_y + padding),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            image_with_circles,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness
        )

    return image_with_circles


def plot_data(img_array: np.ndarray, 
              gt_data: Union[np.ndarray, Tuple[np.ndarray, ...]], 
              dataset: str,
              title: str = "") -> None:
    """
    Plot loaded data and ground truth.
    
    Args:
        img_array: Input image array (channels, height, width)
        gt_data: Ground truth data (single array for MStex, tuple for MSBin)
        dataset: Dataset type ('MSBin' or 'MStex')
        title: Optional title for the plot
    """
    plt.ioff()
    plt.figure(figsize=(15, 8))
    
    # Plot input images
    num_channels = min(4, img_array.shape[0])  # Show up to 4 channels
    for i in range(num_channels):
        plt.subplot(2, 4, i+1)
        plt.imshow(img_array[i], cmap='gray')
        plt.title(f'Channel {i+1}')
        plt.axis('off')
    
    # Plot ground truth
    if dataset == 'MSBin':

        GT1, GT2, Mask, BG = gt_data
        plt.subplot(2, 4, 5)
        plt.imshow(GT1, cmap='gray')
        plt.title('GT1')
        plt.axis('off')
        
        plt.subplot(2, 4, 6)
        plt.imshow(GT2, cmap='gray')
        plt.title('GT2')
        plt.axis('off')
        
        plt.subplot(2, 4, 7)
        plt.imshow(Mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        
        plt.subplot(2, 4, 8)
        plt.imshow(BG, cmap='gray')
        plt.title('Background')
        plt.axis('off')

    else:  
        plt.subplot(2, 4, 5)
        plt.imshow(gt_data, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    output_dir = 'results/visualizations_tmp/'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir /  "current_input_img.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}\n")

    input("Press [enter] to continue...")
    plt.close()


def save_results(model: torch.nn.Module, 
                abundance_maps: torch.Tensor, 
                loss_print: List[float], 
                mse_print: List[float], 
                output_dir: str,
                indices_to_keep: List[int],
                metrics: dict) -> None:
    """
    Save training results, plots and metrics.
    
    Args:
        model: Trained model
        abundance_maps: Generated abundance maps
        loss_print: List of training losses
        mse_print: List of MSE values
        output_dir: Output directory path
        indices_to_keep: List of indices of kept layers
        metrics: Dictionary of computed metrics
    """

    output_dir = Path(output_dir)

    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directories
    (output_dir / 'abundance_maps').mkdir(exist_ok=True)
    (output_dir / 'weight_plots').mkdir(exist_ok=True)
    (output_dir / 'loss_curves').mkdir(exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)

    # Save loss curves
    plt.figure(figsize=(10, 6))
    x = [12-i for i in range(len(mse_print))]
    plt.plot(x, mse_print)
    plt.title('MSE Loss Progression')
    plt.savefig(output_dir / 'loss_curves' / 'mse_loss.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x, loss_print)
    plt.title('Total Loss Progression')
    plt.savefig(output_dir / 'loss_curves' / 'total_loss.png')
    plt.close()

    # Save weights plot
    plot = model.decoder.weight.clone()
    data = plot[:, indices_to_keep, 0, 0].cpu().detach().numpy()
    


    for i in range(data.shape[1]):
        plt.plot(data[:, i], label=f'Weight {indices_to_keep[i]+1}')
    plt.title('Final Weights Visualization')
    plt.savefig(output_dir / 'weight_plots' / 'final_weights.png')
    plt.close()

    # Save abundance maps
    abundance_data = abundance_maps[:, indices_to_keep, :, :].cpu().detach().numpy()

    plt.figure(figsize=(15, 8))
    for i in range(data.shape[1]):
        plt.subplot(3, 4, i+1)
        plt.imshow(abundance_data[0][i], cmap='viridis')
        plt.title(f'abundance {indices_to_keep[i]+1}')
        plt.axis('off')

    plt.savefig(output_dir / 'abundance_maps' / 'final_abundances.png')

    np.save(output_dir / 'abundance_maps' / 'final_abundances.npy', abundance_data)

    # Save metrics
    np.save(output_dir / 'metrics' / 'results.npy', metrics)