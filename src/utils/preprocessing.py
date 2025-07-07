import numpy as np
import matplotlib.pyplot as plt

def minmax_outliers(IM, q_low=1, q_high=99):
    for _ in range(IM.shape[0]):
        low = np.percentile(IM[_], q_low)
        high = np.percentile(IM[_], q_high)
        clipped_data = np.clip(IM[_], low, high)
        IM[_] = (clipped_data - low) / (high - low)
    return IM



def minmax_per_channel(IM):
    """
    Perform min-max normalization per channel of the input image.

    Parameters:
    IM (np.array): Input 3D image in format (channels, height, width).

    Returns:
    np.array: Min-max normalized image (channels, height, width).

    Example:
    >>> normalized = minmax_per_channel(np.random.rand(3, 64, 64))
    """
    for _ in range(IM.shape[0]):
        IM[_] = (IM[_] - np.min(IM[_])) / (np.max(IM[_]) - np.min(IM[_]))
        plt.imshow(IM[_])
        plt.show()
    return IM

def minmax_outliers_per_channel(IM, q_low=1, q_high=99):
    """
    Perform min-max normalization with outlier removal per channel.

    Parameters:
    IM (np.array): Input 3D image in format (channels, height, width).
    q_low (float): Lower percentile for clipping (default: 1).
    q_high (float): Upper percentile for clipping (default: 99).

    Returns:
    np.array: Normalized image with outliers removed.

    Example:
    >>> normalized = minmax_outliers_per_channel(np.random.rand(3, 64, 64), 5, 95)
    """
    low = np.percentile(IM, q_low)
    high = np.percentile(IM, q_high)
    clipped_data = np.clip(IM, low, high)
    normalized_data = (clipped_data - low) / (high - low)
    return normalized_data

def minmax(IM):
    """
    Perform min-max normalization on the entire input image.

    Parameters:
    IM (np.array): Input 3D image in format (channels, height, width).

    Returns:
    np.array: Min-max normalized image (channels, height, width).

    Example:
    >>> normalized = minmax(np.random.rand(3, 64, 64))
    """
    IM = (IM - np.min(IM)) / (np.max(IM) - np.min(IM))
    for _ in range(IM.shape[0]):
        plt.imshow(IM[_])
        plt.show()
    return IM

def Z_norm(IM):
    """
    Perform Z-score normalization on the entire input image.

    Parameters:
    IM (np.array): Input 3D image in format (channels, height, width).

    Returns:
    np.array: Z-score normalized image (channels, height, width).

    Example:
    >>> normalized = Z_norm(np.random.rand(3, 64, 64))
    """
    IM = (0.15 * (IM - np.mean(IM))) / np.std(IM) + 0.5
    return IM

def plot_one(IM):
    """
    Plot one multispectral image.

    Parameters:
    IM (np.array): Input 3D image in format (channels, height, width).

    Example:
    >>> plot_one(np.random.rand(3, 64, 64))
    """
    for _ in range(IM.shape[0]):
        plt.imshow(IM[_])
        plt.show()

def Z_norm_per_channel(IM):
    """
    Perform Z-score normalization per channel of the input image.

    Parameters:
    IM (np.array): Input 3D image in format (channels, height, width).

    Returns:
    np.array: Z-score normalized image (channels, height, width).

    Example:
    >>> normalized = Z_norm_per_channel(np.random.rand(3, 64, 64))
    """
    for _ in range(IM.shape[0]):
        IM[_] = (IM[_] - np.mean(IM[_])) / np.std(IM[_])
        plt.imshow(IM[_])
        plt.show()
    return IM

def minmax_outliers(IM, q_low=1, q_high=99):
    """
    Perform min-max normalization with outlier removal for each channel.

    Parameters:
    IM (np.array): Input 3D image in format (channels, height, width).
    q_low (float): Lower percentile for clipping (default: 1).
    q_high (float): Upper percentile for clipping (default: 99).

    Returns:
    np.array: Normalized image with outliers removed.

    Example:
    >>> normalized = minmax_outliers(np.random.rand(3, 64, 64), 5, 95)
    """
    for _ in range(IM.shape[0]):
        low = np.percentile(IM[_], q_low)
        high = np.percentile(IM[_], q_high)
        clipped_data = np.clip(IM[_], low, high)
        IM[_] = (clipped_data - low) / (high - low)
    return IM
