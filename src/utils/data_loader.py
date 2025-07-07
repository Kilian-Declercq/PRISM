import numpy as np
from PIL import Image
import glob
from pathlib import Path
from typing import Tuple, Union, Optional
from .preprocessing import minmax_outliers
from .visualization import plot_data

def load_data(image_name: str, base_path: str, dataset: str = 'MSBin', crop_h: Optional[Tuple[int, int]] = None, crop_w: Optional[Tuple[int, int]] = None, verbose: bool = False, norm: bool = True) -> Tuple[np.ndarray, ...]:
    """
    Load and preprocess image data from specified dataset.
    
    Args:
        image_name: Name of the image to load
        dataset: Dataset type ('MSBin' or 'MStex')
        
    Returns:
        Tuple containing:
            - img_array: Preprocessed image array
            - GT1: First ground truth mask
            - GT2: Second ground truth mask (MSBin only)
            - Mask: Additional mask (MSBin only)
            - BG: Background mask (MSBin only)
    """
    if base_path: 
        base_path = Path(base_path)


    if dataset == 'MSBin':
        data = _load_msbin(image_name, crop_h = crop_h, crop_w = crop_w, base_path = base_path, norm = norm )
        if verbose:
            print('Image shape', data[0].shape)
            print('GT shape',data[1].shape)
            plot_data(
                data[0],  # img_array
                data[1:],  # GT1, GT2, Mask, BG
                dataset,
                f"MSBin Data - {image_name}"
            )
        
    elif dataset == 'MStex':
        data = _load_mstex(image_name, crop_h = crop_h, crop_w = crop_w, base_path = base_path, norm = norm )
        if verbose:
            print('Image shape', data[0].shape)
            print('GT shape',data[1].shape)
            plot_data(
                data[0],  # img_array
                data[1],  # GT
                dataset,
                f"MStex Data - {image_name}"
            )


    elif dataset == 'DIBCO':
        data = _load_dibco(image_name, crop_h = crop_h, crop_w = crop_w, base_path = base_path, norm = norm)
        if verbose:
            print('Image shape', data[0].shape)
            print('GT shape',data[1].shape)
            plot_data(
                data[0],  # img_array
                data[1],  # GT
                dataset,
                f"Dibco Data - {image_name}"
            )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return data






def _load_dibco(image_name: str,  base_path: Path, norm: bool, crop_h: Optional[Tuple[int, int]] = (5,-5), crop_w: Optional[Tuple[int, int]] = (5,-5)) -> Tuple[np.ndarray, np.ndarray]:
    """Load MStex dataset images and masks"""
    # Load ground truth


    if base_path:
        gt_path = base_path / 'data' / 'DIBCO'/ 'GT'/ f'{image_name}_gt.bmp'

    else:
        gt_path = Path(f'data/DIBCO/GT/{image_name}_gt.bmp')

    if crop_h or crop_w:
        img = np.array(Image.open(gt_path))[crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]].astype(int)

    else:
        img = np.array(Image.open(gt_path))
    



    if len(np.unique(img)) > 2:
        img[img >= 127] = 255
        img[img < 127] = 0
    
    GT = 1 - img/np.max(img)
    if np.mean(GT) >= 0.5:
        GT = 1 - GT
    
    if len(GT.shape) == 3:
        GT = GT[:,:,0]
    
    # Load MS images
    if base_path:
        im_path = base_path / 'data' / 'DIBCO'/ 'RGB'/ f'{image_name}.bmp'

    else:
        im_path = Path(f'data/DIBCO/RGB/{image_name}.bmp')

    if crop_h or crop_w:
        img = np.array(Image.open(im_path)).astype('float32')[crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]].transpose((2,0,1))
    else: 
        img = np.array(Image.open(im_path)).astype('float32').transpose((2,0,1))

    if norm:
        img_array = minmax_outliers(img, 0, 100)
    
    else:
        img_array = img
    
    return img_array, GT





def _load_msbin(image_name: str, base_path: Path, norm: bool, crop_h: Optional[Tuple[int, int]] = None, crop_w: Optional[Tuple[int, int]] = None,verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MSBin dataset images and masks"""
    # Load label

    if base_path:
        file_gt = base_path / 'data' / 'MSBin'/ 'labels'/ f'{image_name}.png'

    else:
        file_gt = Path(f'data/MSBin/labels/{image_name}.png')


    if crop_h or crop_w:
        gt_rgb = np.array(Image.open(file_gt)).astype(int).transpose((2,0,1))[:,crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
    else:
        gt_rgb = np.array(Image.open(file_gt)).astype(int).transpose((2,0,1))
    # Create ground truth masks
    GT1 = np.zeros(gt_rgb[0].shape)
    GT2 = np.zeros(gt_rgb[0].shape)
    Mask = np.zeros(gt_rgb[0].shape)
    BG = np.zeros(gt_rgb[0].shape)
    
    GT1[gt_rgb[0] == 255] = 1
    GT2[gt_rgb[0] == 122] = 1
    Mask[(gt_rgb[2] == 255) & (gt_rgb[0] == 0)] = 1
    BG[(gt_rgb[2] == 0) & (gt_rgb[0] == 0) & (gt_rgb[1] == 0)] = 1
    
    # Load image data

    if base_path:
        file_im = base_path / 'data' / 'MSBin'/ 'images/'

    else:
        file_im = Path('data/MSBin/images/')

    subdirectories = sorted(list(file_im.glob(f'{image_name}_*.png')),
                       key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
    
    images_list = []
    for image_path in subdirectories:
        
        if crop_h or crop_w:
            img = np.array(Image.open(image_path)).astype('float32')[crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
        else:
            img = np.array(Image.open(image_path)).astype('float32')
        images_list.append(img)
        
    img = np.array(images_list)

    if norm:
        img_array = minmax_outliers(img, 0, 100)
    else:
        img_array = img

    return img_array, GT1, GT2, Mask, BG

def _load_mstex(image_name: str, base_path: Path, norm: bool, crop_h: Optional[Tuple[int, int]] = (5,-5), crop_w: Optional[Tuple[int, int]] = (5,-5)) -> Tuple[np.ndarray, np.ndarray]:
    """Load MStex dataset images and masks"""
    
    
    # Load ground truth

    if base_path:
        gt_path = base_path / 'data' / 'MSI-dataset'/ 'GT' / f'{image_name}GT.png'

    else:
        gt_path = Path(f'data/MSI-dataset/GT/{image_name}GT.png')


    
    if crop_h or crop_w:
        img = np.array(Image.open(gt_path))[crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]].astype(int)

    else:
        img = np.array(Image.open(gt_path))
    
    if len(np.unique(img)) > 2:
        img[img >= 127] = 255
        img[img < 127] = 0
    
    GT = 1 - img/np.max(img)
    if np.mean(GT) >= 0.5:
        GT = 1 - GT
    
    if len(GT.shape) == 3:
        GT = GT[:,:,0]
    
    # Load MS images

    if base_path:
        im_path = base_path / 'data' / 'MSI-dataset'/ 'MSI' / f'{image_name}/'

    else:
        im_path = Path(f'data/MSI-dataset/MSI/{image_name}/')

    
    files = sorted(list(im_path.glob('*.png')))
    
    images_list = []
    for file in files:


        if crop_h or crop_w:
            img = np.array(Image.open(file)).astype('float32')[crop_h[0]:crop_h[1],crop_w[0]:crop_w[1]]
        else: 
            img = np.array(Image.open(file)).astype('float32')

        images_list.append(img)

    img = np.array(images_list)
    if norm:
        img_array = minmax_outliers(img, 0, 100)
    else:
        img_array = img
    
    return img_array, GT

