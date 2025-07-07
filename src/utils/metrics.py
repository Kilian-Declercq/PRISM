import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import skeletonize
from scipy import signal


def ortho_reg_abd(latent_space: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """
    Compute orthogonality loss for 2D latent space.

    Args:
        latent_space: Input tensor of shape (batch_size, channels, height, width)
        verbose: Whether to print the difference matrix

    Returns:
        Scalar tensor representing the orthogonality loss
    """
    batch_size, channels, height, width = latent_space.size()
    
    latent_channels = latent_space.view(batch_size, channels, height * width) / (
        torch.linalg.vector_norm(latent_space.view(batch_size, channels, height * width), 
                               dim=2, keepdim=True) + 1e-8)
    
    gram_matrix = torch.bmm(latent_channels, latent_channels.transpose(1, 2))
    identity = torch.eye(channels, device=latent_space.device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    difference = torch.abs(gram_matrix - identity)

    
    if verbose:
        print(difference)
    
    ortho_reg = torch.sum(difference)

    return ortho_reg

class SAD_MSE_loss(nn.Module):
    """Spectral Angle Distance loss module"""
    def __init__(self):
        super(SAD_MSE_loss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        y_true_normed = F.normalize(y_true, dim=1)
        y_pred_normed = F.normalize(y_pred, dim=1)
        mse = self.mse(y_pred, y_true)
        sad = torch.acos(torch.clamp((y_true_normed * y_pred_normed).sum(dim=1), -1., 1.)).mean()

        return sad + 0.5 * mse


def sad_decoder(y_pred, y_true):
    y_true_normed = nn.functional.normalize(y_true, dim=0)
    y_pred_normed = nn.functional.normalize(y_pred, dim=0)
    sad_decoder = torch.pi*1/2 - torch.acos(torch.clamp((y_true_normed * y_pred_normed).sum(),-1.,1.))

    return sad_decoder




def pFM(abd, gt, p):

    gt_inverted = np.logical_not(gt)  # Invert the binary img
    SKL_GT_inverted = skeletonize(gt_inverted)  # skeletonize
    SKL_GT = np.logical_not(SKL_GT_inverted)  # Invert again

    # True Positive (TP): we predict a label of 0 (text), and the ground truth is 0.
    SKL_TP = np.sum(np.logical_and(abd == 0, SKL_GT == 0))

    # False Negative (FN): we predict a label of 1 (background), but the ground truth is 0 (text).
    SKL_FN = np.sum(np.logical_and(abd == 1, SKL_GT == 0))

    pseudo_r = SKL_TP / (SKL_FN + SKL_TP)

    return 100 * 2 * (p * pseudo_r) / (p + pseudo_r)
    



    



def NRM(TP,TN,FN,FP):

  if (TP + FN):
    NRfn = FN / (TP + FN)
  else:
    NRfn = 0
    print("There are no True positive and False Negative")

  if (FP + TN):
    NRfp = FP / (FP + TN)
  else:
    NRfp = 0
    print("There are no False positive and True Negative")

  NRM = (NRfn + NRfp) / 2
  return NRM

def F1_score(TP,TN,FN,FP):

  F1 = 2*TP / (2*TP + FP + FN)
  return F1



def accuracy(TP,TN,FN,FP):

  return (TP + TN)/(TP + TN + FP + FN)


def DRD(IM, GT, n = 2, NUBN_size = 8,verbose = True):
  
  D = IM^GT


  ##### m Initialization #####
  m = n*2+1

  ##### Kernel Calculation #####
  W = np.zeros((m,m))
  ic = jc = (m + 1)/2
  for i in range(m):
      for j in range(m):
        if i == ic - 1 and j == jc - 1  : 	W[i,j] = 0
        else: W[i,j] = 1/np.sqrt((i + 1 - ic)**2 + (j + 1 - jc)**2)
  W = W/np.sum(W)
  W.astype(float)

   

  
  #### DRD calculation #### 
  total_drd = 0
  flipped_coords = np.argwhere(D)
  center = m // 2


  for x, y in flipped_coords: 

    # Extract neighborhood
    neighborhood = GT[max(0, x-center):x+center+1, max(0, y-center):y+center+1]

    # Pad if necessary with IM value(for edge pixels)
    if neighborhood.shape != (m, m):
        padded = np.full((m, m), IM[x, y])
        padded[:neighborhood.shape[0], :neighborhood.shape[1]] = neighborhood
        neighborhood = padded

    # Difference between neighborhood value and flipped pixel value
    Dk = neighborhood ^ IM[x, y]

    # Calculate DRDk
    DRDk = np.sum(Dk * W)
    total_drd += DRDk


  DRD = total_drd

  ##### NUBN #####
  if GT.shape[0] >= NUBN_size and GT.shape[1] >= NUBN_size :

    # Convolve image with uniform kernel. If all zeros, convolution result = 0, if all ones, convolution results = NUBN_size**2 
    Wnubn = np.ones((NUBN_size,NUBN_size))
    Cnubn = signal.convolve2d(GT.astype(float),Wnubn,mode='valid')[::NUBN_size, ::NUBN_size]

    # Set all pixel NUBN_size**2 to zero
    Cnubn[Cnubn == NUBN_size**2] = 0

    # Count non-uniform kernels
    NUBN = np.count_nonzero(Cnubn)

  else: raise Exception("Ground-truth needs to be at least a 8x8 matrix")
  if not NUBN: raise Exception("The ground truth is uniform")
  
  return DRD/NUBN



def compute_metrics_binary(IM, GT,verbose = False, drd_b = True, pfm_b = False):

  if IM.dtype != bool or GT.dtype != bool or IM.shape != GT.shape:
    raise Exception("Image has not the same size than ground truth, or is not binary")
  
  TP = np.sum(( IM & GT ))
  TN = np.sum(( ~IM & ~GT))
  FN = np.sum(( ~IM & GT ))
  FP = np.sum(( IM & ~GT ))

  if pfm_b:
    precision = TP / (TP + FP)
    pfm = pFM(IM, GT, precision)

  nrm = NRM(TP,TN,FN,FP)
  fm = F1_score(TP,TN,FN,FP)
  acc = accuracy(TP,TN,FN,FP)

  if drd_b:
    drd = DRD(IM,GT)
  else:
    drd = 0

  if verbose:
    print("TP : ", TP)
    print("TN : ", TN)
    print("FP : ", FP)
    print("FN : ", FN)
    print("NRM : ", nrm)
    print("F1 : ", fm)
    print("ACC : ", acc)
    print("DRD : ", drd)

  if pfm_b: 
    return nrm, fm, acc, drd, pfm
  else:
    return nrm, fm, acc, drd