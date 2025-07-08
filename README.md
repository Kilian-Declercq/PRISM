[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)  [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)  [![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-b31b1b.svg)](https://arxiv.org/abs/<INDEX>)

# PRISM: Pruning for Rank-adaptive Interpretable Segmentation Model

**PRISM** is a novel segmentation model that integrates **Nonnegative Matrix Factorization (NMF)** within a deep neural architecture. Designed for historical document processing, PRISM automatically extracts meaningful and interpretable components from **multispectral images**, enhancing tasks such as **text binarization** and **image segmentation**. The framework has also demonstrated **generalization capabilities** to the **spectral unmixing** task, making it suitable for a broader range of remote sensing and imaging applications.

## A Prism for Vision

<p align="center">
  <img src="https://github.com/user-attachments/assets/d56ca3a3-65b6-43e0-977e-4528beff784b" alt="PRISM21 illustration" width="600">
</p>

Just like a physical **prism** splits white light into a spectrum of colors, the **PRISM model** decomposes complex spectral data into distinct, interpretable visual components with :

- ✅ Fully interpretable component extraction  
- ✅ Rank-adaptive pruning during training  
- ✅ Built-in NMF module with spectral constraints  
- ✅ Outperforms traditional SoTA binarization methods on challenging heritage documents

---

## Example: Historical Document Segmentation

<p align="center">
  <img src="https://github.com/user-attachments/assets/bd1d82d7-4747-4fec-8c65-f68158c51b0d" alt="Multispectral vs Decomposition" width="900">
</p>

From left to right:

1. **(a) Pseudo-color image** — raw multispectral image of a degraded manuscript  
2. **(b) SoTA binarization only (Howe)** — fails to extract faint or occluded text  
3. **(c) PRISM decomposition** — successfully separates text from background, enhancing readability and segmentation quality

---

## Datasets

### MStex

This dataset is dedicated to the document image binarization of multispectral images and is provided by the Synchromedia Laboratory.

[Link to MStex dataset](https://tc11.cvc.uab.es/datasets/SMADI_1)


### MSBin dataset

This dataset is dedicated to the (document image) binarization of multispectral images with MSBin standing for MultiSpectral Document Binarization. 

[Link to MSBIN GitHub page](https://github.com/hollaus/msbin) 

### DIBCO

DIBCO is the International Document Image Binarization Contest, organized since 2009 in conjunction with ICDAR - International Conference on Document Analysis and Recognition.  
DIBCO is the first effort made by the academic community to assess binarization algorithms and is widely acknowledged as providing important information about the quality of participating algorithms.

[Link to DIBCO data](https://dib.cin.ufpe.br/#!/resources/dibco) 



## Results (Placeholder)

| Dataset     | FM ↑     | DRD ↓    | NRM ↓    | PSNR ↑   |
|-------------|----------|----------|----------|----------|
| DIBCO 2013  | –        | –        | –        | –        |
| DIBCO 2016  | –        | –        | –        | –        |
| MSBin (Set A) | –      | –        | –        | –        |
| MSBin (Set B) | –      | –        | –        | –        |

> *↑: higher is better, ↓: lower is better.*


## Authors

- [Kilian Declercq](https://www.github.com/Kilian-Declercq)  
- [Mohamed Cheriet](https://profs.etsmtl.ca/mcheriet/)


## Links and Citation

📄 **[Read the preprint on arXiv](https://arxiv.org/abs/<INDEX>)**  
📦 Coming soon: results, dataset links, and pretrained models.
