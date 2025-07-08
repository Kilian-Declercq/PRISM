[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)  [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)  [![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-b31b1b.svg)](https://arxiv.org/abs/<INDEX>)

# **PRISM**: Pruning for Rank-adaptive Interpretable Segmentation Model

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
![prismtext](https://github.com/user-attachments/assets/8d0b5bb2-8e9f-4984-b014-e4ffd6b5b54d)

From left to right:
**(a) Pseudo-color image** — raw multispectral image of a degraded manuscript  
**(b) SoTA binarization only (Howe)** — fails to extract faint or occluded text  
**(c) PRISM decomposition** — successfully separates text from background, enhancing readability and segmentation quality

---

## Datasets
### MS-TEx (SMADI)

The **MS-TEx** dataset (Synchromedia Multispectral Ancient Document Images) is provided by the Synchromedia Laboratory.  
It consists of **multispectral images of historical documents** specifically designed for benchmarking text extraction and binarization algorithms under varying degradations.

🔗 [Link to MS-TEx dataset](https://tc11.cvc.uab.es/datasets/SMADI_1)

### MSBin

The **MSBin** dataset focuses on **multispectral document binarization**, containing challenging degraded manuscripts captured under different spectral bands.  
It is a valuable benchmark for evaluating spectral decomposition and binarization performance.

🔗 [Link to MSBin GitHub page](https://github.com/hollaus/msbin)



## Results

| Dataset     | FM ↑     | DRD ↓    | NRM ↓    | PSNR ↑   |
|-------------|----------|----------|----------|----------|
| MStex  | –        | –        | –        | –        |
| MSBin EA  | –        | –        | –        | –        |
| MSBin BT | –      | –        | –        | –        |


> *↑: higher is better, ↓: lower is better.*


## Authors

- [Kilian Declercq](https://www.github.com/Kilian-Declercq)  
- [Mohamed Cheriet](https://profs.etsmtl.ca/mcheriet/)
- [Abderahmanne Rahiche]()


## Links and Citation

📄 **[Read the preprint on arXiv](https://arxiv.org/abs/<INDEX>)**  
📦 Coming soon: results metrics, visual results, training loops.
