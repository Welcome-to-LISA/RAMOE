# RAMoE
PyTorch code for paper [Region-aware fusion for non-overlapping hyperspectral and multispectral remote sensing images](https://doi.org/10.1016/j.isprsjprs.2026.04.045), ISPRS Journal of Photogrammetry and Remote Sensing, 2026.

## Abstract
Hyperspectral image fusion aims to reconstruct a high-resolution hyperspectral image from a high-resolution multispectral image and a low-resolution hyperspectral image. RAMoE follows an abundance-based fusion framework with dual-stream feature extraction, mixture-of-experts feature modeling, PSF-based spatial degradation, and SRF-based spectral degradation.

Main highlights:
* RAMoE uses dual-stream feature extraction for HR-MSI and LR-HSI.
* The fusion stage estimates abundance maps and reconstructs observations through learnable endmember decoders.
* The training loss combines reconstruction, abundance sum-to-one constraint, spatial consistency, and spectral consistency.
* The code supports fixed PSF generation and optional MoESR PSF pretraining.

## Installation
```
conda create -n RAMoE python=3.10
conda activate RAMoE
pip install -r requirements.txt
```

PyTorch installation depends on the CUDA version. If the default `torch` package is not suitable for your device, install the correct PyTorch build first, then install the remaining requirements.

## Data
Place data under:
```
data/
  MAT/
    PA/
      REF.mat
    TG/
      REF.mat
  SRF/
    PA.xls
    TG.xls
```

For real data without ground truth, use:
```
data/
  MAT/
    DATA_NAME/
      HR_MSI.mat
      LR_HSI.mat
  SRF/
    DATA_NAME.xls
```

Supported array keys include `REF`, `GT`, `HRHSI`, `HR_MSI`, and `LR_HSI`.

## Training
Train with the default config:
```
python main.py
```

Train PA:
```
python main.py --data_name PA
```

Train TG:
```
python main.py --data_name TG
```

Use a specific PSF type:
```
python main.py --psf_type matlab_gaussian
```

Available PSF types:
```
matlab_gaussian
standard_gaussian
motion_horizontal
motion_vertical
elliptical_45deg
defocus
```

Useful options:
```
python main.py --mask_lrhsi Yes --mask_ratio 0.5 --mask_direction left_to_right
python main.py --use_moesr_psf No
python main.py --data_type real
```

Outputs are saved under:
```
checkpoints/DATA_NAME_SF_SCALE/
```

## Citation
```
@article{yu2026region,
  title={Region-aware fusion for non-overlapping hyperspectral and multispectral remote sensing images},
  author={Yu, Haoyang and Huang, Baosen and Gao, Lianru and Plaza, Antonio and Zheng, Ke and Zhang, Bing},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={237},
  pages={452--466},
  year={2026},
  doi={10.1016/j.isprsjprs.2026.04.045}
}
```
