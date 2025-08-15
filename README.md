# IGReg: Mitigating Gradient Conflicts for Multi-Task Glioma Phenotyping and Grading via Implicit Regularization

![IGReg Network Architecture](IGreg.jpg)

This repository provides the **PyTorch implementation** of the paper:  
**"Mitigating Gradient Conflicts for Multi-Task Glioma Phenotyping and Grading via Implicit Regularization (IGReg)"**.

## Method Overview

IGReg integrates two plug-and-play modules into a standard multi-task learning (MTL) framework:

### 1. Dynamic Prototype Alignment (DPA)
- Constructs a semantic prototype space using an **auxiliary segmentation dataset** (different from the glioma multi-task classification dataset).
- Aligns task-specific features from the MTL framework with relevant prototypes via contrastive learning.
- Enhances feature consistency and reduces task-specific gradient noise.

### 2. Surrogate Task-Dominant Gradient Projection (STDGP)
- Preserves the dominant gradient direction for the current task by orthogonally projecting gradients from other tasks.
- Ensures the extracted features remain highly discriminative.
- Introduces a conditional regularization loss to enforce similarity between high-confidence surrogate-derived features and primary classification features.

## Backbone Network
The backbone of IGReg is adapted from the [MA-MTLN](https://github.com/infinite-tao/MA-MTLN) architecture proposed in:  
> Zhang, Yongtao, Li, Haimei, Du, Jie, Qin, Jing, Wang, Tianfu, Chen, Yue, Liu, Bing, Gao, Wenwen, Ma, Guolin, & Lei, Baiying. (2021). 3D multi-attention guided multi-task learning network for automatic gastric tumor segmentation and lymph node classification. *IEEE Transactions on Medical Imaging*, 40(6), 1618–1631.

## Data & Preprocessing
- The **auxiliary segmentation dataset** used in the DPA module comes from a source different from the glioma multi-task classification dataset, and is specifically introduced to alleviate gradient noise across tasks.
- Data acquisition, preprocessing, and base model construction follow our previously released code:  
🔗 [CMTLNet Repository](https://github.com/ChiChienMile/CMTLNet/)

## Dependencies
IGReg requires the **Lion optimizer**:
```bash
pip install lion-pytorch
```
🔗 [Lion Optimizer](https://github.com/lucidrains/lion-pytorch)

## Usage

### Full Training
Run the training script:
```bash
python Train_IGReg.py
```
> **Note**: Requires configuring the data loader. Refer to [CMTLNet](https://github.com/ChiChienMile/CMTLNet/) for the code structure.

### Quick Model I/O Demo
Run the demo script to quickly test model input/output:
```bash
python model.IGReg.py
```

## Citation
If you find this work useful, please consider citing our paper:
```bibtex
@article{chen2025103435,
  title = {Cooperative multi-task learning and interpretable image biomarkers for glioma grading and molecular subtyping},
  journal = {Medical Image Analysis},
  pages = {103435},
  year = {2025},
  issn = {1361-8415},
  doi = {https://doi.org/10.1016/j.media.2024.103435},
  url = {https://www.sciencedirect.com/science/article/pii/S1361841524003608},
  author = {Qijian Chen and Lihui Wang and Zeyu Deng and Rongpin Wang and Li Wang and Caiqing Jian and Yue-Min Zhu}
}
```

Please also cite the backbone network source:
```bibtex
@article{zhang20213d,
  title={3D multi-attention guided multi-task learning network for automatic gastric tumor segmentation and lymph node classification},
  author={Zhang, Yongtao and Li, Haimei and Du, Jie and Qin, Jing and Wang, Tianfu and Chen, Yue and Liu, Bing and Gao, Wenwen and Ma, Guolin and Lei, Baiying},
  journal={IEEE transactions on medical imaging},
  volume={40},
  number={6},
  pages={1618--1631},
  year={2021},
  publisher={IEEE}
}
```
