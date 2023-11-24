# MIPSFusion
This project is based on our SIGGRAPH Asia 2023 paper, [MIPS-Fusion: Multi-Implicit-Submaps for Scalable and Robust Online Neural RGB-D Reconstruction](https://arxiv.org/pdf/2308.08741.pdf)


## Introduction
MIPSFusion is a neural RGB-D SLAM method based on multi-implicit submap representation, which enables scalable online tracking and mapping for large indoor scenes. Based on divide-and-conquer mapping scheme, each submap is assigned to learn a sub-area of the scene, as shown by the colored bounding box, and a new submap will be created when scanning a new area. This incremental strategy ensures our method has the potential to reconstruct large scenes. Besides, our method can handle fast camera motion.
<img src="fig/1.png" alt="drawing" width="700"/>

## Installation
We recommend creat an annacoda environment from [environment.yaml](environment.yaml)

## Run
```
python main.py --config {config_file}
```

## Acknowledgement

## Citation
If you find our code or paper useful, please cite
```
@article{tang2023mips,
  title={MIPS-Fusion: Multi-Implicit-Submaps for Scalable and Robust Online Neural RGB-D Reconstruction},
  author={Tang, Yijie and Zhang, Jiazhao and Yu, Zhinan and Wang, He and Xu, Kai},
  journal={arXiv preprint arXiv:2308.08741},
  year={2023}
}
```
