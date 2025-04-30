# Attention Guided Class Activation Maps for Boosting Weakly Supervised Semantic Segmentation

This is the open-source code for the [Attention Guided Class Activation Maps for Boosting Weakly Supervised Semantic Segmentation](https://www.sciencedirect.com/science/article/abs/pii/S0952197625005561).



## Prerequisite

#### 1. install dependencies

```
pip install -r requirements.txt
```

#### 2. data preparation

The dataset is sourced from http://host.robots.ox.ac.uk/pascal/VOC and http://cocodataset.org.

#### 3.weights

You can download our weights  [here](https://drive.google.com/file/d/1hEBrPccnAbk1myN9P38QcRUvNdxITKag/view?usp=drive_link).



## Usage

#### 1. training

python train_AGCAM.py

#### 2. evaluation

python infer_AGCAM.com

python evaluation.py

#### 3.visualize.py

python visualize.py



## Citation

```
@article{li2025attention,
  title={Attention Guided Class Activation Maps for Boosting Weakly Supervised Semantic Segmentation},
  author={Li, Junhui and Zhu, Lei and Wang, Wenwu and Gong, Yin},
  journal={Engineering Applications of Artificial Intelligence},
  volume={151},
  pages={110556},
  year={2025},
  publisher={Elsevier}
}
```



## Acknowledge

This repo is developed based on SIPE [1] and TransCAM [2].

[1] Qi Chen, et al. "Self-Supervised Image-Specific Prototype Exploration for Weakly Supervised Semantic Segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022.

[2] Ruiwen Li, et al. "TransCAM: Transformer Attention-based CAM Refinement for Weakly Supervised Semantic Segmentation." Journal of Visual Communication and Image Representation. 2023.
