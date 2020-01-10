# AdvCalib
# Adversarial Calibrated Loss for Semi-Supervised Semantic Segmentation


This repo is about the extented research of the following paper:

[Adversarial Learning for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/1802.07934) <br/>
[Wei-Chih Hung](https://hfslyc.github.io/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/home), Yan-Ting Liou, [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/), and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/) <br/>
Proceedings of the British Machine Vision Conference (BMVC), 2018.

The output of the segmentation network from unlabeld images can be used as pseudo labels for semi-supervised semantic segmentation. These pseudo labels are based on the supervisory signal, the confidence from the discriminator network. I focused on the fact that pixels with higher confidence have more supervisory information, so I imposed more weights to them. However, the confidence was not calibrated well, i.e. higher confidence does not always guarantee higher accuracy, so network calibration by [temperature scaling](https://dl.acm.org/doi/pdf/10.5555/3305381.3305518?download=true) was required. Finally, weights are decided by the accuracies from the calibrated network.

Furthermore, imposing virtual adversarial noise makes the classification network be more robust to noise, and gives higher performance. I changed [this method](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8417973) for the segmentation network.


The code are heavily borrowed from the pytorch implementation of the original work ([Link](https://github.com/hfslyc/AdvSemiSeg)), temperature scaling ([Link](https://github.com/gpleiss/temperature_scaling)), and Virtual Adversarail Training (VAT) ([Link] (https://github.com/9310gaurav/virtual-adversarial-training)).

## Prerequisite

* CUDA 9.0 / CUDNN 7.6.4
* torchvision
* python 2.7
* pytorch 0.4.0 
* python-opencv >=3.4.2 
