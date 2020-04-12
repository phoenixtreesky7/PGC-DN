


# PGCDN

## Pyramid Global Context Aided Attentive U-Net for Image Dehazing

Dong Zhao1,2, Long Xu 1, Lin Ma 3, Jia Li 4, and Yihua Yan1

Key Laboratory of Solar Activity, National Astronomical Observatories, Chinese Academy of Sciences, Beijing 100101, China

University of Chinese Academy of Sciences, Beijing, 100049, China

Tencent AI Lab, Shenzhen, 518057, China

Beihang University, Beijing, 100191, China

# Abstract

During image capturing, atmospheric scattering and absorption would cause haze in recorded image. With haze in an image, scene visibility of the image would be severely compromised. Thus, image dehazing was raised for removing haze from hazy image. Within a hazy image, haze is not confined in a small local patch/position, while widely diffusing in a whole image. Under this circumstance, global context is a crucial factor of success of dehazing, which was however seldom investigated in existing dehazing algorithms. In the literature, global context (GC) block was designed to learn point-wise long-range dependencies of an image for global context modelling, however, patch-wise longrange dependencies was ignored. In image dehazing, patchwise depth which keeps depth constant/smooth within each patch is welcomed instead of pixel-wise depth which would cause oversaturation. Therefore, we are mostly concerned with patch-wise long-range dependencies which can better adapt to patch-wise depth. In this paper, we extend point-wise GC block into a Pyramid Global Context (PGC) block, where long-range dependency is modeled as a multi-scale form by using pyramid pooling method. Then, we plug the proposed PGC into a UNet, creating an attentive U-Net. Furthermore, to avoid overfitting of a deep model and extend receptive field of convolution kernel, residual neural network and dilated convolution are both employed. Thus, the finalized dehazing model can capture both long-range and patch-wise content dependencies on the one hand, on the other hand, it has the merits of fully exploiting hierarchical features and enlarging receptive field. The extensive comparisons on synthetic databases and real-world hazy images demonstrate the superiority of our model over other representative state-ofthe-art models with respect to restoration accuracy.

# Run the code

## train:

CUDA_VISIBALE_DEVICES=0,1,2,3 python train.py --netG onemsgc_drun_pl --gpu_ids 0,1,2,3 --gc_w 64

## test:

python tesy_my.py --netG onemsgc_drun_pl --which_epoch 50

# Acknowledgement

The code is built on EPDN (Pytorch) and GCNet (Pytorch). We thank the authors for sharing the codes.
