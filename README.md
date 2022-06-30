# RCAN
[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

[论文源码地址](https://github.com/yulunzhang/RCAN)

本 repository 复现 RCAN，并修改 RCAN 的模型结构，使之能够在训练和推断时绕过连续的 res groups，建立每个 res group 到最后层的旁路。在训练时以 50% 的概率遍历所有层（用于训练原始路径），以 50% 的概率在剩余的旁路中随机均匀地选择一条。在测试时可以只使用模型的一部分来进行推断，并随着使用的 res group 的增加，图像质量的增益增加。

## 模型结构
每个模型的 body 由多个 residual groups 构成，每个 residual group 结构一样，由多个 residual channel attention blocks 构成。

![模型结构](https://github.com/yulunzhang/RCAN/raw/master/Figs/RCAN.PNG)

## 参数

- residual group：10
- residual channel attention block：20
- patch_size：48 * 48
- batch size：16
- iteration：1000，每个 epoch 的迭代次数
- epoch：300
- lr：1e-4
- lr_decay：200，lr 每 2*10^5 个 iteration（即 200 个 epoch）减一半
- optimizer：ADAM (gamma=0.5, beta=[0.9, 0.999], epsilon=1e-8, weight_decay=0)
- loss：L1

## 数据集

### 训练数据集
- DIV2K：1-800 张图片，低分辨率图片由高分辨率图片通过 Bicubic 下采样生成
- scale：2, 3, 4
- 数据增强：旋转，上下翻转，左右翻转
- 划分 patch：训练时每张图片随机 crop 一个 patch
- 验证集：取训练集中 10 张图片，每 1 轮验证一次，并计算在每一个 residual group 跳出的增益，即 PSNR

### 测试数据集

- DIV2K（valid）：100 张图片
- B100：100 张图片
- Set5：5 张图片
- Set14：14 张图片
- Urban100：100 张图片

## 评估指标

- PSNR（三通道）
- PSNR（Y通道）
- SSIM
- MSE
- time
