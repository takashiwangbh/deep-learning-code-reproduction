# 深度学习源码复现

每个模型或机制都有独立文件夹。点击项目名称会进入文件夹，先看到源码，再按其中的 README 阅读出处、源码导航和数学解释。建议从基础网络开始，也可以直接选感兴趣的方向。

## 基础网络与训练机制

- [Perceptron（感知机）](Perceptron/)
- [MLP（多层感知机）](Multilayer%20Perceptron/)
- [CNN（卷积神经网络）](CNN/)
- [Siamese Network（孪生网络）](Siamese%20Network/)
- [Residual Learning（残差学习）](Residual%20Learning/)
- [Self-Attention（自注意力与多头注意力）](Self-Attention/)
- [Hopfield Network](Hopfield%20Network/)
- [Restricted Boltzmann Machine（RBM）](Restricted%20Boltzmann%20Machine/)
- [Deep Belief Network（DBN）](Deep%20Belief%20Network/)
- [Highway Network](Highway%20Network/)
- [Dropout](Dropout/)
- [Optimization Algorithms（优化算法）](Optimization%20Algorithms/)
- [Normalization Layers（归一化层）](Normalization%20Layers/)

## 卷积视觉骨干

- [LeNet-5](LeNet-5/)
- [AlexNet](AlexNet/)
- [VGG](VGG/)
- [NiN（Network in Network）](NiN/)
- [GoogLeNet / Inception](GoogLeNet/)
- [ResNet-34](Residual%20Network/)
- [ResNet-50](ResNet-50/)
- [DenseNet](DenseNet/)
- [ResNeXt](ResNeXt/)
- [SENet](SENet/)
- [SqueezeNet](SqueezeNet/)
- [MobileNet](MobileNet/)
- [MobileNetV2](MobileNetV2/)
- [ShuffleNet](ShuffleNet/)
- [Xception](Xception/)
- [EfficientNet](EfficientNet/)
- [ConvNeXt](ConvNeXt/)

## 视觉 Transformer 与图像表示

- [Vision Transformer（ViT）](Vision%20Transformer/)
- [DeiT](DeiT/)
- [Swin Transformer](Swin%20Transformer/)
- [PVT（Pyramid Vision Transformer）](PVT/)
- [MLP-Mixer](MLP-Mixer/)
- [Perceiver](Perceiver/)

## 目标检测与图像分割

- [Feature Pyramid Network（FPN）](Feature%20Pyramid%20Network/)
- [Faster R-CNN](Faster%20R-CNN/)
- [YOLOv1](YOLOv1/)
- [SSD](SSD/)
- [RetinaNet](RetinaNet/)
- [DETR](DETR/)
- [FCN（全卷积网络）](FCN/)
- [U-Net](U-Net/)
- [SegNet](SegNet/)
- [DeepLab V3+](DeepLab%20V3%2B/)
- [PSPNet](PSPNet/)
- [Mask R-CNN](Mask%20R-CNN/)

## 循环网络与词表示

- [Vanilla RNN](Vanilla%20RNN/)
- [LSTM](LSTM/)
- [GRU](GRU/)
- [RNN / LSTM / GRU 对照](Recurrent%20Neural%20Networks/)
- [Word2Vec](Word2Vec/)
- [GloVe](GloVe/)
- [TextCNN](TextCNN/)
- [BiLSTM](BiLSTM/)
- [Seq2Seq](Seq2Seq/)
- [Attention Seq2Seq](Attention%20Seq2Seq/)
- [Bahdanau Attention](Bahdanau%20Attention/)
- [ELMo](ELMo/)
- [ULMFiT](ULMFiT/)

## Transformer 与语言模型

- [Transformer](Transformer/)
- [BERT](BERT/)
- [RoBERTa](RoBERTa/)
- [GPT](GPT/)
- [T5](T5/)
- [BART](BART/)
- [XLNet](XLNet/)
- [ELECTRA](ELECTRA/)
- [Transformer-XL](Transformer-XL/)
- [Longformer](Longformer/)
- [Linformer](Linformer/)
- [Performer](Performer/)
- [Switch Transformer](Switch%20Transformer/)
- [Llama-style Language Model](Llama-style%20Language%20Model/)
- [LoRA](LoRA/)
- [Modern Decoder](Modern%20Decoder/)
- [Mamba](Mamba/)
- [Gated DeltaNet](Gated%20DeltaNet/)
- [Gated DeltaNet-2](Gated%20DeltaNet-2/)
- [DPO](DPO/)

## 时间序列

- [TCN](TCN/)
- [WaveNet](WaveNet/)
- [DeepAR](DeepAR/)
- [N-BEATS](N-BEATS/)
- [Temporal Fusion Transformer](Temporal%20Fusion%20Transformer/)
- [PatchTST](PatchTST/)

## 生成模型

- [Autoencoder](Autoencoder/)
- [Denoising Autoencoder（DAE）](Denoising%20Autoencoder/)
- [Variational Autoencoder（VAE）](Variational%20Autoencoder/)
- [Conditional Variational Autoencoder（CVAE）](Conditional%20Variational%20Autoencoder/)
- [VQ-VAE](VQ-VAE/)
- [RealNVP](RealNVP/)
- [Glow](Glow/)
- [PixelCNN](PixelCNN/)
- [GAN](GAN/)
- [DCGAN](DCGAN/)
- [WGAN-GP](WGAN-GP/)
- [CGAN](CGAN/)
- [pix2pix](pix2pix/)
- [CycleGAN](CycleGAN/)
- [SRGAN](SRGAN/)
- [StyleGAN2](StyleGAN2/)
- [DDPM](DDPM/)
- [DDIM](DDIM/)
- [Latent Diffusion（LDM）](Latent%20Diffusion/)
- [DiT](DiT/)
- [Flow Matching](Flow%20Matching/)
- [Score-SDE](Score-SDE/)
- [Classifier-Free Guidance](Classifier-Free%20Guidance/)
- [ControlNet](ControlNet/)

## 图学习与三维表示

- [GCN](GCN/)
- [GraphSAGE](GraphSAGE/)
- [GAT](GAT/)
- [GIN](GIN/)
- [Graphormer](Graphormer/)
- [PointNet](PointNet/)
- [PointNet++](PointNet%2B%2B/)
- [DGCNN](DGCNN/)
- [NeRF](NeRF/)

## 自监督与多模态

- [SimCLR](SimCLR/)
- [MoCo](MoCo/)
- [BYOL](BYOL/)
- [DINO](DINO/)
- [MAE](MAE/)
- [Barlow Twins](Barlow%20Twins/)
- [VICReg](VICReg/)
- [CLIP](CLIP/)
- [BLIP](BLIP/)

## 深度强化学习

- [REINFORCE](REINFORCE/)
- [DQN](DQN/)
- [Double DQN](Double%20DQN/)
- [A2C](A2C/)
- [PPO](PPO/)
- [DDPG](DDPG/)
- [TD3](TD3/)
- [SAC](SAC/)
- [AlphaZero](AlphaZero/)

## 推荐系统

- [Factorization Machines（因子分解机）](Factorization%20Machines/)
- [NCF](NCF/)
- [Wide & Deep](Wide%20and%20Deep/)
- [DeepFM](DeepFM/)
- [DIN](DIN/)
- [SASRec](SASRec/)

建议使用 Python 3.10 或更新版本。安装依赖见 `requirements.txt`；运行 `python check_all.py` 检查源码。示例使用小规模随机输入，没有预训练权重或论文性能指标。
