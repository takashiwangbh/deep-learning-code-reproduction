# NeRF：从相机射线学习连续三维场景

## 1. 出处、提出者与意义

Ben Mildenhall、Pratul P. Srinivasan、Matthew Tancik、Jonathan T. Barron、Ravi Ramamoorthi、Ren Ng 在 2020 年提出 [《NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis》](https://arxiv.org/abs/2003.08934)。给定若干张有相机位姿的照片，NeRF 用一个连续函数表示场景：查询三维位置和观看方向，得到该处的密度与颜色；再沿相机射线累积，就能合成新视角的像素。它把三维表示、位置编码和可微体渲染连成一个可训练系统。

## 2. 源码导航与直观解释

从 [nerf.py](nerf.py) 的 `FourierFeatures` 开始：不同频率的正弦、余弦使 MLP 能描述细节。`NeRF.forward` 把位置编码送进主干，输出密度 `sigma`；颜色支路还会读取观看方向，因此同一点从不同方向可有不同外观。`render_rays` 在每条射线上按由近到远的深度取样，逐点调用网络，计算透明度、剩余透光量和加权颜色。输入为 `R` 条射线的原点/方向 `[R,3]` 及深度 `[R,S]`，返回 RGB `[R,3]`、深度与不透明度 `[R]`。运行 `python NeRF/nerf.py` 可检查前向和梯度。

这里没有相机数据加载、分层采样、粗细双网络或论文图像训练；随机参数只能合成无意义颜色。取样深度须沿射线递增，实际重建要用真实照片和已知相机姿态监督 RGB。

## 3. 数学解释

位置编码为 `γ(x)=[x,sin(2⁰πx),cos(2⁰πx),…,sin(2ᴸ⁻¹πx),cos(2ᴸ⁻¹πx)]`。射线第 `i` 个点的位置 `p_i=o+t_i d`；网络给出颜色 `c_i` 和非负密度 `σ_i`。在采样间隔 `δ_i` 内，光线被该点吸收的比例是 `α_i=1-exp(-σ_i δ_i)`。抵达它之前的透光率 `T_i=∏_{j<i}(1-α_j)`，所以它对像素的贡献权重 `w_i=T_i α_i`。最终 `C(r)=Σ_i w_i c_i`，深度同理为 `Σ_i w_i t_i`。`render_rays` 用独占当前位置的累积连乘实现 `T_i`，使前面的点正确遮住后面的点；整个颜色累积对 MLP 参数可微。
