# SqueezeNet：用 Fire 模块压缩参数量

## 1. 出处、提出者与意义

Forrest N. Iandola 等人在 2016 年 [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360) 中提出 Fire 模块，以较少参数获得有竞争力的图像分类能力。核心做法是先用少量 1×1 卷积“挤压”通道，再用 1×1 与 3×3 两个分支“展开”特征。

## 2. 源码导航与直观解释

[squeezenet.py](squeezenet.py) 的 `Fire` 先经过 `squeeze`，然后 `expand_1` 和 `expand_3` 分别提取逐位置与邻域信息，最后在通道维拼接。`SqueezeNet` 串联多个 Fire 模块，最后用 1×1 分类卷积和全局平均。运行 `python SqueezeNet/squeezenet.py`，可观察 `[B,3,32,32]→[B,类别数]`。这是缩小后的教学版本，并未声称达到论文报告的模型体积或准确率。

## 3. 数学解释

若输入通道为 `C_in`、先挤压到 `C_s`，则 3×3 展开分支的参数约为 `9·C_s·C_e`，小于直接从 `C_in` 到 `C_e` 的 `9·C_in·C_e`（当 `C_s≪C_in`）。1×1 展开分支另有约 `C_s·C_e` 个参数。两支拼接后输出 `2C_e` 个通道；源码中 `Fire(16,4,16)` 因此输出 32 通道。

Fire 模块刻意让 squeeze 的通道数远小于 expand 分支总输出通道数。若把 squeeze=4 改成 squeeze=32，三乘三分支的参数会明显增加，便失去“先压缩再展开”的主要节省。两个展开分支的输出在通道维拼接，不能逐元素相加，因为它们代表不同的特征族。
