# ShuffleNet v1：组卷积后打乱通道

## 1. 出处、提出者与意义

Xiangyu Zhang、Xinyu Zhou、Mengxiao Lin、Jian Sun 在 2017 年 [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083) 中提出：1×1 卷积虽比 3×3 轻，但在极低计算预算下仍占大头；将其分组能省算力，然而不同组之间会缺少信息交流。通道打乱就是为了解决这件事。

## 2. 源码导航与直观解释

[shufflenet.py](shufflenet.py) 的 `ShuffleUnit` 先做分组 1×1 卷积，再调用 `channel_shuffle` 交换组内/组间通道排列，随后做深度卷积和第二个分组 1×1 卷积。步长 1 时残差相加；步长 2 时主分支与池化捷径拼接。`ShuffleNet` 是缩小后的分类例子。运行 `python ShuffleNet/shufflenet.py`；输入 `[B,3,32,32]`，输出 `[B,类别数]`。

## 3. 数学解释

把 `C` 个通道写成 `[组数 g, 每组 C/g]`，转置这两个轴，再展平，就得到通道打乱。原本同一组的相邻通道会进入下一层不同的组，因此信息可以跨组传播。组卷积参数约为普通 1×1 卷积的 `1/g`，但若连续多层都不打乱，组之间永远彼此隔离。源码的 `reshape→transpose→reshape` 正是这个排列操作，它本身没有可学习参数。

通道打乱可用一个八通道例子检查：先分为两组 [0,1,2,3] 和 [4,5,6,7]，打乱后相邻输出会交错来自不同原组。源码检查确保数值集合完全不变，只改变次序。打乱本身零参数，但紧接着的分组卷积便能接触到前一层别组的信息，这才是操作的作用。
