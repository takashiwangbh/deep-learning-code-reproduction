"""完整 PyTorch 训练循环：合成二维分类数据，无需下载数据集。"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def main():
    torch.manual_seed(7)
    x = torch.randn(1024, 2)
    y = ((x[:, 0] ** 2 + x[:, 1] ** 2) > 1.5).long()
    train_loader = DataLoader(TensorDataset(x[:800], y[:800]), batch_size=64, shuffle=True)
    val_x, val_y = x[800:], y[800:]
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()  # 启用训练模式；BN / Dropout 会据此改变行为
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()           # 清空上一轮累积梯度
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()                 # autograd 执行链式求导
            optimizer.step()                # 根据梯度修改参数
        model.eval()
        with torch.no_grad():               # 验证不建反向传播计算图
            accuracy = (model(val_x).argmax(dim=-1) == val_y).float().mean().item()
        if epoch in (0, 4, 9, 19):
            print(f"epoch={epoch + 1:02d} train_loss={loss.item():.4f} val_acc={accuracy:.3f}")
    return model, accuracy


if __name__ == "__main__":
    main()
