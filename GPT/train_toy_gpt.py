"""零下载的字符级语言模型训练示例：python GPT/train_toy_gpt.py。

重复句子只是为了演示 token 化、右移标签、损失和参数更新。
"""

import torch
from torch.nn import functional as F

from gpt import GPT


def main() -> None:
    torch.manual_seed(11)
    torch.set_num_threads(1)
    corpus = "深度学习从基础开始，逐步理解注意力和生成模型。" * 60
    alphabet = sorted(set(corpus))
    char_to_id = {char: index for index, char in enumerate(alphabet)}
    data = torch.tensor([char_to_id[char] for char in corpus], dtype=torch.long)
    context = 12
    model = GPT(len(alphabet), block_size=32, d_model=32, num_heads=4, num_layers=2, dropout=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    model.train()
    for step in range(101):
        starts = torch.randint(0, len(data) - context - 1, (16,))
        x = torch.stack([data[index : index + context] for index in starts])
        y = torch.stack([data[index + 1 : index + context + 1] for index in starts])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(alphabet)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            print(f"step={step:3d} loss={loss.item():.3f}")

    model.eval()
    prompt = torch.tensor([[char_to_id["深"], char_to_id["度"]]])
    generated = model.generate(prompt, max_new_tokens=20)[0].tolist()
    print("生成示例：", "".join(alphabet[index] for index in generated))


if __name__ == "__main__":
    main()
