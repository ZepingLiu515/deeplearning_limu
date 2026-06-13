# 第四章：多层感知机（MLP）— 完整学习笔记

> 本章是深度学习的入门核心章节，从线性模型过渡到真正的"深度"网络。
> 适合零基础小白，每一行代码都有详细解释。

---

## 目录

1. [多层感知机基础概念](#1-多层感知机基础概念)
2. [激活函数详解](#2-激活函数详解)
3. [MLP 从零开始实现](#3-mlp-从零开始实现)
4. [MLP 简洁实现（使用框架）](#4-mlp-简洁实现使用框架)
5. [模型选择、欠拟合和过拟合](#5-模型选择欠拟合和过拟合)
6. [权重衰减（L2 正则化）](#6-权重衰减l2-正则化)
7. [暂退法（Dropout）](#7-暂退法dropout)
8. [前向传播、反向传播和计算图](#8-前向传播反向传播和计算图)
9. [数值稳定性和模型初始化](#9-数值稳定性和模型初始化)
10. [环境和分布偏移](#10-环境和分布偏移)
11. [实战：Kaggle 房价预测](#11-实战kaggle-房价预测)

---

## 1. 多层感知机基础概念

### 1.1 为什么需要多层感知机？

在上一章我们学了 **softmax 回归**，它是一个"单层"的线性模型：
- 输入 → 一层线性变换 → 输出

但**线性模型有致命缺陷**：它只能表示"线性关系"。

**举几个线性模型搞不定的例子：**

| 场景 | 问题 |
|------|------|
| 预测还款能力 | 收入从 0→5万 和从 100万→105万，对还款能力的影响完全不同，不是线性关系 |
| 根据体温预测死亡率 | 37°C 以上越高越危险，37°C 以下越高越安全——违反单调性 |
| 猫狗图片分类 | 某个像素变亮，不一定就更像猫或更像狗——像素之间有复杂交互 |

**结论：现实世界的数据关系大多是"非线性"的，纯线性模型表达能力不够！**

### 1.2 什么是多层感知机（MLP）？

**MLP = 多层感知机（Multilayer Perceptron）**

简单说：在输入层和输出层之间，加了**隐藏层**，并且在隐藏层后面加了**激活函数**（非线性变换）。

```
输入层 → [隐藏层 + 激活函数] → 输出层
```

**结构图示意：**
```
输入 x (4个特征)
  │
  ├──→ 隐藏神经元1 ──┐
  ├──→ 隐藏神经元2 ──┤
  ├──→ 隐藏神经元3 ──┼──→ 输出 (3个类别)
  ├──→ 隐藏神经元4 ──┤
  └──→ 隐藏神经元5 ──┘
```

### 1.3 关键概念：从线性到非线性

**单隐藏层 MLP 的数学表达：**

$$
\mathbf{H} = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})
$$

$$
\mathbf{O} = \mathbf{H} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}
$$

**逐符号解释：**
- $\mathbf{X}$：输入数据矩阵，形状 `(样本数, 特征数)`
- $\mathbf{W}^{(1)}$：第一层（隐藏层）的权重矩阵
- $\mathbf{b}^{(1)}$：第一层的偏置
- $\sigma$：**激活函数**（比如 ReLU）——这是关键！
- $\mathbf{H}$：隐藏层的输出，也叫"隐藏表示"
- $\mathbf{W}^{(2)}$：第二层（输出层）的权重矩阵
- $\mathbf{b}^{(2)}$：第二层的偏置
- $\mathbf{O}$：最终输出

**为什么必须加激活函数？**

如果不加激活函数 $\sigma$，两层线性变换叠在一起还是线性变换：
$$
\mathbf{O} = (\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X}\mathbf{W} + \mathbf{b}
$$
这跟单层模型完全等价，白加了一层！

**加上激活函数后，模型就有了"非线性"能力，可以拟合复杂的函数。**

### 1.4 通用近似定理

> 只要有一个隐藏层、足够多的神经元、合适的权重，MLP 可以拟合**任意连续函数**。

这就像 C 语言理论上可以写任何程序，但"写出正确的程序"才是最难的部分。
实际中，我们通常用**更深（更多层）而不是更宽（更多神经元）**的网络。

---

## 2. 激活函数详解

激活函数是 MLP 的灵魂，它把"线性"变成"非线性"。

### 2.1 ReLU 函数（最常用）

**公式：**
$$
\text{ReLU}(x) = \max(x, 0)
$$

**通俗理解：** 正数保留，负数变 0。

```python
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)  # torch.relu() 就是 ReLU 函数
```

**代码逐行解释：**
- `torch.arange(-8.0, 8.0, 0.1)`：生成从 -8 到 8 的等差数列，步长 0.1
- `requires_grad=True`：告诉 PyTorch 需要计算梯度（用于反向传播）
- `torch.relu(x)`：对 x 的每个元素取 max(x, 0)

**ReLU 的导数：**
- 输入 > 0 时，导数 = 1
- 输入 < 0 时，导数 = 0
- 输入 = 0 时，不可导（实际中默认取 0）

```python
y.backward(torch.ones_like(x))  # 反向传播计算梯度
# x.grad 就是 ReLU 的导数
```

**为什么 ReLU 最受欢迎？**
1. 计算简单（就是取最大值）
2. 导数要么是 0 要么是 1，不会出现梯度消失问题
3. 实践效果好

### 2.2 Sigmoid 函数

**公式：**
$$
\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

**通俗理解：** 把任意实数"挤压"到 (0, 1) 之间。

```python
y = torch.sigmoid(x)  # 输出在 0 到 1 之间
```

**Sigmoid 的导数：**
$$
\text{sigmoid}'(x) = \text{sigmoid}(x) \cdot (1 - \text{sigmoid}(x))
$$

- 最大导数只有 0.25（在 x=0 处）
- 输入很大或很小时，导数接近 0 → **梯度消失问题！**

**现在 Sigmoid 主要用于：**
- 二分类问题的输出层（输出概率）
- 循环神经网络中的"门控"机制
- 隐藏层中已经很少用了（被 ReLU 取代）

### 2.3 Tanh 函数

**公式：**
$$
\text{tanh}(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}
$$

**通俗理解：** 把任意实数"挤压"到 (-1, 1) 之间，关于原点对称。

```python
y = torch.tanh(x)  # 输出在 -1 到 1 之间
```

**Tanh 的导数：**
$$
\text{tanh}'(x) = 1 - \text{tanh}^2(x)
$$

- 最大导数为 1（在 x=0 处）—— 比 Sigmoid 好
- 但同样存在梯度消失问题（输入远离 0 时导数接近 0）

### 2.4 三种激活函数对比

| 特性 | ReLU | Sigmoid | Tanh |
|------|------|---------|------|
| 输出范围 | [0, +∞) | (0, 1) | (-1, 1) |
| 最大导数 | 1 | 0.25 | 1 |
| 梯度消失 | 不会（正区间） | 严重 | 严重 |
| 计算速度 | 最快 | 较慢（要算 exp） | 较慢 |
| 使用场景 | 隐藏层首选 | 二分类输出层 | 较少使用 |

---

## 3. MLP 从零开始实现

这一节我们**不用任何高级 API**，纯手写一个 MLP，帮你理解每一步在干什么。

### 3.1 准备数据

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

**逐行解释：**
- `batch_size = 256`：每次训练取 256 张图片
- `d2l.load_data_fashion_mnist(batch_size)`：加载 Fashion-MNIST 数据集（28×28 灰度衣服图片，10 个类别），返回训练集和测试集的迭代器

### 3.2 初始化模型参数

```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

**逐行解释：**
- `num_inputs = 784`：输入特征数（28×28 = 784 个像素）
- `num_outputs = 10`：输出类别数（10 种衣服）
- `num_hiddens = 256`：隐藏层有 256 个神经元（超参数，可以调）
- `torch.randn(...)`：从标准正态分布随机生成
- `* 0.01`：乘以 0.01 让初始权重很小（避免梯度爆炸）
- `nn.Parameter(...)`：包装成 PyTorch 的"参数"，这样优化器才能更新它
- `requires_grad=True`：需要计算梯度
- `torch.zeros(...)`：偏置初始化为 0

**为什么要随机初始化？** 如果所有权重都一样，每个神经元学到的东西都一样，等于只有一个神经元（对称性问题）。

### 3.3 定义激活函数

```python
def relu(X):
    a = torch.zeros_like(X)      # 创建一个和 X 同形状的全 0 张量
    return torch.max(X, a)       # 逐元素取最大值：正数保留，负数变 0
```

**逐行解释：**
- `torch.zeros_like(X)`：创建和 X 形状相同的全 0 张量
- `torch.max(X, a)`：逐元素比较 X 和 a，取较大值。效果等价于 `max(X, 0)`

### 3.4 定义模型（前向传播）

```python
def net(X):
    X = X.reshape((-1, num_inputs))   # 把 28×28 的图片展平成 784 维向量
    H = relu(X @ W1 + b1)             # 隐藏层：线性变换 + ReLU 激活
    return (H @ W2 + b2)              # 输出层：线性变换（不加激活）
```

**逐行解释：**
- `X.reshape((-1, num_inputs))`：`-1` 表示自动计算 batch_size，把图片从 (batch, 28, 28) 变成 (batch, 784)
- `X @ W1`：`@` 是矩阵乘法，形状 (batch, 784) × (784, 256) = (batch, 256)
- `+ b1`：加上偏置，广播机制自动扩展
- `relu(...)`：通过 ReLU 激活函数，把负值变 0
- `H @ W2 + b2`：输出层，形状 (batch, 256) × (256, 10) = (batch, 10)

**数据流向：**
```
输入 (batch, 784)
  → X@W1 + b1  → (batch, 256)
  → relu()      → (batch, 256)  [负值变0]
  → H@W2 + b2  → (batch, 10)   [10个类别的得分]
```

### 3.5 定义损失函数

```python
loss = nn.CrossEntropyLoss(reduction='none')
```

**解释：**
- `nn.CrossEntropyLoss`：交叉熵损失函数，专门用于分类问题
- `reduction='none'`：不自动求平均，返回每个样本的损失值
- 交叉熵损失 = softmax + 负对数似然，把模型输出转换成概率再计算损失

### 3.6 训练模型

```python
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

**逐行解释：**
- `num_epochs = 10`：整个数据集训练 10 遍
- `lr = 0.1`：学习率，控制每次参数更新的步长
- `torch.optim.SGD(params, lr=lr)`：随机梯度下降优化器，负责更新参数
- `d2l.train_ch3(...)`：d2l 库封装好的训练函数，会自动完成前向传播、计算损失、反向传播、更新参数的循环

**训练过程（每一轮）：**
1. 取一个 batch 的数据
2. 前向传播：计算预测值
3. 计算损失：预测值 vs 真实标签
4. 反向传播：计算梯度
5. 更新参数：W = W - lr × 梯度
6. 重复直到所有 batch 都用完

---

## 4. MLP 简洁实现（使用框架）

上面从零实现是为了理解原理，实际中我们用 PyTorch 的高级 API 更简洁。

### 4.1 定义模型

```python
net = nn.Sequential(
    nn.Flatten(),           # 把 28×28 的图片展平成 784 维向量
    nn.Linear(784, 256),    # 全连接层：784 → 256
    nn.ReLU(),              # ReLU 激活函数
    nn.Linear(256, 10)      # 全连接层：256 → 10
)
```

**逐行解释：**
- `nn.Sequential(...)`：按顺序堆叠各层
- `nn.Flatten()`：展平操作，把多维输入变成一维
- `nn.Linear(784, 256)`：全连接层，输入 784 维，输出 256 维
- `nn.ReLU()`：ReLU 激活函数
- `nn.Linear(256, 10)`：输出层，256 维输入，10 维输出

**初始化权重：**
```python
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # 用正态分布初始化，标准差 0.01

net.apply(init_weights)  # 对 net 中的每一层应用 init_weights 函数
```

### 4.2 训练

```python
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

**和从零实现的对比：**
- 不需要手动定义 W1, b1, W2, b2
- 不需要手动写 relu 函数
- `net.parameters()` 自动获取所有参数
- 代码量从 20+ 行减少到 5 行

---

## 5. 模型选择、欠拟合和过拟合

### 5.1 核心问题：什么是好的模型？

我们训练模型的目标不是"在训练数据上表现好"，而是"在没见过的新数据上表现好"。

**类比考试：**
- 训练误差 = 做往年真题的分数
- 泛化误差 = 真正考试的分数
- 死记硬背真题答案（过拟合）→ 真考不好
- 理解知识点（好的泛化）→ 真考也好

### 5.2 三个关键概念

| 概念 | 含义 | 症状 |
|------|------|------|
| **欠拟合** | 模型太简单，没学到数据的规律 | 训练误差高，验证误差也高 |
| **过拟合** | 模型太复杂，把噪声也学进去了 | 训练误差很低，验证误差很高 |
| **刚好** | 模型复杂度适中 | 训练误差和验证误差都较低且接近 |

### 5.3 影响过拟合/欠拟合的因素

1. **模型复杂度**：参数越多、层数越多 → 越容易过拟合
2. **数据量**：数据越少 → 越容易过拟合
3. **训练时间**：训练越久 → 越容易过拟合

### 5.4 验证集和 K 折交叉验证

**数据三分法：**
```
全部数据 = 训练集（训练模型）+ 验证集（选超参数）+ 测试集（最终评估）
```

**K 折交叉验证（数据少时用）：**
1. 把训练数据分成 K 份
2. 每次用 K-1 份训练，1 份验证
3. 重复 K 次，取平均误差

### 5.5 多项式回归实验

通过一个简单的实验直观感受欠拟合和过拟合：

**生成数据（真实的三阶多项式）：**
$$
y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6\frac{x^3}{3!} + \epsilon
$$

```python
max_degree = 20  # 最大阶数
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 只有前4个系数非零

features = np.random.normal(size=(n_train + n_test, 1))  # 随机特征
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))  # x^0, x^1, x^2, ...
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # 除以 i! 防止数值太大
labels = np.dot(poly_features, true_w)  # y = X @ w
labels += np.random.normal(scale=0.1, size=labels.shape)  # 加噪声
```

**三种情况对比：**

| 情况 | 使用特征 | 结果 |
|------|----------|------|
| 三阶多项式（刚好） | 1, x, x², x³ | 训练和测试误差都低 ✅ |
| 线性（欠拟合） | 1, x | 训练和测试误差都高 ❌ |
| 20阶多项式（过拟合） | 全部 20 个特征 | 训练误差低，测试误差高 ❌ |

---

## 6. 权重衰减（L2 正则化）

### 6.1 什么是权重衰减？

**核心思想：** 在损失函数中加一个惩罚项，让权重不要太大。

**原始损失：**
$$
L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)})^2
$$

**加了权重衰减后：**
$$
L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2
$$

- $\lambda$：正则化强度（超参数），越大惩罚越重
- $\|\mathbf{w}\|^2$：权重的 L2 范数的平方（所有权重的平方和）

**通俗理解：** "我不光要预测得准，还要求权重尽量小。" 权重小 → 模型更简单 → 不容易过拟合。

### 6.2 为什么叫"权重衰减"？

看梯度下降的更新公式：
$$
\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \frac{\eta}{|\mathcal{B}|}\sum \mathbf{x}^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)})
$$

每次更新时，权重先乘以 $(1 - \eta\lambda) < 1$，相当于"衰减"了一点。

### 6.3 从零实现

```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2  # L2 范数的平方除以 2

def train(lambd):
    w, b = init_params()
    net = lambda X: d2l.linreg(X, w, b)
    loss = d2l.squared_loss
    num_epochs, lr = 100, 0.003
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)  # 损失 + 惩罚项
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
```

**关键改动：** 只有一行！`l = loss(net(X), y) + lambd * l2_penalty(w)`

**实验结果：**
- `lambd=0`（不使用权重衰减）：训练误差低，测试误差高（过拟合）
- `lambd=3`（使用权重衰减）：训练误差略高，测试误差降低（正则化生效）

### 6.4 简洁实现

```python
trainer = torch.optim.SGD([
    {"params": net[0].weight, 'weight_decay': wd},  # 权重使用权重衰减
    {"params": net[0].bias}],                        # 偏置不使用权重衰减
    lr=lr)
```

**注意：** 通常只对权重使用权重衰减，不对偏置使用。

---

## 7. 暂退法（Dropout）

### 7.1 什么是 Dropout？

**核心思想：** 在训练时，随机"丢弃"一些神经元（把它们的输出设为 0）。

**类比：** 想象一个团队做项目，如果每个人都可能随时"请假"，那每个人都不能过度依赖别人，必须自己也能独立完成任务。这样团队更健壮。

### 7.2 Dropout 的工作原理

对每个隐藏层的输出 $h$，以概率 $p$ 进行以下操作：

$$
h' = \begin{cases}
0 & \text{概率为 } p \\
\frac{h}{1-p} & \text{其他情况}
\end{cases}
$$

**为什么要除以 $(1-p)$？** 为了保持期望值不变：$E[h'] = h$。
如果不除，训练时的输出会比测试时小。

### 7.3 从零实现

```python
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:           # 丢弃率 100%：全部丢弃
        return torch.zeros_like(X)
    if dropout == 0:           # 丢弃率 0%：全部保留
        return X
    mask = (torch.rand(X.shape) > dropout).float()  # 生成 0/1 掩码
    return mask * X / (1.0 - dropout)               # 丢弃 + 缩放
```

**逐行解释：**
- `torch.rand(X.shape)`：生成 0 到 1 的均匀随机数
- `> dropout`：大于 dropout 的为 True（保留），小于的为 False（丢弃）
- `.float()`：转成浮点数（True→1.0, False→0.0）
- `mask * X`：被丢弃的位置变成 0
- `/ (1.0 - dropout)`：缩放，保持期望值不变

### 7.4 定义带 Dropout 的模型

```python
dropout1, dropout2 = 0.2, 0.5  # 第一层丢弃率 20%，第二层 50%

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training:              # 只在训练时使用 dropout
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out
```

**关键点：** Dropout 只在训练时使用，测试时不使用！

### 7.5 简洁实现

```python
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(dropout1),      # 在第一层后加 Dropout
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(dropout2),      # 在第二层后加 Dropout
    nn.Linear(256, 10)
)
```

`nn.Dropout` 会自动处理训练/测试模式的切换。

---

## 8. 前向传播、反向传播和计算图

### 8.1 前向传播

**定义：** 从输入层到输出层，依次计算每一层的输出。

**单隐藏层网络的前向传播：**
```
输入 x
  → z = W¹x           （线性变换）
  → h = φ(z)          （激活函数）
  → o = W²h           （输出层）
  → L = l(o, y)       （计算损失）
  → J = L + s          （目标函数 = 损失 + 正则项）
```

### 8.2 反向传播

**定义：** 从输出层到输入层，利用链式法则计算每个参数的梯度。

**链式法则：** 如果 Y = f(X)，Z = g(Y)，则：
$$
\frac{\partial Z}{\partial X} = \frac{\partial Z}{\partial Y} \cdot \frac{\partial Y}{\partial X}
$$

**反向传播的过程：**
```
目标函数 J
  → ∂J/∂o        （输出对损失的梯度）
  → ∂J/∂W²       （输出层权重的梯度）
  → ∂J/∂h        （隐藏层输出的梯度）
  → ∂J/∂z        （激活函数前的梯度）
  → ∂J/∂W¹       （隐藏层权重的梯度）
```

### 8.3 训练中的依赖关系

```
前向传播：需要当前参数值 → 计算中间变量和损失
反向传播：需要中间变量 → 计算梯度
参数更新：需要梯度 → 更新参数
```

**为什么训练比预测需要更多内存？**
- 训练时需要保存所有中间变量（用于反向传播）
- 中间变量的大小与网络层数和 batch size 成正比

---

## 9. 数值稳定性和模型初始化

### 9.1 梯度消失和梯度爆炸

**问题：** 深层网络中，梯度需要经过很多层矩阵乘法，可能导致：

| 问题 | 现象 | 后果 |
|------|------|------|
| **梯度消失** | 梯度越来越接近 0 | 靠近输入层的参数几乎不更新，模型学不动 |
| **梯度爆炸** | 梯度越来越大 | 参数更新太猛，模型发散 |

**Sigmoid 导致梯度消失的原因：**
- Sigmoid 的最大导数只有 0.25
- 经过 n 层后，梯度变成 $(0.25)^n$，指数级衰减
- 10 层后梯度只剩 $0.25^{10} ≈ 0.0000009$

**这也是为什么 ReLU 成为默认选择：** 导数要么是 0 要么是 1，不会衰减。

### 9.2 打破对称性

如果所有权重初始化为相同的值，每个神经元学到的东西完全一样，等于只有一个神经元。

**解决方法：随机初始化！**

### 9.3 Xavier 初始化

**核心思想：** 让每一层的输出方差保持稳定，不随层数增加而爆炸或消失。

**公式：**
$$
\sigma = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}
$$

其中 $n_{\text{in}}$ 是输入维度，$n_{\text{out}}$ 是输出维度。

**PyTorch 中使用：**
```python
nn.init.normal_(m.weight, std=std)  # std = sqrt(2 / (n_in + n_out))
```

或使用均匀分布版本：
```python
nn.init.xavier_uniform_(m.weight)  # PyTorch 内置的 Xavier 初始化
```

---

## 10. 环境和分布偏移

### 10.1 什么是分布偏移？

**训练数据和测试数据来自不同的分布**，导致模型在实际应用中表现差。

### 10.2 三种分布偏移类型

| 类型 | 变化的是什么 | 例子 |
|------|-------------|------|
| **协变量偏移** | 输入分布 P(x) 变了，但 P(y\|x) 没变 | 训练用真实照片，测试用卡通图片 |
| **标签偏移** | 标签分布 P(y) 变了，但 P(x\|y) 没变 | 疾病的流行率变了，但症状没变 |
| **概念偏移** | P(y\|x) 本身变了 | "软饮"在不同地区叫法不同 |

### 10.3 现实中的分布偏移案例

1. **医学诊断**：用大学生的血样训练，用到老年人身上 → 完全不适用
2. **自动驾驶**：用游戏渲染数据训练路沿检测器 → 真实道路完全不行
3. **广告模型**：2009 年训练的模型不知道 iPad 是什么
4. **垃圾邮件**：垃圾邮件发送者不断进化，旧模型失效

### 10.4 解决思路

- **协变量偏移**：给训练样本加权，让权重反映它在目标分布中的重要性
- **标签偏移**：通过混淆矩阵估计目标分布的标签比例
- **概念偏移**：用新数据持续更新模型

---

## 11. 实战：Kaggle 房价预测

### 11.1 数据预处理

```python
import pandas as pd
import torch
from torch import nn

# 读取数据
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# 合并特征（去掉 ID 和标签）
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

**数值特征标准化：**
```python
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))  # 标准化：(x - 均值) / 标准差
all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 缺失值填 0
```

**类别特征独热编码：**
```python
all_features = pd.get_dummies(all_features, dummy_na=True)  # 独热编码
```

**转换为张量：**
```python
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
```

### 11.2 定义模型和损失

```python
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))  # 简单线性模型作为基线
    return net
```

**为什么用对数 RMSE？**
$$
\sqrt{\frac{1}{n}\sum_{i=1}^n(\log y_i - \log \hat{y}_i)^2}
$$

房价是"相对"的：偏差 10 万对 12.5 万的房子是大误差，对 400 万的房子是小误差。
取对数后，变成相对误差的度量。

```python
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))  # 预测值限制在 >= 1
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))  # 对数 RMSE
    return rmse.item()
```

### 11.3 K 折交叉验证

```python
def get_k_fold_data(k, i, X, y):
    """获取第 i 折的数据"""
    fold_size = X.shape[0] // k
    idx = slice(i * fold_size, (i + 1) * fold_size)
    X_valid, y_valid = X[idx, :], y[idx]          # 第 i 折作为验证集
    X_train = torch.cat([X[:i*fold_size], X[(i+1)*fold_size:]], 0)  # 其余作为训练集
    y_train = torch.cat([y[:i*fold_size], y[(i+1)*fold_size:]], 0)
    return X_train, y_train, X_valid, y_valid
```

### 11.4 训练和提交

```python
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels,
                          num_epochs, lr, weight_decay, batch_size)
```

**使用 Adam 优化器（对学习率不那么敏感）：**
```python
optimizer = torch.optim.Adam(net.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)
```

---

## 总结：本章核心知识点

| 知识点 | 一句话总结 |
|--------|-----------|
| MLP | 在输入和输出之间加隐藏层 + 激活函数 |
| 激活函数 | ReLU（首选）、Sigmoid（二分类输出）、Tanh（较少用） |
| 欠拟合 | 模型太简单，训练和测试误差都高 |
| 过拟合 | 模型太复杂，训练误差低但测试误差高 |
| 权重衰减 | 在损失函数中加 L2 惩罚项，让权重不要太大 |
| Dropout | 训练时随机丢弃神经元，防止过拟合 |
| 前向传播 | 从输入到输出计算预测值 |
| 反向传播 | 从输出到输入计算梯度 |
| 梯度消失/爆炸 | 深层网络的梯度可能变得太小或太大 |
| Xavier 初始化 | 让每层输出方差保持稳定的初始化方法 |
| 分布偏移 | 训练数据和测试数据分布不同 |
| K 折交叉验证 | 数据少时的模型选择方法 |

---

## 常见面试问题

1. **为什么 MLP 需要激活函数？** —— 没有激活函数，多层线性变换等于单层线性变换。
2. **ReLU 比 Sigmoid 好在哪？** —— 不会梯度消失，计算快。
3. **什么是过拟合？怎么解决？** —— 训练好测试差。解决：权重衰减、Dropout、增加数据、早停。
4. **Dropout 为什么只在训练时用？** —— 测试时需要完整模型的预测能力。
5. **什么是 Xavier 初始化？** —— 让每层输出方差不随层数变化的初始化方法。