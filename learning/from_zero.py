import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

class CausalSelfAttention(nn.Module):
    # 这里完整的实现了因果注意力机制。

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 这里实现了 embd 和 head 的计算，需要保证 embed 能够被 head 整除。
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 注意力层就是一个线性层，只不过这里的实现更加高效，他直接使用了一个 Linear 就搞定了 QKV 三个权重矩阵。
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 这里的 c_proj 是用来混合多个头的计算结果的。
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # 输出分解，B (Batch Size): 一次处理多少个句子。T (Time/Sequence Length): 每个句子的长度（有多少个单词/Token）。C (Channels/Embedding Dim): 每个单词用多少维度的向量表示（例如 GPT-2 是 768 维）。
        qkv = self.c_attn(x) # qkv 的维度是 (B，T，3 * C)
        # 矩阵乘法只发生在“相互对接”的那两个维度上，而其他的维度（如 Batch 和 Time）只是像“乘客”一样坐在旁边，不参与运算。
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 这里利用一个全连接层一次性计算出 Query (查询), Key (键), 和 Value (值)。
        # 它在第 3 个维度（索引为 2 的维度）进行切分。切分的步长（大小）是 self.n_embd（即 768）。
        # 因此 q、k、v 分别是 (B，T，C)

        # 在 Pytorch 中，在处理多维张量（Tensor）时，PyTorch 的 torch.matmul()（或者 @ 运算符）遵循一个核心原则：前面的维度被视为“批次（Batch）”，只有最后两个维度才参与真正的矩阵乘法。
        # 假设你有两个张量 $A$ 和 $B$：$A$ 的形状： $(..., N, M)$$B$ 的形状： $(..., M, P)$结果的形状： $(..., N, P)$这里的 ... 可以是任意维度的组合（比如 [Batch, Head]）。PyTorch 会保持这些前面的维度不变，只对最后的二维矩阵进行乘法：$(N \times M) \times (M \times P) = (N \times P)$。

        # 因此，我们这里需要使用 transpose 来改变维度的位置，我们希望参与计算的是 token 及其对应的维度表示，而不是每个 token 的头及其维度表示，前者的计算表示 token 之间的信息流动，而后者的计算表示 token 内部向量表征的信息流动。

        # transpose 前后，数据的顺序不会发生变换吗？
        # 数据的“逻辑顺序”发生了变换，但数据的“物理存储”在内存中暂时没有变。但是这种逻辑顺序的改变正是我们要的效果！我们主动改变了数据之间“谁和谁发生关系”的规则。
        # 另外，即使我们使用了 transpose(1,2) 交换了向量位置，但是对于当前某个特定的“头”来说，它的 $Q$ 矩阵逻辑上依然是 (T, hs)。计算逻辑依然是：
        # 第一行（词1的 Query）去乘以 $K^T$ 的第一列（词1的 Key）。
        # 第一行（词1的 Query）去乘以 $K^T$ 的第二列（词2的 Key）。
        # 结果生成的 (T, T) 矩阵中，第一个位置 (0,0) 永远是“词1对词1”的分数，(0,1) 永远是“词1对词2”的分数。因此，只要你使用的是 transpose，PyTorch 内部的索引机制就会保证：虽然维度换了位置，但“词1”永远是“词1”，“头1”永远是“头1”。它们只是在不同的维度坐标系下排队，彼此的身份标签（Identity）是锁死的。

        # 最后，一个最重要的结论是，变成这个样子之后 (B, nh, T, hs)，参与计算的其实只有 T 和 hs，比如我们只会让第一个 batch 中的第一个头中的 Token 参与计算， 而不会出现跨batch和跨头的现象。

        # 当你把张量变成 (B, nh, T, hs) 时，你实际上是给数据加了两层“次元壁”：
        # 第一层（Batch 维度）：确保“张三”句子的单词，绝不会和“李四”句子的单词产生联系。
        # 第二层（Head 维度）：确保“负责语法的头”看到的特征，绝不会和“负责语义的头”混在一起。

        # 为什么这能保证顺序？
        # 因为在 T 维度（时间步/单词顺序）参与计算之前，其他的维度（B 和 nh）已经被提前锁定在了批次索引里。在计算注意力分数 $Q \times K^T$ 时，计算只发生在特定的 (T, hs) 内部。这意味着：第 $i$ 个位置的 Query 永远只能在同一个 Batch、同一个 Head 里面，去寻找第 $j$ 个位置的 Key。   

        # 上述矩阵运算的结论是普适的：

        # 法则一：最后两维是“演播室”，前面都是“观众席”
        # 在 PyTorch 中，几乎所有涉及矩阵相乘的算子（如 @, torch.matmul, torch.bmm）都遵循这个铁律：
        # 参与计算的维度：永远是张量的最后两个维度。
        # 携带维度（Batch Dimensions）：倒数第三位及以前的所有维度，都被视为“并行的副本”。
        # 举例来说，如果你有两个形状为 (A, B, C, D, N, M) 和 (A, B, C, D, M, P) 的超级巨型张量相乘：
        # * PyTorch 会自动在后台启动 $A \times B \times C \times D$ 个完全独立的矩阵乘法。每一个乘法都是 $(N \times M) \times (M \times P)$。
        # 它们之间绝对不会发生任何跨维度的“串门”或数据交换。

        # 法则二：顺序的“保真性” (Identity Preservation)
        # 这是你刚才提到的核心点。
        # 只要你在计算前，通过 view 和 transpose 将你想要保护的属性（比如“批次”、“头”、“样本编号”）挪到了倒数第三位或更前，那么在整个矩阵运算过程中，这些属性对应的顺序和位置是绝对安全的。
        # 这种普适性保证了：你可以放心地把 1024 张图片、128 个句子、或者 12 个注意力头塞进一个张量里丢给 GPU。GPU 只管算最后的矩阵，绝不会把第一张句子的单词和第二张句子的单词算到一起。

        # 法则三：广播机制的自动补全
        # 如果两个张量的维度不一致，PyTorch 会尝试“克隆”前面的维度来匹配：
        # 在注意力计算中，有一个非常经典的广播应用：Causal Mask（因果掩码）。
        # 注意力分数评分：[B, nh, T, T]（每对单词的互动分）。
        # 掩码 (Mask)：通常我们只生成一个 [1, 1, T, T] 的矩阵。
        # 当你把它们相加时：
        # 这个单一的掩码会被沿着 B（Batch）克隆。
        # 会被沿着 nh（Head）克隆。
        # 最终效果：你只定义了一个掩码，但它自动保护了所有 Batch、所有 Head 的数据，确保它们都不能“偷看未来”。

        # 广播机制生效的规则有哪些：
        # 并不是任意两个张量都能广播，必须满足以下条件：
        # 右对齐：从最后一个维度（最右边）开始往左比对。
        # 维度相等 OR 其中一个是 1：
        # - 如果某个维度一个是 12，另一个是 1，可以广播（1 会变成 12）。
        # - 如果一个是 12，另一个是 5，报错！（无法克隆出这种对应关系）。
        # 维度缺失：如果一个张量比另一个维度少，它会被自动在左边补 1。
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # 必须是 view 和 transpose 配合起来，不能直接使用 view 实现 (B, nh, T, hs) 的效果，否则就行就像把不同人的手、脚、躯干随机拼凑成一个新的人，语义完全崩塌。

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        # 它把我们之前准备好的 $Q, K, V$ 拿过来，算出单词两两之间的相关性（Attention Score），并根据这个相关性对 $V$（信息内容）进行加权求和。
        # is_causal=True（关键点），这是为了 “防止偷看答案”。在训练 GPT 这种生成模型时，第 5 个单词只能看到前 4 个单词的信息，不能看到第 6 个。设置这个参数后，函数会自动把未来的信息屏蔽掉。
        # Mask 矩阵是对一句话的 Mask，如下所示：
        # [[1, 0, 0, 0, 0],
        #  [1, 1, 0, 0, 0],
        #  [1, 1, 1, 0, 0],
        #  [1, 1, 1, 1, 0],
        #  [1, 1, 1, 1, 1]]
        # 因为一句话有 $T$ 个词，所以词与词之间的关系是一个 $T \times T$ 的方阵。行（Row）：代表“当前正在处理哪个词”。列（Column）：代表“它在看哪个词”。对于因果注意力（Causal Attention）：当你在处理第 3 个词（第 3 行）时，Mask 会允许你看第 1, 2, 3 列，但挡住第 4, 5 列。这个逻辑对这句话里的每个词都适用。
        # 这种 mask 机制保证了即使是对于一句话，Transformer 也能够实现并行训练：
        # 在训练时，我们并不需要像生成文本时那样一个词一个词地蹦。输入：整句话 $x = [w_1, w_2, w_3, w_4, w_5]$。目标（Label）：整句话向后平移一个位置 $y = [w_2, w_3, w_4, w_5, w_6]$。
        # 由于矩阵运算 @ 和 F.scaled_dot_product_attention 是对整个 $T \times T$ 矩阵同时操作的，GPU 会同时算出：看到 $w_1$ 后预测 $w_2$。看到 $[w_1, w_2]$ 后预测 $w_3$。看到 $[w_1, w_2, w_3, w_4, w_5]$ 后预测 $w_6$。这些预测是同时发生的！ 
        # GPU 并不需要等第一组算完再去算第二组。

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # 计算完成后，y 的形状是 (B, nh, T, hs)（Batch, 头数, 时间步, 每个头的大小）。但模型接下来的层需要的是原始的 (B, T, C) 形状，所以我们要把这层皮“剥回来”。
        # output projection
        y = self.c_proj(y)
        # c_proj（一个全连接层）会让这 12 个头的信息进行深度交换和融合。它能学到哪些头的信息更重要，并把这些信息重新压缩或映射回模型的核心特征空间。
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # 这里是对 GeLU 的近似。
        # 在这里介绍一下激活函数的本质。
        # 激活函数的第一个作用是引入非线性，根据通用近似定理（Universal Approximation Theorem），只要拥有足够的神经元和至少一个非线性激活函数，神经网络就可以模拟任何复杂的函数。
        # 激活函数会影响梯度的大小，而且这种影响至关重要。 激活函数决定了信号在反向传播时如何“衰减”或“增强”。像 Sigmoid 或 Tanh 这样的函数，在输入值非常大或非常小时，导数会趋近于 0（进入饱和区）。如果激活函数的导数在某些区间非常大，且权重初始值也较大，连乘效应会导致梯度变得无穷大，模型训练会直接崩溃。
        # 现代解决方案是 ReLU ($f(x) = \max(0, x)$)，其之所以成为主流，正是因为它在 $x > 0$ 区域的导数恒为 1。它不压缩梯度，有效缓解了深层网络中的梯度消失问题，让训练更深的网络成为可能。
        # 目前，在深度学习（尤其是计算机视觉和普通的深层感知机）领域，ReLU 及其变体确实占据了统治地位。
        # ReLU 其实引入了非线性，现代深度学习之所以强大，其实就是靠**“分段线性逼近（Piecewise Linear Approximation）”**。ReLU 的聪明之处在于：它保留了线性函数计算简单、梯度不消失（在正区间）的优点。它又通过那一个**“拐角”**，引入了足以模拟宇宙万物的非线性表达能力。
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        # “夹心饼干”式结构（线性层 + 激活函数 + 线性层）是神经网络中最经典、最有效的模块设计方式。
        return x

class Block(nn.Module):
    # nn.Parameter：它是最底层的。它本质上就是一个特殊的 torch.Tensor，只要把它赋值给 nn.Module 的属性，PyTorch 就会自动将其视为“需要优化的参数”。其他三个：nn.Linear、nn.Embedding 和 nn.LayerNorm 内部都包含了一个或多个 nn.Parameter。Linear 内部有 .weight 和 .bias 参数。LayerNorm 内部有 .weight (也叫 $\gamma$) 和 .bias (也叫 $\beta$) 参数。

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) 
        self.attn = CausalSelfAttention(config) # 做 token 之间的信息交换。
        self.ln_2 = nn.LayerNorm(config.n_embd) 
        self.mlp = MLP(config) # 做升维和降维的映射。

    # 在 PyTorch 中，你不应该直接显式调用 model.forward(x)。
    # 相反，你应该像调用函数一样直接调用对象：model(x)。
    # 为什么？ 当你执行 model(x) 时，PyTorch 会在后台调用一个叫 __call__ 的内置方法。
    # __call__ 做了什么？ 它不仅会运行你写的 forward 逻辑，还会负责注册 Hooks（钩子）。这些钩子对于调试、可视化（比如 TensorBoard）至关重要。
    # 结论： 永远通过 out = model(x) 调用，而不是 out = model.forward(x)。
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
# 当你打上这个装饰器时，Python 会自动为你生成以下方法：
# __init__：自动处理参数传递。你可以直接 config = GPTConfig(n_layer=24)。
# __repr__：自动生成美观的打印格式。直接 print(config) 就能看到完整的属性名和值。
# __eq__：自动支持比较。如果有两个配置类属性完全一样，config1 == config2 会返回 True。
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
# 在 PyTorch 中，nn.Module 是所有神经网络模块的基类。如果你想构建一个模型（不管是像 GPT 这样的大模型，还是一个简单的线性层），继承 nn.Module 是必须的“入场券”。如果不继承它，你写的只是一个普通的 Python 类；继承了它，你的类就变成了拥有“超能力”的 神经网络组件。

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 模型结构的定义，在这里要学会几个 API 的使用。
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)