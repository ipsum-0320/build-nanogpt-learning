import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # 这样的操作节省了 30% 的参数。
        # 在像 GPT 这样的生成式 Transformer 模型中，模型最前端的 Token Embedding 层（将单词索引转换为向量）和最后端的 Language Model Head（将隐藏层向量映射回单词概率的线性分类层）使用的是同一组权重矩阵。
        # 这种做法被称为 权重绑定（Weight Tying），最早由 Press 和 Wolf 在 2016 年提出。其背后的逻辑主要基于以下几个维度：
        # 1. 语义空间的互通性。Embedding 层的作用是学习词与词之间的语义关系，将 N 维的 One-hot 编码压缩到连续的低维向量空间。**分类头（Linear Head）**的作用是计算当前隐藏状态与词表里每一个词的“相似度”。逻辑： 如果两个词在语义上相似，它们在 Embedding 空间中距离很近；那么在预测时，如果模型认为应该填入其中一个词，另一个词也理应获得较高的逻辑得分（Logits）。使用同一组权重能确保“输入语义”和“输出预测”在同一个坐标系下。
        # 2. 节省海量的参数量在 GPT 模型中，词表大小（Vocabulary Size）通常在 50,000 到 100,000 之间，而隐藏层维度（Hidden Size）可能在 768 到 12,288 之间。如果两者不共享，权重参数量为：$2 \times (Vocab\_Size \times Hidden\_Size)$。对于大型模型，词表权重往往占了非 Transformer 模块参数的很大一部分。共享权重可以直接节省数亿个参数，显著降低显存占用。
        # 3. 正则化与性能提升。研究表明，权重绑定起到了一种正则化的作用。由于输出层（分类头）在训练过程中更新非常频繁（每个 Token 都要计算交叉熵损失），这种更新信号会直接反馈给 Embedding 层。这有助于 Embedding 层更快地收敛，并防止模型在小语料上出现过拟合。

        # init params
        self.apply(self._init_weights)
        # model.apply(fn) = 遍历模型里所有层、所有子模块 **，对每一层都执行一遍 fn 函数**

    def _init_weights(self, module):
        # 不对 nn.LayerNorm 做初始化，因为 nn.LayerNorm $\gamma=1, \beta=0$ 保持输入分布的标准化，不引入额外扰动。
        # nn.LayerNorm 初始状态下，gamma 全是 1，beta 全是 0。 
        # 在模型刚开始训练的第一步，我们希望 LayerNorm 只负责“稳定分布”，而不是去“干扰信号”。所以设为 1 和 0，相当于告诉模型：“先按标准分布传下去，改不改由你后面练出来再说。”
        # $1/\sqrt{d}$ 解决了 $f(x)$ 内部的方差平衡。$1/\sqrt{2N}$ 解决了 $x + f(x)$ 叠加时的方差平衡。
        if isinstance(module, nn.Linear):
            # 根据 Xavier 初始化，初始化的参数权重的标准差分布应该满足 1/√d_model，所谓的 kai_ming 初始化也是类似的原理。
            # 有一个前提是，方差越大，数值越大，这个是数学严格证明的。

            # 那么权重初始化非要用 1/√d，不然训练就崩？我们核心目标只有一个：保证信号经过每一层后，输出的方差既不爆炸也不消失，维持在 1 左右。也就是每一层的输入输出都需要方差维持在 1 左右，那么为什么要维持在 1 左右呢？
            # 在深度网络中，信号要经过 $L$ 层（比如 $L=50$），如果方差不为 1，那么经过多层之后，方差就会连乘，只有 1 能够抵抗指数级增长或衰减。 任何偏离 1 的微小因子，在深度学习的层数放大下，都会变成灾难。因此，我们只能依赖方差为 1。
            # 综上所述，我们所有的初始化技巧（Xavier, Kaiming, 0.02 缩放，数据标准化）和归一化技术（LayerNorm, BatchNorm），全都是为了在每一层维持那个神秘的“方差 1”，y 和 x 的方差为 1。
            # y = wx + b，一般来说，我们需要对输入数据进行标准化，保证 x 的方差为 1，然后通过 “Xavier, Kaiming, 0.02 缩放” 来保证权重 * d_model 之后方差也为 1，最后 wx 的方差也就为 1，然后 b 的方差设置为 0，这样就能保证 y 的输出方差也为 1 了。

            # 好的，上述逻辑其实只保证了一点，那就是只保证了初始化是 y = wx + b 这套逻辑的 x 方差为 1，然后 y 方差也为 1，那如何保证训练过程中 y 的方差也为 1？
            # 1. 首先是学习率，会比较小，如果学习率太大，如果学习率太大，权重 $W$ 会剧烈波动，方差瞬间爆炸。
            # 2. LayerNorm。这是保证训练中方差稳定的定海神针。不管你的 $W$ 和 $b$ 在训练中变成了什么样，在每一层线性计算结束后，信号都会经过 LayerNorm：强制归零：它计算当前这一批数据的均值 $\mu$，然后减去它。强制归一：它计算当前数据的方差 $\sigma^2$，然后除以 $\sigma$。
            
            # 其中，LayerNorm 是更加重要的，所以这就是为什么 LayerNorm 需要再每一层后面都添加，就是为了让方差保持在 1。
            # 那么 LayerNorm 中的 gamma 和 beta 两个可学习参数的作用是什么呢？一言以蔽之，为了重新让均值和方差不变为 0 和 1。这两个参数有如下几个作用：
            # 1. “数值稳定性”与“表达能力”的博弈归一化（操作部分）：是为了数值稳定性。它像是一个“安全带”，保证数值不会飞到外太空（爆炸）或者缩到看不见（消失）。$\gamma$ 和 $\beta$（参数部分）：是为了表达能力。它像是一个“调节旋钮”。如果模型发现，某一层的特征必须要在数值很大时（比如方差为 10）才能触发后续的逻辑，那么模型可以自己通过训练把 $\gamma$ 调大。关键点：这赋予了模型一种**“自愈”**能力。模型可以自主决定：“这一层我需要标准分布”，或者“那一层我需要非标准分布”。
            # 2. 保护非线性（不让激活函数“变傻”）这是最精妙的一点。很多激活函数在 $0$ 附近是线性的：比如 Tanh 和 Sigmoid：在 $0$ 附近几乎就是一条斜线。如果每一层都严格死守 $\mu=0, \sigma=1$，那么信号永远落在激活函数的线性区。后果：多层神经网络就会退化成一个巨大的线性层（线性层的堆叠还是线性层），失去了处理复杂逻辑的能力。
            # 这也是为什么 LayerNorm 要一开始初始化为全 1 值和全 0 值。这意味着，在训练刚开始的第一秒，LayerNorm 确实是严格地把方差和均值限制在 $1$ 和 $0$。随着训练的进行，模型如果觉得“不爽”，它会通过反向传播自己去修改这两个值。
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 除了残差，别的地方都是 y = wx + b，因此之前的优化可以保证方差为 1，后来多了残差，就需要额外的优化手段了，也就是下面的这行代码。
                std *= (2 * self.config.n_layer) ** -0.5
                # 它是为了解决**“残差叠加导致的方差爆炸”**问题。
                # 之前我们讨论过，初始化 $1/\sqrt{d}$ 是为了保证单个线性层（y = wx + b）的输入输出方差平衡。但在 Transformer Block 中，结构是这样的： y = x + wx + b。
                # 这里的加法打破了平衡。根据方差的性质（假设 $x$ 和 $\text{Layer}(x)$ 独立）：$$\text{Var}(x + \text{Layer}(x)) = \text{Var}(x) + \text{Var}(\text{Layer}(x))$$如果 $x$ 的方差是 1，而 Layer 层的输出方差也是 1。那么经过 1 层后，$x_{next}$ 的方差就变成了 2。经过 $N$ 层后，方差就会线性增长到 $N$。对于一个 12 层的模型，最后一层的方差会是第一层的 12 倍；对于 GPT-3 这种 96 层的模型，方差会飙升到 96 倍。这会导致深层网络的数值极其不稳定。
                # 上述代码，std *= (2 * n_layer) ** -0.5 也就是把标准差除以了 $\sqrt{2N}$。（注：这里的 $2$ 是因为每个 Transformer Block 包含两个子层：Self-Attention 和 MLP，所以总层数是 $2 \times n\_layer$）。

                # 注意，我们这里缩减的是非残差部分的输出方差，将其缩短了 1/根号2N，只是为了保证残差的主干优先，让信号可以像在高速公路上一样，无损地从第 1 层冲到第 100 层。
                # 因此我们不强行让每一层输出 $y$ 的方差都为 1（否则每一层 LN 都在做“除法”。经过 $N$ 层后，最初的输入信号 $x_0$ 贡献的权重会以 $\frac{1}{\sqrt{2}} \times \frac{1}{\sqrt{2}} \dots$ 的速度指数级衰减），而是选择使用 $1/\sqrt{2N}$ 缩放，$x_1$ 几乎和 $x$ 一模一样！原始信号被完美保留了。经过 $2N$ 层后：总方差变成了 $1 + \frac{1}{2N} \times 2N = 2$。
                
                # 一般来说，我们只会对出口地方进行降权，因为它是支流（非残差部分）汇入主干前的最后一关。就是在这里，支流的信号要和主干相加。为了不让这次相加导致主干方差变成 $1+1=2$，我们必须在 c_proj 这里把方差压下去。
                # 在一个标准的 Transformer Block 中，通常只有两个层需要缩放，因为只有它们直接和主干相加：
                # Attention 模块里的 c_proj：
                # - 它负责把 Attention 的结果送回主干。
                # MLP 模块里的 c_proj (通常是第二个 Linear 层)：
                # - MLP 通常由两层组成：Linear1 -> ReLU/GELU -> Linear2。
                # - 其中 Linear2 的结果会直接加回主干：x = x + MLP_output。因此，Linear2 也需要加上这个标记。

                # 我们不能给残差 x 降级，否则就无法保护“恒等映射” (Identity Mapping)，这是残差的灵魂，主干 $x$ 是“保命线”，必须无损传递。我们要让信息能从第一层“瞬移”到最后一层。
                # 之所以我们通过 $\sqrt{2N}$ 把支流压得很低，本质上是告诉模型：“你刚出生的时候，先别急着大刀阔斧地修改前人的成果。你先作为一个微小的修正项存在，等训练开始了，你觉得确实有必要，再通过权重更新把自己调大。”
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                # 避免在训练开始时引入人为的偏置（Bias），让模型从一张“白纸”开始学习特征。
                # 如果说权重 (Weight) 决定了输入信号的“重要程度”，那么偏置 (Bias) 就决定了要触发神经元到底有多“难”。
                # 这里会保证 b 的方差和均值都为 0。
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        # torch.range 有点类似于 python 的 range，它直接创建一个 PyTorch 张量（Tensor），一维的 [0, T)。
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        # 这里会进行向量计算，或者说矩阵计算，将 pos 进行向量化。
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # 这里也进行矩阵计算，将 tok 向量化。
        # nn.Embedding 是专门为了优化与 one-hot 编码的计算而出现的，因此其物理实现是查表操作——output = wte.weight[idx]。

        x = tok_emb + pos_emb
        # 这里会进行 tensor 的广播。
        # 每一个 Batch 里的每一行，都加上了相同的一套位置向量。

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
            # 做特征提取，其实就是过了很多层的 Transformer Encoder。
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        # 做了归一化。
        # 现在来解释一下为什么需要做归一化：
        # 在深度学习中，特征最终要通过激活函数（如 GeLU, ReLU, Softmax）。如果输入的数值（Logits）非常大（比如 100）或非常小（比如 -100），Softmax 的梯度会变得极其微小（接近 0）。均值 0，方差 1 的意义：这保证了大部分数值都落在 $[-2, 2]$ 这个范围内。在这个区间内，Softmax 和其他激活函数的梯度最灵敏，模型能学得最快。直观理解：如果数值失控，模型就像进入了“死胡同”，无论怎么调整参数，输出几乎没变化，模型就“学不动”了。
        # 此外，归一化后，等 Loss 线比较圆，不圆（椭圆）：梯度震荡，学习率得设得很小，训练慢；圆（圆形）：梯度顺畅，学习率可以设大，训练快且稳。
        # 因此加了归一化后，解决残差累加导致的数值爆炸，配合 lm_head（线性层）进行高效分类，确保训练的稳定性（尤其是针对 Pre-Norm 架构）。

        # Pre-Norm 还是 Post-Norm
        # A. Post-Norm（原始 Transformer 做法）结构： x = LayerNorm(x + SubLayer(x))逻辑： 先计算，再加残差，最后归一化。缺点： 归一化层在输出端，梯度回传时会受到较大干扰，导致深层模型很难训练（容易崩）。
        # B. Pre-Norm（现在 GPT 的主流做法，即你这段代码所属的架构）结构： x = x + SubLayer(LayerNorm(x))逻辑： 在每一层计算之前先归一化。优点： 梯度可以从残差连接（那个 x + ...）中无损地流向很深的层。这让训练几十层甚至上百层的模型变得非常稳定。为什么还需要 ln_f？ 因为 Pre-Norm 架构中，主干路径上的 $x$ 一直在累加，没有被整体归一化过，所以最后必须补一刀 ln_f，把数值拉回“圆润”的分布。

        # 是不是每一层后面都要加归一化层？
        # 是的，在 Transformer 架构中，归一化就像是每一层的“标配呼吸机”。 * Block 内部：负责维持每一层特征提取的稳定。最后的 ln_f：负责给所有层的累积结果做一个最终的“质检”和“格式化”，确保交给分类器的数据是完美分布的。
        logits = self.lm_head(x) # (B, T, n_embd) * (n_embd, vocab_size) = (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # PyTorch 的 F.cross_entropy 期望输入是 二维 的（样本数, 类别数）。
            # logits 变成：(B*T, vocab_size)，targets 变成：(B*T)。
            # 这样每一行就是一个预测结果，对应一个正确标签。
            # 基本上可以理解为如下：
            #      logits       label
            # softmax(@@@@@@@@)   @
            # softmax(@@@@@@@@)   @
            # softmax(@@@@@@@@)   @
            # 在随机初始化权重参数的情况下，loss 应该就等于 1/V。
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    @classmethod
    # 这里的 @classmethod 装饰器用于将普通方法转换为类方法，核心作用的是绑定到类本身（而非类的实例），无需创建类的对象即可调用。
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        # 创建 config 配置。
        model = GPT(config)
        sd = model.state_dict()
        # 获取自定义模型的参数字典
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        # 在 PyTorch 的 nn.Module 中，这种不需要梯度更新、但需要随模型一起保存的张量被称为 Buffer（缓存）。
        # 真正的偏置：比如 attn.c_attn.bias，它是模型学习出来的，用来调整线性变换的结果。
        # 这里的 .attn.bias：它其实是 tril（下三角矩阵）。在 Hugging Face 的实现里，它被注册为了一个 buffer，名字恰好叫 bias。

        # 在 Pytorch 的 nn.Module 中，如果想存储一些数据，通常有两种方式：
        # - Parameters，这就是我们常用的 nn.Parameter（或者 nn.Linear 里的 weight）。特点：它们需要计算梯度，会被优化器更新，且会自动包含在 state_dict 中。
        # Buffers，这就是通过 register_buffer 注册的张量。特点：它们不需要计算梯度，不会被优化器更新，但它们依然会自动包含在 state_dict 中，并随模型一起保存和加载。一般来说，mask、绝对位置编码都会放在 Buffers 中，我们写的 y = F.scaled_dot_product_attention(...) 是目前最高效的 GPT 实现方式。它通过算法逻辑取代了物理存储。
        # 当你通过 self.register_buffer('name', tensor) 注册一个张量后，它获得了三个普通变量没有的“特权”：
        # - 跟随设备转移：当你执行 model.to('cuda') 时，所有 Buffer 会自动从 CPU 搬运到 GPU。
        # - 自动序列化：当你执行 torch.save(model.state_dict(), 'path.pth') 时，Buffer 会被包含在字典里存入硬盘。
        # - 不参与梯度更新：它永远不会出现在 optimizer.step() 的更新列表中，也不会计算导数。
        # Buffers 对比普通属性：路人甲，不随行，不保存，self.my_attr = torch.zeros(3, 3)
        # 使用 Buffers 可以是 self.mask = nn.Buffer(torch.tril(torch.ones(1024, 1024)))，也可用 self.register_buffer('my_buffer', tensor)

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # 获取官方模型的参数字典

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                # 通过 shape[::-1]，你实际上是在检查：“虽然这两个矩阵现在的形状不一样，但如果我把官方的转置（Transpose）一下，它们是不是就一样了？”
                # [::-1]: 这是 Python 列表/元组的切片操作，表示倒序（反转），如果原形状是 (768, 2304)，反转后就变成了 (2304, 768)。
                with torch.no_grad():
                    # with torch.no_grad(): 是一个上下文管理器（Context Manager），它的核心作用是：暂时关闭梯度计算系统（Autograd 引擎）。通俗点说，它告诉 PyTorch：“接下来的操作我只是在搬运数据或者做推断，不需要你记录任何求导信息，也不需要为反向传播做准备。”
                    # 默认情况下，当你对 PyTorch 的 Tensor（且 requires_grad=True）进行操作时，PyTorch 会在后台构建一个“计算图”来记录所有操作，以便后续计算导数。加载权重只是简单的赋值操作，完全不需要计算梯度。
                    # 它能带来什么好处？内存优化：不记录计算图意味着不占用额外的显存/内存来存储中间激活值。速度提升：省去了后台追踪计算逻辑的时间，运行效率更高。
                    # 一般来说，我们在模型推理或者手动加载权重时都需要关闭梯度系统。
                    # with torch.no_grad(): 的有效范围仅限于其下方所有缩进一级的代码行。一旦代码恢复到原来的缩进级别，梯度追踪就会自动恢复。如果你想让整个函数都不追踪梯度，不需要缩进整个函数体，可以使用装饰器：@torch.no_grad()。
                    # 下面这些参数都会来追踪梯度：
                    # 模型前向传播、损失计算、权重加载/拷贝，索引切片、链接以及维度变化。
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

class DataLoaderLite:
    # 这里的 DataLoader 是自定义的，同时具备了 DataLoader 和 DataSet 的功能。
    # 在标准的 PyTorch 工作流中，数据加载通常被拆分为两个角色：Dataset：负责“有什么数据”以及“如何取某一条数据”（定义索引索引 $i$ 对应的内容）。DataLoader：负责“怎么取数据”（定义 Batch Size、是否打乱、多线程加载、分布式分发等）。你提供的这段 DataLoaderLite 确实将这两者合二为一了。
    # 为什么它是 Dataset？
    # 持有数据源：它直接管理 self.shards（磁盘上的文件列表）和 self.tokens（当前加载到内存的张量）。定义切片逻辑：它通过 buf[current_position : ...] 直接从原始序列中切出训练样本。
    # 为什么它是 DataLoader？
    # 状态管理：它自己维护指针 self.current_position，不需要外部循环提供索引。批处理（Batching）：它内部直接执行了 .view(B, T)，直接输出形状为 $(B, T)$ 的张量，而不是单条数据。分布式分发（Distributed Sharding）：它通过 process_rank 和 num_processes 实现了多机多卡的数据并行逻辑，这原本是 DistributedSampler 做的事情。

    # 在 PyTorch 的常规开发中，使用 Dataset 和 DataLoader 这套标准 API 是绝对的主流。
    # Dataset (食材库)它定义了数据的来源和读取方式。主要任务：告诉程序数据集有多大（__len__），以及给定一个序号 $i$ 后，如何把第 $i$ 个样本读出来（__getitem__）。类型：Map-style: 像字典一样，按索引取样（最常用）。Iterable-style: 像流一样，只能一直往后读（适合海量数据）。
    # - 我们需要实现的方法： 如果我们自定义一个类继承自 Dataset，通常必须重写两个方法：
    #   1. __len__：返回数据集的总样本数。
    #   2. __getitem__：给定一个索引 i，返回对应的样本及其标签。

    # DataLoader (厨师/传送带)它定义了数据如何组织成 Batch 并送到模型手里。主要任务：打乱顺序（Shuffle）、并行读取（Num Workers）、把 $B$ 个样本打包成 Tensor（Batching）、把数据搬运到 GPU，DataLoader 的功能如下：
    #   - Batch Size（批量大小）： 将数据分块，比如一次给模型 32 个样本。
    #   - Shuffle（打乱）： 每一轮训练（Epoch）开始前，随机打乱数据顺序，防止模型过拟合。
    #   - Multiprocessing（多线程）： 使用 num_workers 开启多个进程并行加载数据，防止 CPU 加载太慢让 GPU “空转”。
    #   - Drop Last： 如果总数据量不能被 Batch Size 整除，决定是否舍弃最后不完整的一块。
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    

# ⭐️ 对小批数据进行过拟合是确保训练 Pipeline 正确的关键，在业内，我们把它叫做 "Overfit a single batch"。如果你能让模型在这一小撮数据（比如只有 1~2 个样本）上把 Loss 降到接近 0，或者准确率达到 100%，就证明你的整个 Pipeline（流水线）是通的。
# ⭐️ “先过拟合一小批，再征服星辰大海。” 只有确定模型有“死记硬背”的能力，我们才能去谈论它的“泛化”能力。

max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# torch.Tensor（张量）：必须重新赋值，需要是 tensor = tensor.to(device)。
# nn.Module（模型/层）：不需要重新赋值，需要 model.to(device)。
# 当你写 optimizer = torch.optim.Adam(model.parameters()) 时，一定要先把 model.to('cuda')，再把模型参数传给优化器。如果先建优化器再移模型，优化器里记下的参数地址可能还是 CPU 上的，会导致报错。
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

for step in range(max_steps):
    model.train()
    # 这行代码的作用是将模型设置为训练模式。
    # 有些模型层（Layer）在“训练”和“推理（预测）”时的行为是完全不同的。最典型的例子有两个：
    # Dropout 层：在训练时会随机“关掉”一部分神经元来防止过拟合；但在预测时，必须全部打开。
    # Batch Normalization（批归一化）层：在训练时会计算当前 Batch 的均值和方差；但在预测时，会使用整个训练集累积下来的均值和方差。
    optimizer.zero_grad()
    # 这行代码的作用是把模型参数中缓存的梯度全部归零。
    # 为什么要归零？在 PyTorch 中，梯度的计算有一个特殊的设定：梯度是累加的（Accumulative）。
    # 当你调用 loss.backward() 时，新计算出的梯度会加到原有的梯度上，而不是替换掉它们。后果：如果你不手动归一化，第二步的梯度就会加上第一步的梯度，导致梯度数值越来越大，模型参数更新方向彻底乱套。
    # 但有时候也会去做梯度累计，下面是小 batch 梯度和大 batch 梯度的作用：
    # 小 Batch 的路径——“醉汉走路”：虽然它会左右晃荡（震荡），但这种震荡有时是好事。它能帮助模型跳出那些微小的“局部最小值”（Local Minima）或“鞍点”，因此泛化能力好。但因为样本少，这 2 个样本可能刚好是“特例”（比如都是生僻词或长句子）。计算出的梯度方向可能会偏移大部队的方向。
    # 大 Batch 的路径——“高铁行驶”：路径非常直，非常稳。但由于它太稳了，一旦进入一个坑（局部最优解），它可能就直接停在那了，没有足够的随机性能让它跳出来。但样本多（比如 128 个），通过平均，那些极端的特例被抵消了。计算出的梯度更接近整个数据集的真实梯度方向。
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        # loss 去做反向传播，计算梯度。
        optimizer.step()
        # 优化器根据梯度更新模型参数。
