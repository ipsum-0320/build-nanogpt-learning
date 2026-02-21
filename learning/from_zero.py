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
        # 假设我们有两个张量 $A$ 和 $B$：$A$ 的形状： $(..., N, M)$$B$ 的形状： $(..., M, P)$结果的形状： $(..., N, P)$这里的 ... 可以是任意维度的组合（比如 [Batch, Head]）。PyTorch 会保持这些前面的维度不变，只对最后的二维矩阵进行乘法：$(N \times M) \times (M \times P) = (N \times P)$。

        # 因此，我们这里需要使用 transpose 来改变维度的位置，我们希望参与计算的是 token 及其对应的维度表示，而不是每个 token 的头及其维度表示，前者的计算表示 token 之间的信息流动，而后者的计算表示 token 内部向量表征的信息流动。

        # transpose 前后，数据的顺序不会发生变换吗？
        # 数据的“逻辑顺序”发生了变换，但数据的“物理存储”在内存中暂时没有变。但是这种逻辑顺序的改变正是我们要的效果！我们主动改变了数据之间“谁和谁发生关系”的规则。
        # 另外，即使我们使用了 transpose(1,2) 交换了向量位置，但是对于当前某个特定的“头”来说，它的 $Q$ 矩阵逻辑上依然是 (T, hs)。计算逻辑依然是：
        # 第一行（词1的 Query）去乘以 $K^T$ 的第一列（词1的 Key）。
        # 第一行（词1的 Query）去乘以 $K^T$ 的第二列（词2的 Key）。
        # 结果生成的 (T, T) 矩阵中，第一个位置 (0,0) 永远是“词1对词1”的分数，(0,1) 永远是“词1对词2”的分数。因此，只要我们使用的是 transpose，PyTorch 内部的索引机制就会保证：虽然维度换了位置，但“词1”永远是“词1”，“头1”永远是“头1”。它们只是在不同的维度坐标系下排队，彼此的身份标签（Identity）是锁死的。

        # 最后，一个最重要的结论是，变成这个样子之后 (B, nh, T, hs)，参与计算的其实只有 T 和 hs，比如我们只会让第一个 batch 中的第一个头中的 Token 参与计算， 而不会出现跨batch和跨头的现象。

        # 当我们把张量变成 (B, nh, T, hs) 时，我们实际上是给数据加了两层“次元壁”：
        # 第一层（Batch 维度）：确保“张三”句子的单词，绝不会和“李四”句子的单词产生联系。
        # 第二层（Head 维度）：确保“负责语法的头”看到的特征，绝不会和“负责语义的头”混在一起。

        # 为什么这能保证顺序？
        # 因为在 T 维度（时间步/单词顺序）参与计算之前，其他的维度（B 和 nh）已经被提前锁定在了批次索引里。在计算注意力分数 $Q \times K^T$ 时，计算只发生在特定的 (T, hs) 内部。这意味着：第 $i$ 个位置的 Query 永远只能在同一个 Batch、同一个 Head 里面，去寻找第 $j$ 个位置的 Key。   

        # 上述矩阵运算的结论是普适的：

        # 法则一：最后两维是“演播室”，前面都是“观众席”
        # 在 PyTorch 中，几乎所有涉及矩阵相乘的算子（如 @, torch.matmul, torch.bmm）都遵循这个铁律：
        # 参与计算的维度：永远是张量的最后两个维度。
        # 携带维度（Batch Dimensions）：倒数第三位及以前的所有维度，都被视为“并行的副本”。
        # 举例来说，如果我们有两个形状为 (A, B, C, D, N, M) 和 (A, B, C, D, M, P) 的超级巨型张量相乘：
        # * PyTorch 会自动在后台启动 $A \times B \times C \times D$ 个完全独立的矩阵乘法。每一个乘法都是 $(N \times M) \times (M \times P)$。
        # 它们之间绝对不会发生任何跨维度的“串门”或数据交换。

        # 法则二：顺序的“保真性” (Identity Preservation)
        # 这是我们刚才提到的核心点。
        # 只要我们在计算前，通过 view 和 transpose 将我们想要保护的属性（比如“批次”、“头”、“样本编号”）挪到了倒数第三位或更前，那么在整个矩阵运算过程中，这些属性对应的顺序和位置是绝对安全的。
        # 这种普适性保证了：我们可以放心地把 1024 张图片、128 个句子、或者 12 个注意力头塞进一个张量里丢给 GPU。GPU 只管算最后的矩阵，绝不会把第一张句子的单词和第二张句子的单词算到一起。

        # 法则三：广播机制的自动补全
        # 如果两个张量的维度不一致，PyTorch 会尝试“克隆”前面的维度来匹配：
        # 在注意力计算中，有一个非常经典的广播应用：Causal Mask（因果掩码）。
        # 注意力分数评分：[B, nh, T, T]（每对单词的互动分）。
        # 掩码 (Mask)：通常我们只生成一个 [1, 1, T, T] 的矩阵。
        # 当我们把它们相加时：
        # 这个单一的掩码会被沿着 B（Batch）克隆。
        # 会被沿着 nh（Head）克隆。
        # 最终效果：我们只定义了一个掩码，但它自动保护了所有 Batch、所有 Head 的数据，确保它们都不能“偷看未来”。

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
        
        # non-flash attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # 1. 内存效率：从 $O(T^2)$ 到 $O(T)$。在被注释掉的代码中，att = (q @ k.transpose(-2, -1)) 会生成一个形状为 (B, nh, T, T) 的中间矩阵（Attention Matrix）。痛点：如果序列长度 $T=1024$，这个矩阵的大小就是 $1024 \times 1024$。随着 $T$ 的增加，显存占用呈指数级增长，很容易导致 OOM (Out of Memory)。优化：SDPA 内部通常集成了 FlashAttention 算法。它通过 Tiling（分块） 技术，在不显式存储整个 $T \times T$ 矩阵的情况下计算输出，将显存复杂度降低到了线性级别。
        # 2. 算子融合（Operator Fusion。手动版本：涉及多次显存读写（矩阵乘法 $\rightarrow$ 掩码填充 $\rightarrow$ Softmax $\rightarrow$ 再次矩阵乘法）。每一行代码都是一个独立的 GPU Kernel 调用，数据需要在显存和寄存器之间反复搬运。SDPA 版本：这是一个融合算子（Fused Kernel）。它将缩放、掩码、Softmax 和加权求和全部封装在一个 CUDA Kernel 中完成。减少了内存带宽的开销，显著提升了执行速度。
        # 3. 硬件级加速。SDPA 会根据我们的硬件自动选择最优的底层实现：FlashAttention-2：针对 NVIDIA Ampere (A100) 及更现代的架构进行了高度优化。Memory-Efficient Attention：针对较旧的 GPU 架构。C++ / Triton 实现：确保在不同环境下都能跑出接近硬件极限的 TFLOPS。

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        # FlashAttention 最大的改动是它将矩阵拆分成小的 Block，在 SRAM（高速片上缓存）里循环计算。为了在内存受限的情况下计算 Softmax，它引入了特殊的在线 Softmax 算法（Online Softmax）。这种逻辑结构非常复杂，目前的通用编译器很难自动从普通的矩阵乘法代码推导出这种精妙的循环结构。
        # 虽然 Flash Attention 的计算量（FLOPs）其实比传统方法略多（因为反向传播有重算），但它极大地减少了内存访问（Memory IO）。在现代 GPU 上，计算很便宜，搬运数据才是最贵的。

        # 它把我们之前准备好的 $Q, K, V$ 拿过来，算出单词两两之间的相关性（Attention Score），并根据这个相关性对 $V$（信息内容）进行加权求和。
        # is_causal=True（关键点），这是为了 “防止偷看答案”。在训练 GPT 这种生成模型时，第 5 个单词只能看到前 4 个单词的信息，不能看到第 6 个。设置这个参数后，函数会自动把未来的信息屏蔽掉。
        # Mask 矩阵是对一句话的 Mask，如下所示：
        # [[1, 0, 0, 0, 0],
        #  [1, 1, 0, 0, 0],
        #  [1, 1, 1, 0, 0],
        #  [1, 1, 1, 1, 0],
        #  [1, 1, 1, 1, 1]]
        # 因为一句话有 $T$ 个词，所以词与词之间的关系是一个 $T \times T$ 的方阵。行（Row）：代表“当前正在处理哪个词”。列（Column）：代表“它在看哪个词”。对于因果注意力（Causal Attention）：当我们在处理第 3 个词（第 3 行）时，Mask 会允许我们看第 1, 2, 3 列，但挡住第 4, 5 列。这个逻辑对这句话里的每个词都适用。
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

    # 在 PyTorch 中，我们不应该直接显式调用 model.forward(x)。
    # 相反，我们应该像调用函数一样直接调用对象：model(x)。
    # 为什么？ 当我们执行 model(x) 时，PyTorch 会在后台调用一个叫 __call__ 的内置方法。
    # __call__ 做了什么？ 它不仅会运行我们写的 forward 逻辑，还会负责注册 Hooks（钩子）。这些钩子对于调试、可视化（比如 TensorBoard）至关重要。
    # 结论： 永远通过 out = model(x) 调用，而不是 out = model.forward(x)。
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
# 当我们打上这个装饰器时，Python 会自动为我们生成以下方法：
# __init__：自动处理参数传递。我们可以直接 config = GPTConfig(n_layer=24)。
# __repr__：自动生成美观的打印格式。直接 print(config) 就能看到完整的属性名和值。
# __eq__：自动支持比较。如果有两个配置类属性完全一样，config1 == config2 会返回 True。
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
# 在 PyTorch 中，nn.Module 是所有神经网络模块的基类。如果我们想构建一个模型（不管是像 GPT 这样的大模型，还是一个简单的线性层），继承 nn.Module 是必须的“入场券”。如果不继承它，我们写的只是一个普通的 Python 类；继承了它，我们的类就变成了拥有“超能力”的 神经网络组件。

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
        # 在模型刚开始训练的第一步，我们希望 LayerNorm 只负责“稳定分布”，而不是去“干扰信号”。所以设为 1 和 0，相当于告诉模型：“先按标准分布传下去，改不改由我们后面练出来再说。”
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
            # 2. LayerNorm。这是保证训练中方差稳定的定海神针。不管我们的 $W$ 和 $b$ 在训练中变成了什么样，在每一层线性计算结束后，信号都会经过 LayerNorm：强制归零：它计算当前这一批数据的均值 $\mu$，然后减去它。强制归一：它计算当前数据的方差 $\sigma^2$，然后除以 $\sigma$。
            
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
                # 之所以我们通过 $\sqrt{2N}$ 把支流压得很低，本质上是告诉模型：“我们刚出生的时候，先别急着大刀阔斧地修改前人的成果。我们先作为一个微小的修正项存在，等训练开始了，我们觉得确实有必要，再通过权重更新把自己调大。”
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
        # B. Pre-Norm（现在 GPT 的主流做法，即我们这段代码所属的架构）结构： x = x + SubLayer(LayerNorm(x))逻辑： 在每一层计算之前先归一化。优点： 梯度可以从残差连接（那个 x + ...）中无损地流向很深的层。这让训练几十层甚至上百层的模型变得非常稳定。为什么还需要 ln_f？ 因为 Pre-Norm 架构中，主干路径上的 $x$ 一直在累加，没有被整体归一化过，所以最后必须补一刀 ln_f，把数值拉回“圆润”的分布。

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
        # 逻辑：首先拿到模型中所有的参数名（pn）和参数对象（p）。
        # 筛选：只保留 requires_grad=True 的参数（即那些在训练中需要计算梯度并更新的参数，排除掉被冻结的层）。

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # decay_params (维数 $\ge 2$)：包括：线性层的权重矩阵（2D）、Embedding 层（2D）。逻辑：这些矩阵很大，最容易导致过拟合，所以需要执行 weight_decay。
        # nodecay_params (维数 $< 2$)：包括：所有偏置（Bias）（1D）、LayerNorm 的 $\gamma$ 和 $\beta$（1D）。逻辑：它们是用来控制归一化分布的。如果强迫 gamma 接近 0，会导致信号消失，模型反而练不动。因此这些参数不应该被强迫变小，因此设置衰减为 0。
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # p.numel()：计算参数张量中的元素总数（例如 $768 \times 768$）。
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # 在日志里清楚地看到，模型中有多少参数正在被“惩罚”，有多少在“自由生长”。
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # inspect.signature(...): inspect 是 Python 的内置模块，用于“自省”（即查看代码自身的信息）。这一步是获取 AdamW 构造函数的所有参数签名。.parameters: 获取该函数所有参数的名称列表。
        use_fused = fused_available and device_type == "cuda"
        # fused 是 PyTorch 较新版本引入的优化。
        # 原理：标准的 AdamW 会对每个参数跑一次内核循环。fused=True 会将多个操作合并为一个 CUDA 内核调用。
        # 好处：在 NVIDIA GPU 上，这能带来 10%~20% 的训练加速。

        # Fused AdamW 的核心思想是：“一次搬运，全部搞定。”它通过 CUDA 编写了一个专门的内核函数，能够一次性处理模型中的多个（甚至所有）参数。原理拆解：
        # 多张量合并 (Multi-Tensor Apply)：
        # Fused 模式不再是一个一个地处理张量，而是将所有待处理的张量指针（Pointer）列表一次性传给一个大型的 CUDA Kernel。
        # 内存访问优化：
        # 它在 GPU 内部实现了更高效的内存读取。由于只需要启动一次（或极少数次）内核，原本分散的、碎片化的显存访问变得更加连续和规范。
        # 指令并行：
        # 在同一个 GPU 线程块内，可以利用硬件特性同时更新多个参数，减少了 CPU 和 GPU 之间的通信频繁切换。
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        # 这里的参数来自于 GPT-3。
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
        # 当我们通过 self.register_buffer('name', tensor) 注册一个张量后，它获得了三个普通变量没有的“特权”：
        # - 跟随设备转移：当我们执行 model.to('cuda') 时，所有 Buffer 会自动从 CPU 搬运到 GPU。
        # - 自动序列化：当我们执行 torch.save(model.state_dict(), 'path.pth') 时，Buffer 会被包含在字典里存入硬盘。
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
                # 通过 shape[::-1]，我们实际上是在检查：“虽然这两个矩阵现在的形状不一样，但如果我把官方的转置（Transpose）一下，它们是不是就一样了？”
                # [::-1]: 这是 Python 列表/元组的切片操作，表示倒序（反转），如果原形状是 (768, 2304)，反转后就变成了 (2304, 768)。
                with torch.no_grad():
                    # with torch.no_grad(): 是一个上下文管理器（Context Manager），它的核心作用是：暂时关闭梯度计算系统（Autograd 引擎）。通俗点说，它告诉 PyTorch：“接下来的操作我只是在搬运数据或者做推断，不需要我们记录任何求导信息，也不需要为反向传播做准备。”
                    # 默认情况下，当我们对 PyTorch 的 Tensor（且 requires_grad=True）进行操作时，PyTorch 会在后台构建一个“计算图”来记录所有操作，以便后续计算导数。加载权重只是简单的赋值操作，完全不需要计算梯度。
                    # 它能带来什么好处？内存优化：不记录计算图意味着不占用额外的显存/内存来存储中间激活值。速度提升：省去了后台追踪计算逻辑的时间，运行效率更高。
                    # 一般来说，我们在模型推理或者手动加载权重时都需要关闭梯度系统。
                    # with torch.no_grad(): 的有效范围仅限于其下方所有缩进一级的代码行。一旦代码恢复到原来的缩进级别，梯度追踪就会自动恢复。如果我们想让整个函数都不追踪梯度，不需要缩进整个函数体，可以使用装饰器：@torch.no_grad()。
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
    # 在标准的 PyTorch 工作流中，数据加载通常被拆分为两个角色：Dataset：负责“有什么数据”以及“如何取某一条数据”（定义索引索引 $i$ 对应的内容）。DataLoader：负责“怎么取数据”（定义 Batch Size、是否打乱、多线程加载、分布式分发等）。我们提供的这段 DataLoaderLite 确实将这两者合二为一了。
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
    

# ⭐️ 对小批数据进行过拟合是确保训练 Pipeline 正确的关键，在业内，我们把它叫做 "Overfit a single batch"。如果我们能让模型在这一小撮数据（比如只有 1~2 个样本）上把 Loss 降到接近 0，或者准确率达到 100%，就证明我们的整个 Pipeline（流水线）是通的。
# ⭐️ “先过拟合一小批，再征服星辰大海。” 只有确定模型有“死记硬背”的能力，我们才能去谈论它的“泛化”能力。

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it): # 带预热的余弦退火学习率调度策略
    # 1) linear warmup for warmup_iters steps
    # 线性预热阶段，这是为了防止模型在训练刚开始时，由于权重随机初始化导致梯度过大，直接把模型“跑飞”了。
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    # 如果当前的迭代次数超过了设定的最大步数，学习率将不再下降，而是维持在一个极小的常数 min_lr。这通常是为了在训练最后阶段进行微调。
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    # 这是代码的核心，负责在预热结束后平滑地降低学习率，在训练后期，学习率下降速度变慢，能帮助模型在最优点附近进行精细搜索。
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)



torch.set_float32_matmul_precision('high')
# 允许 PyTorch 在进行矩阵乘法（MatMul）时，使用 TensorFloat-32 (TF32) 数据格式，从而在牺牲极小精度的情况下，大幅提升计算速度。
# 在传统的深度学习中，float32 (FP32) 是标准精度。FP32：精度高，但计算开销大。TF32：这是 NVIDIA 推出的一种中间格式。它结合了 FP32 的动态范围（指数位）和 FP16 的精度（尾数位）。当我们设置 precision='high' 时，GPU 内部会将 FP32 的矩阵乘法通过 TF32 核心来运行。这就像是给计算做了一次“无损（或极低损）压缩”，硬件跑起来飞快。
# 但是这只是计算变快了，实际的数据还是 fp32，因此通信成本还是比较高的。这正是 TF32 与 混合精度训练（Mixed Precision, FP16/BF16） 最大的区别。
# torch.set_float32_matmul_precision('high') 确实是一个“内部转换”：
# 输入/输出：依然是实打实的 32-bit (FP32) 占用。
# 内存/显存：没有任何节省，权重和激活值在显存里还是占 4 字节。
# 通信（Distributed）：多卡同步（All-Reduce）时，传的还是 FP32，带宽压力一点没减。
# 计算：只有在 GPU 核心内部 做乘法那一瞬间，被转成了 TF32 处理。

# 顺带在这里介绍下 fp32 tf32 fp16 bf16 的区别：
# FP32 (Full Precision)
# 结构：1位符号 + 8位指数 + 23位尾数。
# 现状：它是所有深度学习的基准。如果我们不差钱（显存够）也不差时间，用它永远最稳。

# TF32 (TensorFloat-32)
# 本质：它不是一种存储格式，而是一种运算模式。
# 逻辑：它取了 FP32 的指数位（保证数值范围够大，不会溢出）和 FP16 的尾数位（保证计算速度快）。
# 优势：在 RTX 30/40 或 A100 上，开启 matmul_precision('high') 后，原本 FP32 的矩阵乘法会自动走 TF32 单元，不需要改模型代码就能提速。

# FP16 (Half Precision)
# 挑战：它的指数位只有 5 位，最大能表示的数只有 65504。在训练深层网络时，梯度很容易超过这个值（上溢）或者变得太小（下溢）。
# 对策：必须配合 torch.cuda.amp.GradScaler 使用，通过缩放梯度来防止数值消失或爆炸。

# BF16 (Brain Floating Point)
# 设计思路：由 Google Brain 提出。它直接把 FP32 的 23 位尾数截断成 7 位，保留完整的 8 位指数。
# 优势：它和 FP32 拥有完全一样的数值范围。这意味着我们不用做任何“梯度缩放”，模型就能跑得很稳，且显存和通信带宽比 FP32 省了一半。它是目前大模型（LLM）训练的首选。

# 顺便介绍下如何理解指数和尾数：
# 我们可以用科学计数法来完美类比。在数学中，科学计数法写做：$$1.234 \times 10^{5}$$在这个例子中：$1.234$ 就是尾数（Mantissa/Fraction）：它决定了数字的精度（也就是这把尺子的刻度有多细）。$5$ 就是指数（Exponent）：它决定了数字的范围（也就是这个数字能有多大或多小，小数点往哪挪）。计算机存储浮点数（FP32/FP16等）的原理一模一样，只不过底数从 $10$ 变成了 $2$。
# 指数位决定了我们能表示的数值边界。如果指数位多（如 FP32, BF16 有 8 位）：我们可以表示极大的数（如 $2^{127}$）和极小的数（如 $2^{-126}$）。这就像我们的手特别长，既能摸到天上的星星，也能摸到地上的尘埃。如果指数位少（如 FP16 只有 5 位）：我们最大只能表示到 $2^{15}$（即 65504），最小只能到 $2^{-14}$。这就像我们的手很短，太远的东西（大数）我们摸不到（上溢），太小的微粒（小梯度）我们也抓不住（下溢/变成0）。
# 尾数位决定了我们在某个范围内能分得多细。如果尾数位多（如 FP32 有 23 位）：在 $1.0$ 到 $2.0$ 之间，我们可以切成 $2^{23}$（约 800 万）个小刻度。我们能分辨出 $1.0000001$ 和 $1.0000002$ 的区别。如果尾数位少（如 BF16 只有 7 位）：在 $1.0$ 到 $2.0$ 之间，我们只能切成 $2^{7}$（128）个刻度。如果我们想表达 $1.0000001$，由于刻度太粗，它会直接被“四舍五入”成 $1.0$。
# FP16 就像一个珠宝秤：它很精确（尾数位多），但量程很小（指数位少）。如果我们放个秤砣（大梯度），它就爆了；如果我们放粒灰尘（小梯度），它可能也感觉不到。
# BF16 就像一个改造过的工业秤：它的量程极大（指数位多），虽然看不太准克数（尾数位少），但它绝不会因为数字太小而直接罢工，它总能给我们一个大概的数值。激进派：牺牲精度换范围。它认为在深度学习中，“能表示极小的梯度”比“表示得极其精确”更重要。


# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# 先简单介绍一下目前主流的分布式训练方案及其对应的实现。
# 首先是技术维度，这是解决问题的逻辑手段。
# DP (Data Parallel): 单进程多线程（老旧，基本废弃）。
# DDP (Distributed Data Parallel): 多进程，每卡完整模型（主流标配）。
# TP (Tensor Parallel): 拆分算子/矩阵（针对超宽层）。
# PP (Pipeline Parallel): 拆分层/模型段（针对超深模型）。
# ZeRO / FSDP: 参数分片（DDP 的进化版，解决了显存冗余）。

# 然后是工具维度，这是我们写代码时真正调用的东西。我们可以把它们分成三个层级：
# A. 底层引擎 (Engines)
# 这些框架真正实现了 TP、PP 或复杂的显存优化算法。
# DeepSpeed (Microsoft): 实现了 ZeRO 1/2/3、Offload、PP。
# Megatron-LM (NVIDIA): 实现了极致性能的 TP 和 PP。
# PyTorch 原生: 提供了 DDP 和 FSDP 两个核心类。FSDP 是 PyTorch 1.11 推出的原生分布式训练技术。我们可以把它看作是 PyTorch 官方版的 DeepSpeed ZeRO-3。

# B. 启动器 (Launchers)
# 负责把我们的代码分发到各个进程/机器。
# torchrun: PyTorch 官方推荐的启动工具（处理环境变量、容错重启）。
# deepspeed (CLI): DeepSpeed 自带的启动器。

# C. 高层封装/脚手架 (Wrappers/High-level)
# 它们不发明算法，而是让我们更容易地切换不同的底层引擎。
# Hugging Face Accelerate: 一个轻量级工具。我们写一份代码，通过配置文件就能决定是用 DDP、FSDP 还是 DeepSpeed。
# PyTorch Lightning: 一个更重的框架。它定义了一套写代码的规范（如 training_step），通过参数 strategy="ddp" 或 "deepspeed" 自动帮我们调用底层逻辑。

# 因为 Karpathy 主要实践的是 DDP，因此先介绍一下 DDP。
# 在 PyTorch 的大规模训练中，DistributedDataParallel (DDP) 是目前的行业标配。它比早期的 DataParallel (DP) 更快、更稳定，且能够跨多台机器扩展。简单来说，DDP 的核心逻辑是：在每个 GPU 上启动一个进程，每个进程拥有模型的一个副本，独立计算梯度，最后通过高效的通信协议同步梯度。

# DDP 的运行遵循以下几个关键步骤：
# 模型副本 (Model Replicas): 在训练开始时，主进程将模型的初始参数广播到所有进程，确保起点一致。
# 数据分发 (Data Partitioning): 使用 DistributedSampler 确保每个进程在每一轮迭代（epoch）中看到不同的数据子集。
# 梯度同步 (Gradient Sync): 这是最关键的一步。DDP 使用 All-Reduce 算法。当进程计算完梯度后，它们会相互通信并取梯度的平均值。
# 重叠计算与通信: DDP 不会等所有梯度算完才通信。它将参数分成多个“桶（Buckets）”，当某一部分梯度算完时，就立即开始后台通信，从而隐藏网络延迟。

# 那么，是不是 DDP 条件下，每个进程中模型的参数每时每刻都一致？
# 答案是，不是每时每刻都一致，但在最关键的时刻（前向传播和参数更新时）是一致的。在 DDP 的运行循环中，模型参数的状态经历了一个“从统一到分歧，再回归统一”的过程。我们可以把这个过程拆解开来看：
# 1. 初始时刻：绝对一致在训练开始时（调用 DDP(model) 时），主进程（Rank 0）会将自己的参数状态 Broadcast（广播） 给所有其他进程。此时，所有进程的模型参数是 $100\%$ 镜像一致的。
# 2. 前向传播 (Forward)：保持一致由于参数一致，且每个进程拿到的数据（虽然不同）都会经过相同的数学运算，此时各进程模型中的参数依然保持一致。
# 3. 反向传播 (Backward)：分歧出现这是最关键的点。梯度（Gradients）： 每个进程计算的是自己那部分数据的梯度。因为数据不同，产生的梯度是不一样的。参数（Parameters）： 在这个阶段，参数本身还没变，依然是一致的。
# 4. 梯度同步 (All-Reduce)：重合点当反向传播进行时，DDP 启动了 All-Reduce 通讯。它将所有进程的梯度累加并取平均值。结果： 每一个进程现在都拿到了完全相同的“平均梯度”。
# 5. 优化器更新 (Optimizer Step)：回归一致当执行 optimizer.step() 时：每个进程都拿着那份相同的平均梯度去更新自己本地的参数。既然起点（参数）一样，步长和方向（平均梯度）也一样，那么更新后的结果（新参数）自然又是完全一致的了。

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
# 这里的 vocab_size 从 vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token 改成 vocab_size=50304，为什么可以进一步加速训练？
# 这是一个非常经典且硬核的工程优化问题。将 vocab_size 从 50257 增加到 50304，看似多加了 47 个无用的空 token，但实际上是为了满足现代 GPU 架构对内存对齐（Memory Alignment）和算力分配的偏好。
# 核心是因为 50304 满足“8 的倍数”原则。现代 GPU（尤其是 NVIDIA 的 Ampere、Hopper 架构）在进行矩阵乘法时，使用 Tensor Cores 进行加速。硬件偏好：Tensor Cores 在处理维度是 8、16、32 或 64 的倍数 的矩阵时效率最高。计算效率：50257 是一个素数（且是奇数），无法被 8 整除。而 $50304 = 128 \times 393$。结果：当维度对齐到 128 的倍数时，GPU 可以启用更高效的 CUDA Kernel，减少指令周期，避免处理边缘情况（Padding logic）带来的额外开销。
# 由此带来的一个问题是，词表嵌入和分类头共享了权重参数，二者都不会关注多出来的 token 吗？
# 1. 嵌入层（Embedding）：它们被锁在了“门外”嵌入层本质上是一个查表操作（Lookup Table）。输入限制：我们的分词器（Tokenizer）是模型唯一的“守门人”。如果 Tokenizer 的词表大小是 50257，那么它输出给模型的 ID 永远只会在 $0$ 到 $50256$ 之间。结果：索引 $50257$ 到 $50303$ 的那些权重行，永远不会被索引到。在每一轮前向传播中，这些多出来的权重根本没机会进入模型的主体计算流程。
# 2. 解码层（Linear/Softmax）：它们被“无视”了在模型的最后一层，模型会为所有 50304 个 token 计算一个得分（Logit）。训练阶段：我们的训练数据（Label）里永远不会出现 ID 为 50257 的目标。当模型计算 Cross Entropy Loss 时，由于这些多出来的 token 永远不是“正确答案”，损失函数会不断告诉模型：“这个位置的概率应该是 0”。
# 这种做法唯一的副作用是： 我们的模型权重文件（.bin 或 .safetensors）会因为多了这 47 个向量而变大那么几百 KB。但比起训练速度提升的 5%~10%，这点硬盘空间完全可以忽略不计。

model = torch.compile(model)
# torch.compile(model) 是一个非常重要的优化手段，标志着 PyTorch 从**即时执行（Eager Mode）向编译模式（Graph Mode）**的跨越。
# torch.compile 的底层由三个核心技术支撑：

# ① TorchDynamo (捕捉图)
# 它使用 Python Frame Evaluation API 来拦截 Python 字节码。即使我们的模型里有复杂的 if-else 分支或第三方库调用，它也能通过“图追踪”把能加速的张量计算部分提取出来，形成一个 FX Graph。也就是说，它利用 TorchDynamo 像“探照灯”一样扫描我们的 Python 字节码，把一系列离散的算子打包成一个完整的计算图（Graph）。消除 Python 开销： 一旦图被捕获并编译成内核，执行时就不再需要 Python 解释器介入，直接在底层高性能运行。
# ② AOTAutograd (处理反向传播)
# 传统的 PyTorch 动态生成反向传播。torch.compile 则是“提前”（Ahead-Of-Time）生成反向传播的计算图，并将其纳入优化范围。
# ③ OpenAI Triton (生成内核)
# 这是最关键的一步。它将优化后的计算图转化为 Triton Kernels（一种类 CUDA 的高效编程语言）。算子融合 (Operator Fusion)： 这是提升速度的核心。例如，它能把 ReLU(Add(A, B)) 这三个原本需要三次内存读写的操作，合并成一个 CUDA Kernel。原本数据要回传到显存三次，现在只需一次。

model.to(device)
# torch.Tensor（张量）：必须重新赋值，需要是 tensor = tensor.to(device)。
# nn.Module（模型/层）：不需要重新赋值，需要 model.to(device)。
# 当我们写 optimizer = torch.optim.Adam(model.parameters()) 时，一定要先把 model.to('cuda')，再把模型参数传给优化器。如果先建优化器再移模型，优化器里记下的参数地址可能还是 CPU 上的，会导致报错。
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

for step in range(max_steps):
    model.train()
    # 这行代码的作用是将模型设置为训练模式。
    # 有些模型层（Layer）在“训练”和“推理（预测）”时的行为是完全不同的。最典型的例子有两个：
    # Dropout 层：在训练时会随机“关掉”一部分神经元来防止过拟合；但在预测时，必须全部打开。
    # Batch Normalization（批归一化）层：在训练时会计算当前 Batch 的均值和方差；但在预测时，会使用整个训练集累积下来的均值和方差。
    optimizer.zero_grad()
    # 这行代码的作用是把模型参数中缓存的梯度全部归零。
    # 为什么要归零？在 PyTorch 中，梯度的计算有一个特殊的设定：梯度是累加的（Accumulative）。
    # 当我们调用 loss.backward() 时，新计算出的梯度会加到原有的梯度上，而不是替换掉它们。后果：如果我们不手动归一化，第二步的梯度就会加上第一步的梯度，导致梯度数值越来越大，模型参数更新方向彻底乱套。
    # 但有时候也会去做梯度累计，下面是小 batch 梯度和大 batch 梯度的作用：
    # 小 Batch 的路径——“醉汉走路”：虽然它会左右晃荡（震荡），但这种震荡有时是好事。它能帮助模型跳出那些微小的“局部最小值”（Local Minima）或“鞍点”，因此泛化能力好。但因为样本少，这 2 个样本可能刚好是“特例”（比如都是生僻词或长句子）。计算出的梯度方向可能会偏移大部队的方向。
    # 大 Batch 的路径——“高铁行驶”：路径非常直，非常稳。但由于它太稳了，一旦进入一个坑（局部最优解），它可能就直接停在那了，没有足够的随机性能让它跳出来。但样本多（比如 128 个），通过平均，那些极端的特例被抵消了。计算出的梯度更接近整个数据集的真实梯度方向。
    loss_accum = 0.0
    # 
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # 这段代码是实现**自动混合精度（Automatic Mixed Precision, AMP）**的核心。简单来说，它是让模型在保持准确率的同时，跑得更快、省更多显存。
            # 通常情况下，深度学习模型使用 FP32（单精度浮点数，32位）进行计算。混合精度则是指在训练过程中，同时使用 FP32 和 FP16/BF16（半精度，16位）。FP32： 精度高，范围广，但计算慢，占用显存多；BF16/FP16： 精度较低，但计算速度极快（通常是 FP32 的数倍），占用显存减半。如果使用了 FP16 需要用 torch.cuda.amp.GradScaler()。
            # torch.autocast 的原理本质上是**“各司其职”**，自动切换，它会自动检测神经网络中的操作。
            # 对于计算密集型的操作（如矩阵乘法 Linear、卷积 Conv），它会将数据转为 BF16，利用 GPU 的 Tensor Cores 加速。
            # 对于对精度敏感的操作（如 Softmax、Loss 计算、LayerNorm），它会保持使用 FP32 以确保数值稳定性。
            # 权重更新： 虽然前向传播和反向传播可能使用低精度，但主权重（Master Weights）通常以 FP32 存储，以防止梯度更新时的微小变化被“舍入”掉。
            # device_type：指定设备（如 'cuda' 或 'cpu'）。
            # dtype：指定使用的低精度类型（这里是 torch.bfloat16）。
            # 上下文管理器 (with)：它只在这个代码块内生效。一旦退出，系统会恢复到默认精度。
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # 如果不 detach()：当我们执行 loss_accum += loss 时，PyTorch 会认为我们可能还要对 loss_accum 进行反向传播。为了能算导数，它会把产生这个 loss 的所有中间变量（激活值、权重指针等）都保留在显存里。
        # 后果：随着循环进行（比如累积 64 步），显存会像滚雪球一样迅速堆积，直到 OOM（显存溢出）。
        # detach() 的作用：它像一把剪刀，把 loss 这个值从复杂的计算图中剪下来。现在它只是一个单纯的、孤零零的数字（Tensor），不再关联任何模型参数，也不会占用额外的显存来保存中间状态。
        # .item() 也会剥离计算图，但是他还会从 GPU 2 CPU，而 detach() 则是只剥离计算图，然后还在 GPU 中。
        loss.backward() # loss 去做反向传播，计算梯度。
    
    # 上述过程在做梯度累积，有三个问题：

    # 1. 为什么要梯度累积？大 Batch Size 有什么好处？
    # 简单来说，梯度累积是为了在显存有限的情况下，“伪装”出大 Batch Size 的效果。
    # 数值稳定性（更准的梯度）。小 Batch 抽取的样本太少，产生的梯度随机性很大（噪声多）。大 Batch Size 相当于在一次更新前看了更多数据，计算出的梯度方向更接近整个数据集的真实梯度方向，训练曲线更平滑。
    # 收敛速度。在很多现代架构（如 Transformer）中，较大的 Batch Size 配合适当调高的学习率，通常能让模型更快收敛，并达到更好的泛化效果。
    # 硬件效率。GPU 的并行计算能力非常强。如果 Batch 太小，GPU 还没发力任务就结束了，导致利用率低。大 Batch 能填满 GPU 的计算单元，提高吞吐量（Tokens/sec）。

    # 2. 为什么 loss = loss / grad_accum_steps？
    # 其实简单来说，就是因为分母没有变。想象我们在算全班 64 人的平均分，但我们手里的表格一次只能填 8 个人：
    # 错误做法：算出每 8 个人的平均分，然后把这 8 个平均分直接加起来。（结果会比真实平均分大 8 倍）
    # 正确做法：算出每 8 个人的平均分，先除以 8，再把它们加起来。（这就得到了全班的真实平均分）

    # 3. 梯度裁剪 clip_grad_norm_ 需要等量提高吗？
    # 不需要提高。因为我们在计算梯度之前已经通过 loss / grad_accum_steps 修正了分母，
    # 所以最终累积出来的梯度，物理意义上依然是一个“平均梯度”。

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # 这行代码执行的是深度学习中非常关键的一个步骤：梯度裁剪（Gradient Clipping）。简单来说，它的作用是防止“梯度爆炸”，让模型训练更稳健。如果把模型训练比作下山，这行代码就像是给我们的鞋底加了防滑垫，防止我们步子迈得太大直接摔下山崖。
    # torch.nn.utils.clip_grad_norm_ 的工作流程如下：
    # 计算总范数（Total Norm）：它会把模型中所有参数的梯度当成一个巨大的向量，计算这个向量的 $L_2$ 范数（即长度）。对所有参数求和（计算范数），是为了把整个模型看作一个不可分割的整体。这保证了在遭遇“梯度爆炸”需要刹车时，我们是均匀地踩下四个轮子的刹车（等比例缩放），而不是只锁死其中一个轮子（导致赛车甩尾失控）。
    # 判断是否超标：我们设置的第二个参数 1.0 就是我们的“阈值”（max_norm）。如果 Total Norm <= 1.0：梯度保持不变，原样通过。如果 Total Norm > 1.0：它会把所有梯度等比例缩小，使得缩放后的总范数正好等于 1.0。
    # 原地修改：函数名末尾的下划线 _ 表示这是一个 inplace 操作，它会直接修改 model.parameters() 里的 .grad 属性。
    # 这个 norm 返回的是裁剪之前梯度的原始总范数。如果 norm 经常远大于 1.0，说明我们的模型训练非常不稳定。如果 norm 变成 NaN 或 Inf，说明我们的训练已经崩了。很多开发者会把这个 norm 记录到 TensorBoard 或 WandB 中，作为判断模型收敛状态的“体温计”。

    # 如果我们在 TensorBoard 上观察 grad_norm，模型训练正常的话：
    # 初期：像心电图一样剧烈抖动，数值很高。
    # 中期：抖动幅度收窄，整体趋势向右下方倾斜，但保留一定的“毛刺”（代表模型仍在对不同数据产生反应）。
    # 后期：数值变得很小且非常平稳，说明模型已经“吃饱了”，再怎么训练也难有大改进。

    # grad_norm 本质上就是梯度的宏观表现。只要这个值：不是 0（说明还在学）；不是 NaN（说明没崩）；正在慢慢变小（说明在变强）。那我们的训练就是在通往成功的路上。
    # 把 grad_norm 看作**“优化空间的剩余量”**是非常天才的视角。在实际大模型训练中，很多资深算法工程师甚至不看 Loss（因为 Loss 受到数据难度影响很大），而是盯着 grad_norm 看。只要它在稳步下降，大家就敢放心去睡觉；如果它突然“诈尸”跳高，那就是要出事的预兆。
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # 数据结构上：每个 param_group 都可以拥有独立的 lr。
        # 通过这个循环，我们把计算出的余弦退火学习率应用到了所有的参数组上，使它们步调一致。
    optimizer.step()
    # 优化器根据梯度更新模型参数。
    # torch.cuda.synchronize()
    # 这个方法会强制让 CPU 等待 GPU 完成当前排队的所有任务。
    # CPU 和 GPU 是异步执行的，当我们调用 model(inputs) 或 loss.backward() 时，Python（CPU）并不会等 GPU 算完，它只是把指令“扔”给 GPU 队列，然后立刻执行下一行代码。
