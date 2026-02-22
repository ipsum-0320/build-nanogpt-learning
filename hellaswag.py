"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")
# 定义数据存储路径

def download_file(url: str, fname: str, chunk_size=1024):
    """通用的下载函数，带进度条"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    # 确保本地有 HellaSwag 的数据文件（JSONL 格式）。
    """根据指定的 split (train/val/test) 下载对应的 HellaSwag 文件"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    # 这是最重要的一段，它决定了如何把“题目”喂给模型。
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"] # 题目背景
    label = example["label"] # 正确答案索引 (0-3)
    endings = example["endings"] # 四个候选项

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx) # # 将背景文字转为 Token ID 列表
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        # 关键点：GPT-2 这种模型在拼接时，通常要在后续文本前加空格
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        # 将 背景 + 结局 拼接在一起
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        # mask 用于区分：背景部分为 0，结局部分为 1
        # 我们只计算模型预测“结局”部分的准确度
        data["ending_tokens"].append(end_tokens)

    # 由于 4 个结局长度不一，需要进行填充（Padding）对齐成 4xN 的矩阵
    # 默认是使用 0 进行填充的。
    # Padding Mask 核心是 `mask` 中的 0 发挥了作用。
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
        # 主要是 Padding Mask。

    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    # 下载 hellawag 数据集。
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        # 打开 val 的 jsonl 数据，然后读取其中的每一行（json）。
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
# 评测模式，不需要计算梯度（节省内存）
def evaluate(model_type, device):
    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    # 这里的 model 就是一个端到端的大语言生成式模型。
    model.to(device)
    # model = torch.compile(model) # optionally torch compile the model

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        # tokens：背景 + 结局，他就是一个 `二维` 的矩阵。
        # mask：遮罩。
        # label：正确标签。
        tokens = tokens.to(device)
        mask = mask.to(device)

        # 1. 获取模型的预测结果 (Logits)
        logits = model(tokens).logits
        # 2. 计算损失：我们需要对比“模型预测的下一个词”和“实际的下一个词”
        # shift_logits: 取前 n-1 个位置的预测
        # token 就是一个 `二维` 的矩阵，B,T。
        # logits[..., :-1, :] 分别表示了 ...（Batch）、:-1（直到最后一个 token）、:（对整个词表所有候选词的预测得分）。
        # 最后一维的大小是 词表大小（Vocab Size，比如 GPT-2 是 50257），每一位代表模型认为“下一个词是词表里第 $i$ 个词”的得分（未经过 Softmax 的原始数值）。

        # 关于这里的 logits，要补充一个很重要的概念。
        # 那就是 prefill。
        # 首先我们需要知道模型训练时的表现是什么，那就是并行化，对于 a b c，他会同时执行三个训练任务：
        # a -> b
        # a -> b -> c
        # a -> b -> c -> [END]
        # 那么，因此看似是我们输入了一长串的 prefix context，但其实等效于我们输入了 n 行 prefix tokens，然后预测下一个 token。
        # 所以，在训练中，logits 的 size 为 (B, T, Vocab Size)。

        # 而在模型的推理过程中，我们之前的认知一直是端到端的做 Gen，但其实也会有一个关键的 Prefill 过程，流程和 training 一样。
        # 例如，当我们执行 logits = model("qwertyui").logits 时，模型并不会傻傻地先算 q 再算 w。它会利用 GPU 的并行能力，一次性处理这 8 个字符。此时，输出的 logits 形状是 $(1, 8, V)$。这 8 个位置其实包含了丰富的信息：
        # - logits[:, 0, :]：基于 q 预测下一个字符（它预测的结果应该是 w）。
        # - logits[:, 1, :]：基于 qw 预测下一个字符（它预测的结果应该是 e）。
        # - ...
        # - logits[:, 7, :]：基于完整的 qwertyui 预测下一个字符（比如可能是 o 或 p）。
        # 上述流程都是并行在做的，和我们的 training 流程完全一样。
        
        # 那么，为什么还需要 Prefill 的过程呢？因为没有必要基于 q 预测 w 呀？
        # 其实就是为了 KVCache，预测出来的 logits 反而没有用。模型必须通过一次类似于训练的完整前向传播，把我们的输入（qwertyui）转化成它能理解的高维向量堆栈。只有当这些向量填满了它的“短期记忆”（KV Cache），它才能在下一时刻表现得像是一个知道我们在说什么的智能体。下面是一个简单的总结：

        # 在训练中：
        # - Logits： 全都要。 每一个位置预测得准不准，直接决定了梯度的更新。
        # - KV Cache： 不需要。 训练是静态的，不需要保留状态给下一回合，算完就扔。
        # 在推理（Prefill）中：
        # - Logits： 只要最后一个。 之前的 $T-1$ 个预测（比如已知 q 预测 w）纯属计算过程中的“副产品”，被直接丢弃。
        # - KV Cache： 命根子。 这是 Prefill 唯一的“遗产”，没有它，后面的生成（Decoding）就会失忆。

        shift_logits = (logits[..., :-1, :]).contiguous() # 不包括最后一个，因为最后一个没有后续的预测 label。
        # .contiguous() 是一个看似不起眼但非常重要的函数。它的作用是在内存中把张量（Tensor）重新排列成连续的块。
        # shift_tokens: 取后 n-1 个位置的真实标签。
        shift_tokens = (tokens[..., 1:]).contiguous() # 不包括第一个

        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # prefix context - (B * (T - 1), V)
        flat_shift_tokens = shift_tokens.view(-1) # label - (B * (T - 1))
        # 这两行代码是深度学习（尤其是 NLP 训练）中计算交叉熵损失（Cross Entropy Loss）前的标准准备动作。简单来说，它们在做**“压平”**操作：把三维的概率张量和二维的标签张量，全部展平为一维（或二维矩阵），以便对齐并计算每个位置的得分。

        # 使用交叉熵计算每个位置的 Loss。
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        # 计算交叉熵，同时保留每一个 token 的损失（因为 reduction='none' 参数）。
        # shift_losses 是一个长向量（size 为 (B * (T - 1))），里面的每一个元素代表模型预测某一个具体词时的“痛苦程度”，Loss 越大，猜得越不对。
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # 将那一长串损失值重新折叠回二维矩阵，size 为 B, T - 1。

        # 3. 过滤出“结局”部分的 Loss
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        # 只计算模型真正该负责的那些词的 Loss，而无视掉那些填充词（Padding）或不需要计算的位置。
        # 如果是 Padding token，那么 shift_mask 对应位置上的元素值为 0。
        masked_shift_losses = shift_losses * shift_mask

        # 4. 做出决策
        sum_loss = masked_shift_losses.sum(dim=1)   # 总 Loss
        avg_loss = sum_loss / shift_mask.sum(dim=1) # 平均每个 Token 的 Loss (归一化)
        # exp(avg_loss) 就是 困惑度（Perplexity, PPL）。
        # 如果我们算出的 困惑度是 10，它的物理意义是：“模型在预测下一个词时，表现得就像是从 10 个 同样可能的备选词中做选择一样”。所以，困惑度越小，模型越“笃定”。

        # 同时，这种“逐 token 损失”在 RL 中也有一些应用。
        # 首先，我们需要介绍一下，什么叫做高熵 token，简单来说，熵（Entropy） 是衡量“混乱度”或“不确定性”的指标。高熵 Token 就是那些让模型感到非常纠结、完全吃不准该选哪一个的位置。
        # 但是我们不能够根据逐 token 的 loss 来表述熵的大小，举个例子，模型预测 $ P(A)=0.9, P(B)=0.1 $。这是一个低熵状态（很笃定）。如果正确答案恰好是 B，此时 Loss 会极大，但熵依然很低。

        # 那么应该如何描述熵呢？
        # 答案是使用 logits，如果我们拿到了逐 token 的词表概率分布，那么就可以描述熵了，对于某一个位置的 token，其熵 $ H $ 的计算不需要任何标签（Label），只需要概率分布，H = -sum(p * logp)，p 是词表的概率分布。

        # 有了逐 token 的熵，我们就能开启很多高级的玩法，这在**推理模型（Reasoning Models）**中非常常见：
        # - 识别“逻辑路口”：在生成过程中，如果发现某个 token 的熵突然升高，说明模型进入了逻辑决策区；如果熵很低，说明它在输出固定的套话。
        # - 自适应搜索：如果熵很高，模型可以开启“多路径搜索”（比如 Beam Search），尝试不同的可能性；如果熵很低，直接贪婪搜索（Greedy Search）最快的那个词。
        # - 强化学习（RL）的惩罚项：在 RL 训练中，我们经常会加一个 熵正则项（Entropy Bonus），鼓励模型保持一定的熵，防止它过早地收敛到某一种死板的回答上（即保持生成的“多样性”）。
        # 逐 token Loss 是“他评”（通过标准答案看表现），而逐 token 熵是“自评”（看自己内心的确定性）。

        # ⭐️⭐️⭐️ 在了解了熵（entropy）的概念后，在 LLM RL 领域，有一种算法是基于熵过滤的 loss 加权，后续可以重点看看。⭐️⭐️⭐️
        
        # Loss 越小，代表模型认为这个结局出现的概率越高
        # pred = sum_loss.argmin().item()         # 选总概率最大的
        pred_norm = avg_loss.argmin().item()    # 选平均概率最大的（更科学）

        # 统计正确率。
        num_total += 1
        # num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
