# 其它的开源数据集包括了：
# - https://github.com/togethercomputer/RedPajama-Data。
# - https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
# - https://github.com/huggingface/fineweb-2
"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
# tiktoken: OpenAI 开发的高性能分词器，这里使用了 GPT-2 的编码方式。
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B" 
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards
# shard_size: 这样做的目的是防止单个文件过大，方便在训练时流式读取或在多机之间分配数据。

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
# # 从 Hugging Face 下载指定的训练集。

# 初始化 GPT-2 编码器，其编码方式采用了 BBPE。
enc = tiktoken.get_encoding("gpt2")
# enc 是一个 Map 吗？
# 从表现形式上看，是的。但在底层实现上，它比简单的 Python dict 要复杂得多：
# 双向映射：它不仅实现了 Token -> ID（编码/Encoding），还实现了 ID -> Token（解码/Decoding）。
# 高效查找：tiktoken 是用 Rust 编写的，它内部使用了极其优化的 Trie 树或哈希表结构，每秒可以处理数十万个字符，比传统的 Python 原生分词器快得多。
# 不仅仅是映射：它还存储了 Merge Rules（合并规则）。BPE 算法不是一次性查表，而是根据预定义的规则，不断地将小的字符块合并成更大的 Token。

eot = enc._special_tokens['<|endoftext|>'] # end of text token
# 获取 <|endoftext|> 标记的 ID，用于作为文档之间的分隔符。
# 这一行从分词器中提取 End of Text（文本结束） 标志对应的数字 ID。
# - <|endoftext|>：这是一个特殊的标记（Special Token）。在 GPT-2 中，它的作用有两个：
# - 界限标记：告诉模型一段完整的文本到此结束，下一段文本与此无关。
# - 填充/占位：在批处理中，如果文本长度不一，有时也用它填充。
# - enc._special_tokens：这是一个字典，存储了所有特殊标记及其对应的 ID。
# - 结果： 对于 GPT-2 来说，这个 eot 的值通常是 50256。

def tokenize(doc):
    # 将单个文档（一段文本）转换为 uint16 类型的 numpy 数组
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # 每个文档开头先放一个结束符，起到隔离文档的作用
    tokens.extend(enc.encode_ordinary(doc["text"])) # 将文本转为数字 ID
    # encode_ordinary 的过程可以拆解为以下三步：
    # 文本标准化：处理空格、换行符等。
    # Byte-level BPE 拆解：根据 GPT-2 的规则，将单词或字符拆分成最小的 Token 单元。
    # 例如，常见的词  "hello" 可能会被直接映射为一个 ID。
    # 罕见的词或符号会被拆分成多个字节（Bytes）对应的 ID。
    # encode_ordinary (普通文本版)：它会把所有的输入都当做普通纯文本。如果文档内容里有人写了 "<|endoftext|>" 这几个字符，它不会触发任何特殊逻辑，而是会像对待普通单词一样，把它拆解成 <、|、end、of 等一系列普通字符的 ID。
    tokens_np = np.array(tokens)
    # 检查 token ID 是否在 uint16 范围内（0-65535），GPT-2 词表大小约 50k，没问题
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    # 将 numpy 数组保存为 .npy 文件
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
# 使用一半的 CPU 核心进行并行分词
with mp.Pool(nprocs) as pool:
    # 它创建了一个“进程池”，让你的电脑能够同时利用多个 CPU 核心来处理数据，而不是排队一个一个来。
    # Pool(nprocs): 这是一个“员工池”。nprocs（通常是你 CPU 核心数的一半）代表池子里有多少个“员工”在待命。
    # with ... as pool: 这是一个上下文管理器。它保证了任务干完后，这些“员工”会被自动遣散（释放内存和资源），不会常驻后台占用系统资源。
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    # 预分配一个全空的 buffer，用来存放当前分片的所有 token
    token_count = 0
    progress_bar = None

    # 下面的数据切割过程非常粗暴，这难道不会造成语义中断，从而使得数据训练受影响吗？
    # 这种“切断”方式看起来确实很“暴力”：一个句子可能在中间被劈开，上半句在 shard_001.npy 的结尾，下半句跑到了 shard_002.npy 的开头。但神奇的是，在 LLM（大语言模型）的预训练中，这不仅不是问题，反而是一种标准做法。以下是为什么这种“语义中断”不会影响训练的几个核心原因。
    # 1. 训练时的“滑动窗口”机制。模型在训练时，并不是一个分片（Shard）一个分片独立读取的。数据流化（Streaming）：训练代码通常会将所有分片看作一个连续的字节流。拼接（Packing）：DataLoader 会把所有的 Token 连成一长串。如果模型一次训练的上下文长度（Context Length）是 1024，它就从这个长串里每隔 1024 个 Token 截一段。跨越分片：当读取到 shard_001 的末尾时，程序会立刻无缝衔接打开 shard_002。对于模型来说，它看到的依然是连续的 Token 序列，根本感觉不到“文件分片”的存在。
    # 2. <|endoftext|> 的“隔离墙”作用。还记得代码里那行 tokens = [eot] 吗？每一篇文档开头都强行加了一个“结束符”。语义标记：当模型读到这个 ID（50256）时，它在数学上被训练去理解：“前面的信息到此为止，后面的信息是全新的开始。”中断的影响：即使一个文档被物理分片切断了，只要它在逻辑上（Token 序列中）还是连续的，模型就能通过 eot 识别文档边界。
    # 3. 这里的“切割”只是为了磁盘管理。shard_size = 1e8（1 亿 Token）的设计目的纯粹是为了工程方便：内存友好：如果把 100 亿 Token 存成一个 20GB 的巨型文件，读取和随机访问会非常慢。多机训练：在分布式训练时，我们可以让机器 A 读前 10 个分片，机器 B 读后 10 个分片，实现并行。容错性：如果下载或处理过程中断，只需要重跑出错的那个分片，而不是全部重来。

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # imap 把数据集 fw 中的每一个文档（doc）依次取出来，丢给 tokenize 函数去处理，得到 tokens。
        # chunksize 使得主进程一次性打包 16 个文档发给一个子进程。子进程埋头干完这 16 个活儿后再回来汇报。
        # pool.map 会试图一次性把所有结果都计算出来并存在内存里，对于 10T 的数据，这会直接导致 OOM (内存溢出)。
        # pool.imap 是懒加载的，它只在循环需要下一个结果时才去取，是处理大规模数据集的标准姿势。
        if token_count + len(tokens) < shard_size:
            # 如果当前 buffer 还能放得下这篇文档
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # # Buffer 满了，需要切分数据。shard_index 为 0 的作为验证集(val)，其余为训练集(train)
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")

            # 计算当前文档能填满当前 buffer 剩余空间的长度
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

            # 保存填满的分片
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None

            # 将文档剩下的部分放到下一个分片的 buffer 开头，这是在做直接的覆盖。
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # 循环结束后，如果 buffer 里还有没写满一个 shard 的剩余 token，也存起来
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
