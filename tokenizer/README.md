# Tokenizer

## token algorithm

优秀的分词算法应该避免OOV(out of vocabulary), 同时词表要尽可能地小，每个token有对应的含义。

- [FlagAI-Open/tokenization](https://github.com/FlagAI-Open/FlagAI/blob/master/doc_zh/tokenization.md)

### sample sentence

```
Let's tokenize! Isn't this easy?
```

### word-based tokenizaton

以词为单位对文本进行切分，这种方式在处理大量语料的时候会生成一个巨大的词表（因为大量的语料中会存在大量的只出现一次/少次的token）。比如 Transformer-XL基于空格和标点分词，词表大小是267,735。

- 基于空格切分: 会将标点与单词组合

```
["Let's", 'tokenize!', "Isn't", 'this', 'easy?']
```

- 基于标点切分： 对"Isn't" 的处理稍有瑕疵

```
["Let", "'", "s", "tokenize", "!", "Isn", "'", "t", "this", "easy", "?"]
```

- 基于规则切分，如spacy

```
["Let", "'s", "tokenize", "!", "Is", "n't", "this", "easy", "?"]
```

### character-base tokenization

基于字符级别的分词方法, 词表小，但由于单个字符没有意义，学习难度大。

```
['L', 'e', 't', "'", 's', ' ', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'e', '!', ' ', 'I', 's', 'n', "'", 't', ' ', 't', 'h', 'i', 's', ' ', 'e', 'a', 's', 'y', '?']
```

### subword tokenization

子词切分方法在词级别和字符级别中进行了折中：

- 词级别的分词：1、词表巨大；2、容易出现大量词表外的token（OOV问题）；3、对于相似的词语失去相似度信息（比如dog和dogs在词表中是两个完全不同的词）

​- 字符级别的分词：1、文本转为 id 序列之后长度非常长；2、字母本身没有内在含义，从单一的字母中很难学习到有意的词表示。

子词切分方法遵从两个原则：

1. 常用的词语不应该被切分为更小的片段

2. 不常用的词语应该被分解为有意义的子词

> ​ 举个例子，tokenization这个词，可以被分解为token和ization，token和ization作为单独的子词出现的概率更高（比如token、tokens、tokenizing、tokenization，或者tokenization、modernization等等），同时tokenization的意思被作为token和ization的复合意思被保存下来。

#### BPE (byte pair encoding)

- 需要基础词表与预分词, 如基于空格(GPT-2, RoBERTa), 如基于规则(XLM和FlauBERT 采用Mosed, GPT采用Spacy)。

- 需要终止符区分单词边界

- 每次挑选频率最高的组合进行合并

- 对于多国语言模型，采用unicode基础词表将会很大，这时可采用BBPE，即Byte-Level BPE，使用大小固定为256的基础词表。

- 应用模型示例：GPT-2, RoBERTa

#### WordPiece

- 按概率变化进行组合，选择 P(x_y)/P(x)/P(y) 最大的。

- 应用模型示例：Bert

#### Unigram

- Unigram会建立Unigram language model(ULM), 它可以输出带概率的多个分词序列。

- 初始化一个巨大词表，根据维特比算法估计每个词的概率

- 计算删除每个词带来的整体损失上升，按损失上升值排序，删除词表中损失上升最小的一部分。

#### SentencePiece

由于上述分词算法建立在空格分隔的基础上，因此会出现编码解码时的不一致。一种方法是通过规则实现。另一种方案是给不同语言预设不同的编码与解码算法。

> Raw text: Hello world.

> Tokenized: [Hello] [world] [.]

> Decoded text: Hello world .

## special token

特殊token 往往不对应具体的字符。

### 常用token

- bos: begin of sentence

- eos: end of sentence

- pad

- unk: unknown

### 常用模型与token

| model | bos | eos | pad | unk | other | tokenizer | size |
| - | - | - | - | - | - | - | - |
| baichuan-7b | <s> | </s> | - | <unk> | - | SentencePiece |
| baichuan-13b-chat | <s> | </s> | <unk> | <unk> | - | SentencePiece |
| baichuan2-7b-chat | <s> | </s> | <unk> | <unk> | - | SentencePiece |
| bert-base-chinese | [CLS] | [SEP] | [PAD] | [UNK] | [MASK] | WordPiece | 21,128|
| chatglm-6b | <sop>, [CLS] | <eop>, </s>, [SEP] | <pad> | <unk> | [gMASK], [sMASK], [MASK], <ENC>, <dBLOCK>, <unused_0>, <|blank_{length}|>, <|tab|>, <n>, <image_{num_image_tokens}> | SentencePiece |
| chatglm2-6b | <bos> | <eos> | <pad> | <unk> | [gMASK], <sop> | SentencePiece |
| flan-t5-large | - | </s> | <pad> | <unk> | <extra_id_{}> | - |
| gpt2 | - | <|endoftext|> |
| llama2-chinese-13b-chat | <s> | </s> | - | <unk> | - | - |
