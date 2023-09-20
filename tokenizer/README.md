# Tokenizer

## special token

特殊token 往往不对应具体的字符。

### 常用token

- bos: begin of sentence

- eos: end of sentence

- pad

### 常用模型与token

| model | bos | eos | pad | unk | other | tokenizer |
| - | - | - | - | - | - | - |
| baichuan-7b | <s> | </s> | - | <unk> | - | SentencePiece |
| baichuan-13b-chat | <s> | </s> | <unk> | <unk> | - | SentencePiece |
| baichuan2-7b-chat | <s> | </s> | <unk> | <unk> | - | SentencePiece |
| bert-base-chinese | [CLS] | [SEP] | [PAD] | [UNK] | [MASK] | WordPiece |
| chatglm-6b | <sop>, [CLS] | <eop>, </s>, [SEP] | <pad> | <unk> | [gMASK], [sMASK], [MASK], <ENC>, <dBLOCK>, <unused_0>, <|blank_{length}|>, <|tab|>, <n>, <image_{num_image_tokens}> | SentencePiece |
| chatglm2-6b | <bos> | <eos> | <pad> | <unk> | [gMASK], <sop> | SentencePiece |
| flan-t5-large | - | </s> | <pad> | <unk> | <extra_id_{}> | - |
| gpt2 | - | <|endoftext|> |
| llama2-chinese-13b-chat | <s> | </s> | - | <unk> | - | - |
