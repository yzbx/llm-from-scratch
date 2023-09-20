# Preprocess

数据预处理, 参考 https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py, 将jsonline文件转为idx与bin文件。

## 数据格式

建议保留id, type, src. 其中id可以检查过滤百分比与过滤程序. type可用于抽样， src可用于溯源， 而text用于训练。

```
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

- 对于字典格式的数据， 可以将字典转为人类习惯的格式， 或者是原始字典格式，例如：

```
# 人类习惯的格式一
标题：静夜思
作者：李白
正文：床前明月，疑是地上霜。
举头望明月，低头思故乡。

# 人类习惯的格式二
《赋得古原草送别》（白居易） 离离原上草，一岁一枯荣。 野火烧不尽，春风吹又生。

# 原始字典格式
'{"title": "静夜思", "author": "李白", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"}'

# 原始字典格式二
'{
        "title": "赋得古原草送别",
        "author": "白居易",
        "text": "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。"
}'
```
