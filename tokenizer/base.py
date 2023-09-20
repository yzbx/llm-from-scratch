from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from typing import List


# the index of a token in the vocab represents the integer id for that token
# i.e. the integer id for "heroes" would be 2, since vocab[2] = "heroes"
vocab = ["all", "not", "heroes", "the", "wear", ".", "capes", "<unk>"]

# a pretend tokenizer that tokenizes on whitespace
tokenizer = Tokenizer(WordLevel(vocab={k:idx for idx,k in enumerate(vocab)}, unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# the encode() method converts a str -> list[int]
encoding = tokenizer.encode("not all heroes wear", is_pretokenized=False)
ids: List[int] = encoding.ids # ids = [1, 0, 2, 4]
print(ids)

encoding = tokenizer.encode(["not", "all", "heroes", "wear"], is_pretokenized=True)
ids: List[int] = encoding.ids # ids = [1, 0, 2, 4]
print(ids)

# we can see what the actual tokens are via our vocab mapping
tokens: List[str] = encoding.tokens # tokens = ["not", "all", "heroes", "wear"]
print(tokens)

tokens = [vocab[i] for i in ids]
print(tokens)

# the decode() method converts back a list[int] -> str
text = tokenizer.decode(ids) # text = "not all heroes wear"
print(text)
