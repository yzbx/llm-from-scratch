from functools import partial

import lightning as L
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel, LlamaTokenizer, LlamaTokenizerFast


class LightningModuleWrapper(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self.model(batch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

def encode(tokenizer, examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

if __name__ == '__name__':
    cfg  = LlamaConfig(num_hidden_layers=4, num_attention_heads=4)
    gpt = LlamaForCausalLM(cfg)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    dataset = load_dataset("json", data_files="/data/wangjiaxin/zh_wiki_v0.3.jsonl")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

    dataset = dataset.map(partial(encode, tokenizer=tokenizer), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    trainer.fit(model=gpt, train_dataloaders=train_loader)



