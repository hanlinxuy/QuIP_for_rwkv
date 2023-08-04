import torch
from .rwkv_tokenizer import TRIE_TOKENIZER
from types import SimpleNamespace


class FakeRWKVTokenier(TRIE_TOKENIZER):
    def __call__(self, src, return_tensors="pt"):
        list_out = self.encode(src)
        out_sn = SimpleNamespace()
        out_sn.input_ids = torch.Tensor([list_out])
        out_sn.attention_mask = None
        return out_sn
