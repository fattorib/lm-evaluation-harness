import torch 
from typing import List 

class ByteTokenizer:

    def __init__(self) -> None:
        self.eos_token_id = 256 
        self.vocab_size = 257

    def encode(self,text: str) -> bytes:
        # encode a string of text as a bytearray 
        return torch.tensor([list(text.encode("utf-8"))])

    def decode(self, tokens: List[int]) -> str:
        # decode a list of bytes to string 
        return bytearray(tokens[0]).decode('utf-8')