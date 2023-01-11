import torch 
from typing import List 
from transformers import AutoTokenizer

class ByteTokenizer:

    def __init__(self) -> None:
        self.eos_token_id = 256 
        self.vocab_size = 257

        self.sub_tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    def encode(self,text: str) -> bytes:
        # encode a string of text as a bytearray 
        # call byt5 tokenizer then shift outputs 
        encoded = torch.tensor(self.sub_tokenizer.encode(text)) - 3
        return  list(encoded[:-1])

    def decode(self, tokens: List[int]) -> str:
        # decode a list of bytes to string 
        return self.sub_tokenizer.decode(tokens + 3)


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    test = 'hey this is a test sentence!'

    out_tok = tokenizer.encode(test)


    enc = ByteTokenizer().encode(test) + 3
    print(enc) 
    print(out_tok)
    print(tokenizer.decode(enc))