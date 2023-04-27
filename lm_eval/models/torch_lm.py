import transformers
import torch
from lm_eval.base import BaseLM
from transformers import GPTNeoXTokenizerFast
from lm_eval.models.gpt_pytorch import model_getter



class GPTCustom(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        model_size = None,
        model_weights_path = None,
        eval_ctx = None
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        print(f"Model Size: {model_size}")
        print(f"Model Weights Path: {model_weights_path}")
        print(f"Evaluation Context: {eval_ctx}")

        self.gpt = model_getter(
            model_size,
            vocab_size=50304,
            num_ctx=1024 if 'distill' not in model_size else 2048
        )
        state_dict = torch.load(
            model_weights_path,
            map_location="cpu",
            )

        self.gpt.load_state_dict(state_dict)

        self.ctx = eval_ctx

        self.gpt.half()
        torch.cuda.empty_cache()

        if self.gpt.head_qk_trick:
            del self.gpt.copy_mask
            self.gpt.register_buffer("copy_mask", torch.tril(
                torch.ones(self.ctx, self.ctx)))

        del state_dict
                
        self.gpt.to(self.device)        
        self.gpt.eval()

        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.GPTNeoXTokenizerFast
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ):
            assert self.tokenizer.encode("hello\n\nhello") == [
                31373,
                198,
                198,
                31373,
            ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size


    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):  
        return self.ctx


    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt(inps)[:, :, :50304]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.gpt.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, sample=False
        )

