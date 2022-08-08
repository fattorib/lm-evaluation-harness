import transformers
import torch
from lm_eval.base import BaseLM
import sys
from transformers import GPT2TokenizerFast

from lm_eval.models.fastergpt_base import model_getter



class BFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        model_size = 'XL*',
        model_weights_path = ''
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

        print(model_size)
        if model_size == 'base':
            self.gpt = model_getter(
                model_size,
                vocab_size=50257,
                num_ctx=1024,
                **{"use_alibi": False},
            )
            state_dict = torch.load(
                model_weights_path,
                map_location="cpu",
                )

            self.gpt.load_state_dict(state_dict)

            del state_dict
                
        elif model_size == 'XL*':
            self.gpt = model_getter(
                model_size,
                vocab_size=50257,
                num_ctx=512,
                **{"fused_residuals": True, "num_head": 8, "use_alibi": True},
            )

            state_dict = torch.load(
                model_weights_path,
                map_location="cpu",
                )

            self.gpt.load_state_dict(state_dict['state_dict'])

            del state_dict

            PRIME_CTX = 1024
            # prime with ctx of 1024:
            with torch.no_grad():
                data_batch = torch.randint(low=0, high=50257, size=(1, PRIME_CTX))
                self.gpt(data_batch)
                print(f'Evaluating ALiBi model with context: {PRIME_CTX}')
        
        
        self.gpt.to(self.device)        
        self.gpt.eval()

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
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

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return 1024
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return 1024

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
            return self.gpt(inps)[:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.gpt.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, sample=False
        )

