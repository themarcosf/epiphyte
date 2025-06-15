import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

### Implementation
@dataclass
class GPTConfig:
    context_lengh: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # batched K, Q, V projections
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization params
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch
                                          .ones(config.context_lengh, config.context_lengh))
                                          .view(1, 1, config.context_lengh, config.context_lengh))

    def forward(self, inputs):
        # B: batch size, T: time dimension (sequence length), C: channel dimensions (embedding size)
        B, T, C = inputs.size()

        # calculate K, Q, V
        qkv = self.c_attn(inputs)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_heads, T, h_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_heads, T, h_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_heads, T, h_size)

        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = att @ v # (B, n_heads, T, T) x (B, n_heads, T, h_size) -> (B, n_heads, T, h_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        out = self.c_proj(out)
        return out

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, inputs):
        inputs = self.c_fc(inputs)
        inputs = self.gelu(inputs)
        inputs = self.c_proj(inputs)
        return inputs

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FFN(config)

    def forward(self, inputs):
        inputs = inputs + self.attn(self.ln_1(inputs)) # map operation
        inputs = inputs + self.mlp(self.ln_2(inputs))  # reduce operation
        return inputs

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.context_lengh, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, inputs):
        B, T = inputs.size()
        assert T <= self.config.context_lengh, f'Cannot forward, model context length is exhausted. ' \
                                               f'Input has {T} tokens, but the maximum is {self.config.context_lengh}'
        
        # forward input embedding and positional embedding
        positions = torch.arange(0, T, dtype=torch.long, device=inputs.device)
        position_embeddings = self.transformer.wpe(positions)
        token_embeddings = self.transformer.wte(inputs)
        inputs = token_embeddings + position_embeddings
        
        # forward the transformer blocks
        for block in self.transformer.h:
            inputs = block(inputs)

        # forward the final layer norm and linear layer
        inputs = self.transformer.ln_f(inputs)
        logits = self.lm_head(inputs)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT2 model weights from huggingface.co 🤗"""

        assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        print(f'Loading weights from pretrained gpt: {model_type}')

        # create model config
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['context_lengh'] = 1024

        # create model with random weights
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # ignore buffer keys of autoregressive masks

        # create model with pretrained weights
        from transformers import GPT2LMHeadModel
        pretrained_model = GPT2LMHeadModel.from_pretrained(model_type)
        pretrained_sd = pretrained_model.state_dict()

        # copy pretrained weights into randomly initialized model
        pretrained_sd_keys = pretrained_sd.keys()
        pretrained_sd_keys = [k for k in pretrained_sd_keys if not k.endswith('.attn.masked_bias')]
        pretrained_sd_keys = [k for k in pretrained_sd_keys if not k.endswith('.attn.bias')]

        #  OpenAI checkpoints use `Conv1D` modules from TensorFlow that needs to be transposed to match PyTorch
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys) == len(pretrained_sd_keys), f'Mismatched keys: {len(sd_keys)} vs {len(pretrained_sd_keys)}'

        for k in pretrained_sd_keys:
            if any(k.endswith(t) for t in transposed):
                # special treatment for `Conv1D` weights
                assert pretrained_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(pretrained_sd[k].T)
            else:
                # vanilla copy of weights
                assert pretrained_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(pretrained_sd[k])

        return model


# Example usage
if __name__ == '__main__':
    # Set the device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Using device: {device}')

    # Load the tokenizer
    num_return_sequences = 5

    import tiktoken
    encodings = tiktoken.get_encoding('gpt2')
    tokens = encodings.encode('Once upon a time')
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    print('Tokens loaded')

    # Load the model
    model = GPT.from_pretrained('gpt2')
    model.eval()
    model.to(device)
    print('Model loaded')

    # Generate predictions
    max_length = 50

    while x.size(1) < max_length:
        with torch.no_grad():
            # forward pass
            logits = model(x)
            # only the logit at the last location is needed
            # in effect, all other logits are basically thrown away
            logits = logits[:, -1, :]
            # probabilities of the next token
            probs = F.softmax(logits, dim=-1)
            # top-k sampling of 50 tokens (huggingface default)
            top_k_probs, top_k_indices = probs.topk(k=50, dim=-1)
            # sample from the top-k tokens
            top_k_token = torch.multinomial(top_k_probs, num_samples=1)
            # gather the top-k token index
            top_k_index = torch.gather(top_k_indices, -1, top_k_token)
            # append the sampled token to the input
            x = torch.cat((x, top_k_index), dim=1)

    # decode the predicted tokens
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        text = encodings.decode(tokens)
        print('>', text)