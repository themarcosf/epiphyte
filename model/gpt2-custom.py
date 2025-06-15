import inspect
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from hellaswag import render_example, iterate_examples, get_seed_encodings, decode_seed_encodings

################################################################################
### Implementation
################################################################################

### Data Loader ################################################################
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ('train', 'val')

        data_root = 'edu_fineweb10B_shards'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f'No shards found for split {split} in {data_root}'
        if master_process:
            print(f'Found {len(shards)} shards for split {split}')

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.state = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # buffer = self.tokens.clone().detach()
        # buffer = self.tokens[self.state:self.state + B * T + 1].clone().detach()
        buffer = self.tokens[self.state:self.state + B * T + 1]

        x = (buffer[:-1]).view(B, T)  # input tokens
        y = (buffer[1:]).view(B, T)   # target tokens

        self.state += B * T * self.num_processes

        if self.state + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.state = B * T * self.process_rank
        return x, y



### Neural network #############################################################
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
        self.c_proj.SCALE_INIT = 1.0

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

        # flash attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # out = att @ v # (B, n_heads, T, T) x (B, n_heads, T, h_size) -> (B, n_heads, T, h_size)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        self.c_proj.SCALE_INIT = 1.0

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

        # parameter sharing scheme (check #3.2 in the gpt2.ipynb)
        self.transformer.wte.weight = self.lm_head.weight

        # initialize weights (check #3.3 in the gpt2.ipynb)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_ratio, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {k: v for k, v in param_dict.items() if v.requires_grad}

        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        no_decay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f'Number of decayed parameters tersors: {len(decay_params)}, with {num_decay_params:,} total elements')
        print(f'Number of non-decayed parameters tersors: {len(no_decay_params)}, with {num_no_decay_params:,} total elements')

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        print(f'Using fused AdamW: {use_fused}')

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_ratio, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def forward(self, inputs, targets=None):
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

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

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


### Example usage ##############################################################

# Simple launch: python gpt2-custom.py
# DDP launch: torchrun --standalone --nproc_per_node=<NUM_GPU> gpt2-custom.py
# GPU dashboard: watch -n 0.1 nvidia-smi
if __name__ == '__main__':

    # helper function for HellaSwag eval
    def get_most_likely_row(tokens, mask, logits):
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask
        
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        
        pred_norm = avg_loss.argmin().item()
        return pred_norm

    # DDP initialization
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), 'DDP requires CUDA'
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ.get('RANK'))
        ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
        ddp_world_size = int(os.environ.get('WORLD_SIZE'))
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        device = 'cpu'

        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'

        print(f'Using device: {device}, DDP enabled: {ddp}')

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Create data loader with gradient accumulation
    total_batch_size = 524_288     # 2**19, ~ 500k tokens
    B = 8                          # micro batch size
    T = 1024                       # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, f'Total batch size {total_batch_size} must be divisible by B * T * ddp_world_size = {B * T * ddp_world_size}'
    gradient_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f'Training with total batch size: {total_batch_size} tokens')
        print(f'Batch size: {B}, Sequence length: {T}, Gradient accumulation steps: {gradient_accum_steps}')

    train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
    print('Data loader created')

    # set internal precision for matrix multiplication
    torch.set_float32_matmul_precision('high')

    # Create new model
    model = GPT(GPTConfig(vocab_size=50257))
    model.to(device)

    use_compile = True
    if use_compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    print('Model loaded')

    # Optimizer
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715     # 375e6 / 2**19 [GPT3 paper]
    max_steps = 19_073     # 10e9 / 2**19

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1, f'Invalid decay ratio: {decay_ratio}'
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_ratio=max_lr, device=device)

    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log.txt')
    with open(log_file, 'w') as f:
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Evaluate loss
        if step % 2 == 0 or last_step:
            model.eval()
            val_loader.reset()

            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20

                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)

                    with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
                        _, loss = model(x, y)

                    loss /= val_loss_steps
                    val_loss_accum += loss.detach()

            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            if master_process:
                print(f'Step {step+1}: Validation loss = {val_loss_accum.item():.6f}')
                with open(log_file, 'a') as f:
                    f.write(f'Step {step+1}: Validation loss = {val_loss_accum.item():.6f}\n')

        # HellaSwag evaluation
        if step % 2 == 0 or last_step:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples('val')):
                if i % ddp_world_size != ddp_rank:
                    continue
                
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f'HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}')
                with open(log_file, 'a') as f:
                    f.write(f'{step} hella {acc_norm:.4f}\n')

        # Sample from the model
        if ((step > 0 and step % 2 == 0) or last_step):
            model.eval()

            num_return_sequences = 4 
            max_length = 32

            tokens = get_seed_encodings('Once upon a time')
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            x_gen = tokens.to(device)

            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)

            while x_gen.size(1) < max_length:
                with torch.no_grad():
                    # forward pass
                    with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
                        logits, _ = model(x_gen)
                    # only the logit at the last location is needed
                    logits = logits[:, -1, :]
                    # probabilities of the next token
                    probs = F.softmax(logits, dim=-1)
                    # top-k sampling of 50 tokens (huggingface default)
                    top_k_probs, top_k_indices = probs.topk(k=50, dim=-1)
                    # sample from the top-k tokens
                    top_k_token = torch.multinomial(top_k_probs, num_samples=1, generator=sample_rng)
                    # gather the top-k token index
                    top_k_index = torch.gather(top_k_indices, -1, top_k_token)
                    # append the sampled token to the input
                    x_gen = torch.cat((x_gen, top_k_index), dim=1)

            # decode the predicted tokens
            for i in range(num_return_sequences):
                tokens = x_gen[i, :max_length].tolist()
                text = decode_seed_encodings(tokens)
                print(f'Rank {ddp_rank}, sample {i+1} > {text}')


        # Training loop
        model.train()

        optimizer.zero_grad()

        loss_accum = 0.0

        for micro_step in range(gradient_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
                logits, loss = model(x, y)

            loss = loss / gradient_accum_steps
            loss_accum += loss.detach()

            if ddp:
                # naughty hack to avoid using no_sync context manager
                # may break in future PyTorch versions
                model.require_backward_grad_sync = (micro_step == gradient_accum_steps - 1)

            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if device == 'cuda':
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tp = (train_loader.B * train_loader.T * gradient_accum_steps * ddp_world_size) / dt
        if master_process:
            print(f'Step {step+1}: Loss = {loss_accum.item():.6f}, Learning rate = {lr:.4e}, Time = {dt:.2f} s, Norm: {norm:.4f}, Throughput = {tp:.2f} tokens/s')

            if step % 1 == 0 or last_step:
                ckpt = {
                    'step': step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(ckpt, 'checkpoint_step.pth')
                print(f'Checkpoint saved: checkpoint_step{step}.pth')

        with open(log_file, 'a') as f:
            f.write(f'Step {step+1}: Loss = {loss_accum.item():.6f}, Learning rate = {lr:.4e}, Time = {dt:.2f} s, Norm: {norm:.4f}, Throughput = {tp:.2f} tokens/s\n')

    if ddp:
        destroy_process_group()
        print('Destroyed DDP process group')
