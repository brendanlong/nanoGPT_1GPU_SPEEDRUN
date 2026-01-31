"""
RoPE ablation study: compare different (base, truncation_fraction) configurations.

Runs short training (500 steps) for each config and reports validation loss.
"""
import os
import sys
import glob
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

dynamo.config.recompile_limit = 128

# Compile these once
flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
create_block_mask_compiled = torch.compile(create_block_mask, dynamic=False)

# -----------------------------------------------------------------------------
# Muon Optimizer
# -----------------------------------------------------------------------------
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, backend_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, backend_steps=backend_steps))
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.grad)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(p.grad)
                g = p.grad.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.data.add_(g, alpha=-group['lr'])

# -----------------------------------------------------------------------------
# Model with configurable RoPE
# -----------------------------------------------------------------------------
def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
    def reset_parameters(self):
        with torch.no_grad(): self.weight.zero_()
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class ConfigurableRotary(nn.Module):
    """RoPE with configurable base and truncation fraction."""
    def __init__(self, dim, base=10000, trunc_frac=0.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.trunc_frac = trunc_frac
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

        # Compute angular frequencies based on config
        n_active = int(dim * (1 - trunc_frac) / 2)
        if n_active > 0:
            # Log-spaced frequencies from 1 to 1/base
            t = torch.linspace(0, 1, n_active)
            angular_freq = (1.0 / base) ** t
            # Repeat each frequency for the sin/cos pair
            angular_freq = angular_freq.repeat_interleave(2)
            # Pad with zeros for truncated dimensions
            n_zeros = dim - len(angular_freq)
            if n_zeros > 0:
                angular_freq = torch.cat([angular_freq, torch.zeros(n_zeros)])
        else:
            angular_freq = torch.zeros(dim)

        self.register_buffer('angular_freq', angular_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).float()
            freqs = torch.outer(t, self.angular_freq.to(x.device))
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()

        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        d = x.shape[3] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos[..., :d] - x2 * sin[..., :d],
                          x1 * sin[..., :d] + x2 * cos[..., :d]], -1).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = 1
        self.head_dim = config.n_embd // config.n_head
        self.c_q = CastedLinear(config.n_embd, config.n_embd)
        self.c_k = CastedLinear(config.n_embd, self.n_kv_head * self.head_dim)
        self.c_v = CastedLinear(config.n_embd, self.n_kv_head * self.head_dim)
        self.val_proj = CastedLinear(config.n_embd, self.n_kv_head * self.head_dim)
        with torch.no_grad():
            std = 0.5 * (config.n_embd ** -0.5)
            bound = (3 ** 0.5) * std
            self.c_q.weight.uniform_(-bound, bound)
            self.c_k.weight.uniform_(-bound, bound)
            self.c_v.weight.uniform_(-bound, bound)
            self.val_proj.weight.uniform_(-bound, bound)
        self.lamb = nn.Parameter(torch.tensor(0.5))
        self.rotary = rotary
        self.c_proj = CastedLinear(config.n_embd, config.n_embd)
        self.attn_gate = CastedLinear(12, config.n_head)
        self.flex_kernel_options = config.flex_kernel_options

    def forward(self, x, v1, block_mask, attn_scale=0.1):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if v1 is None: v1 = v
        else:
            if v1.size(-1) != self.head_dim:
                v1 = self.val_proj(v1).view(B, T, self.n_kv_head, self.head_dim)
        v = (1 - self.lamb) * v + self.lamb * v1
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention_compiled(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                           block_mask=block_mask, scale=attn_scale,
                           kernel_options=self.flex_kernel_options, enable_gqa=True)
        y = y.transpose(1, 2).contiguous()
        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).unsqueeze(-1)
        y = y * gate
        y = y.view(B, T, -1)
        return self.c_proj(y), v1

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        with torch.no_grad():
            self.c_fc.weight.uniform_(-(3**0.5)*(0.5*dim**-0.5), (3**0.5)*(0.5*dim**-0.5))
    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())

class Block(nn.Module):
    def __init__(self, config, layer_idx, rotary):
        super().__init__()
        self.attn = CausalSelfAttention(config, rotary) if layer_idx not in [0, 7] else None
        self.mlp = MLP(config.n_embd) if layer_idx != 0 else None
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
    def forward(self, x, v1, x0, block_mask, attn_scale):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x1, v1 = self.attn(norm(x), v1, block_mask, attn_scale)
            x = x + x1
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x, v1

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    flex_kernel_options: dict = None
    rope_base: float = 10000
    rope_trunc_frac: float = 0.0

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        # Create shared rotary embedding with config
        head_dim = config.n_embd // config.n_head
        self.rotary = ConfigurableRotary(head_dim, config.rope_base, config.rope_trunc_frac)

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, i, self.rotary) for i in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.smear_gate = CastedLinear(12, 1)
        self.value_embeds = nn.ModuleList([nn.Embedding(config.vocab_size, config.n_embd) for _ in range(3)])
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, idx, target, attn_blocksize):
        docs = (idx == 50256).cumsum(0)
        def document_causal_mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx]) & (q_idx - kv_idx < attn_blocksize)
        S = len(idx)
        block_mask = create_block_mask_compiled(document_causal_mask, None, None, S, S, device="cuda", _compile=True)
        x = self.transformer.wte(idx[None])
        smear_gate_out = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)
        x = norm(x); x0 = x; v1 = None
        ve = [value_embed(idx) for value_embed in self.value_embeds]
        ve_list = [None, ve[1], ve[2]] + [None] * (len(self.transformer.h) - 6) + [ve[0], ve[1], ve[2]]
        skip_connections = []; x_backout = None; backout_layer = 8
        for i in range(self.num_encoder_layers):
            if ve_list[i] is not None and v1 is None: v1 = ve_list[i][None]
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask, 0.1)
            skip_connections.append(x)
            if i == backout_layer: x_backout = x
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            x = x + self.skip_weights[i] * skip_connections.pop()
            if ve_list[layer_idx] is not None and v1 is None: v1 = ve_list[layer_idx][None]
            x, v1 = self.transformer.h[layer_idx](x, v1, x0, block_mask, 0.1)
            if layer_idx == backout_layer: x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda * x_backout
        logits = 30 * torch.tanh(self.lm_head(norm(x)) / 30)
        return F.cross_entropy(logits.view(-1, logits.size(-1)).bfloat16(), target.view(-1))

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def _load_data_shard(filename):
    with open(filename, "rb") as f:
        return torch.frombuffer(f.read(), dtype=torch.uint16)[256*2:]

class SimpleDataLoader:
    def __init__(self, filename_pattern, T):
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"No files found: {filename_pattern}"
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[0])

    def next_batch(self):
        if self.current_position + self.T + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.files)
            self.tokens = _load_data_shard(self.files[self.current_shard])
            self.current_position = 0
        buf = self.tokens[self.current_position:self.current_position+self.T+1]
        self.current_position += self.T
        data = torch.tensor(buf.numpy().astype('int32'), dtype=torch.long).cuda()
        return data[:-1], data[1:]

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[0])

# -----------------------------------------------------------------------------
# Run single experiment
# -----------------------------------------------------------------------------
def run_experiment(rope_base, rope_trunc_frac, num_steps=500, batch_size=8, seq_len=4096):
    """Run a short training and return final validation loss."""

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    config = GPTConfig(
        flex_kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32},
        rope_base=rope_base,
        rope_trunc_frac=rope_trunc_frac
    )

    train_loader = SimpleDataLoader('finewebedu10B/finewebedu_train_*.bin', seq_len)
    val_loader = SimpleDataLoader('finewebedu10B/finewebedu_val_*.bin', seq_len)

    model = GPT(config).bfloat16().cuda()
    model = torch.compile(model)

    # Optimizers
    param_dict = {pn: p for pn, p in model.named_parameters()}
    matrix_params = [p for p in param_dict.values() if p.ndim == 2]
    scalar_params = [p for p in param_dict.values() if p.ndim < 2]
    opt_matrix = Muon(matrix_params, lr=0.05, momentum=0.95)
    opt_scalar = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True)
    opt_wte = torch.optim.Adam([model.transformer.wte.weight] + [v.weight for v in model.value_embeds],
                               lr=0.6, betas=(0.8, 0.95), fused=True)
    opt_head = torch.optim.Adam([model.lm_head.weight], lr=0.008, betas=(0.8, 0.95), fused=True)
    optimizers = [opt_wte, opt_head, opt_matrix, opt_scalar]

    # Training loop
    t0 = time.time()
    for step in range(num_steps):
        window = 256  # Fixed window for fair comparison
        attn_blocksize = torch.tensor(window, dtype=torch.int, device='cuda')

        model.train()
        for i in range(batch_size):
            x, y = train_loader.next_batch()
            loss = model(x, y, attn_blocksize)
            loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad /= batch_size

        frac = min(step / 300, 1)
        opt_matrix.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        if step % 100 == 0:
            torch.cuda.synchronize()
            print(f"  Step {step}/{num_steps}, loss={loss.item():.3f}")

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(16):
            vx, vy = val_loader.next_batch()
            attn_bs = torch.tensor(256, dtype=torch.int, device='cuda')
            vloss = model(vx, vy, attn_bs)
            val_losses.append(vloss.item())

    val_loss = sum(val_losses) / len(val_losses)
    elapsed = time.time() - t0

    return val_loss, elapsed

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("RoPE Ablation Study")
    print("=" * 60)

    # Configurations to test
    # Format: (base, trunc_frac, description)
    # Testing lower truncation values since 0.40 < 0.50 < 0.60
    configs = [
        (570, 0.00, "trunc=0.00"),
        (570, 0.20, "trunc=0.20"),
        (570, 0.30, "trunc=0.30"),
        (570, 0.40, "trunc=0.40"),
    ]

    results = []

    for base, trunc, desc in configs:
        print(f"\n--- Testing: {desc} (base={base}, trunc={trunc}) ---")
        try:
            val_loss, elapsed = run_experiment(base, trunc, num_steps=500, batch_size=4, seq_len=4096)
            results.append((desc, base, trunc, val_loss, elapsed))
            print(f"  Val loss: {val_loss:.4f}, Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((desc, base, trunc, float('inf'), 0))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Config':<30} {'Base':>6} {'Trunc':>6} {'ValLoss':>8}")
    print("-" * 60)
    for desc, base, trunc, val_loss, elapsed in sorted(results, key=lambda x: x[3]):
        print(f"{desc:<30} {base:>6} {trunc:>6.2f} {val_loss:>8.4f}")
