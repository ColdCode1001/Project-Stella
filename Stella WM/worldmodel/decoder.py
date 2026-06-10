"""
SmallDecoder -- ~38M params, autoregressive Transformer condicionado por estado RSSM.

Arquitectura:
  - 4 bloques Transformer con causal self-attention + cross-attention al estado del WM
  - d_model=512, n_heads=8, d_ff=1024
  - GPT-2 tokenizer (vocab_size=50257), embeddings tied con lm_head
  - RSSM context: h_t [256] + z_t [64] = 320D → proyectado a D_MODEL

El decoder aprende a traducir el estado latente del WM a texto.
El RSSM NO recibe gradientes del decoder (language_grads=False).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 50257    # GPT-2 tokenizer
D_MODEL    = 512
N_HEADS    = 8
N_LAYERS   = 4
D_FF       = 1024
MAX_SEQ    = 256
CTX_DIM    = 320      # RSSM hidden_dim(256) + latent_dim(64)

DECODER_WEIGHTS = Path("worldmodel/weights/decoder.pt")


class CausalSelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.d_head  = D_MODEL // N_HEADS
        self.qkv  = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        mask = torch.tril(torch.ones(MAX_SEQ, MAX_SEQ)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale = 1.0 / math.sqrt(self.d_head)
        attn  = (q @ k.transpose(-2, -1)) * scale
        attn  = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn  = F.softmax(attn, dim=-1)
        out   = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class CrossAttn(nn.Module):
    """Cross-attention de los tokens del decoder al estado del RSSM."""
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.d_head  = D_MODEL // N_HEADS
        self.q   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.kv  = nn.Linear(D_MODEL, 2 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]   ctx: [B, L, D]  (L=1 para contexto unico)
        B, T, C = x.shape
        L = ctx.shape[1]
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.kv(ctx).split(C, dim=-1)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_head)
        attn  = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        out   = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1         = nn.LayerNorm(D_MODEL)
        self.self_attn  = CausalSelfAttn()
        self.n2         = nn.LayerNorm(D_MODEL)
        self.cross_attn = CrossAttn()
        self.n3         = nn.LayerNorm(D_MODEL)
        self.ff         = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.n1(x))
        x = x + self.cross_attn(self.n2(x), ctx)
        x = x + self.ff(self.n3(x))
        return x


class SmallDecoder(nn.Module):
    """
    Decoder autoregresivo condicionado por el estado del RSSM.
    ~38M parametros (con tied embeddings).
    """

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ, D_MODEL)
        self.ctx_proj = nn.Linear(CTX_DIM, D_MODEL)       # RSSM state → D_MODEL
        self.blocks   = nn.ModuleList([DecoderBlock() for _ in range(N_LAYERS)])
        self.norm     = nn.LayerNorm(D_MODEL)
        self.lm_head  = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # tied embeddings
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, ctx_vec: torch.Tensor) -> torch.Tensor:
        """
        tokens:  [B, T]  long
        ctx_vec: [B, CTX_DIM]  -- RSSM h+z concatenados
        Returns: logits [B, T, VOCAB_SIZE]
        """
        B, T = tokens.shape
        device = tokens.device
        pos = torch.arange(T, device=device)
        x   = self.tok_emb(tokens) + self.pos_emb(pos)             # [B, T, D]
        ctx = self.ctx_proj(ctx_vec).unsqueeze(1)                   # [B, 1, D]
        for block in self.blocks:
            x = block(x, ctx)
        return self.lm_head(self.norm(x))                           # [B, T, V]

    @torch.no_grad()
    def generate(
        self,
        ctx_vec: torch.Tensor,
        max_new: int = 80,
        temperature: float = 0.85,
        top_k: int = 50,
        eos_id: int = 50256,
    ) -> list[int]:
        """Generacion autoregresiva condicionada en el estado del RSSM."""
        self.eval()
        device = ctx_vec.device
        tokens = torch.tensor([[eos_id]], device=device)   # BOS = EOS en GPT-2
        for _ in range(max_new):
            if tokens.shape[1] >= MAX_SEQ:
                break
            logits = self.forward(tokens, ctx_vec)[:, -1, :]   # [1, V]
            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                topvals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topvals[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            if next_tok.item() == eos_id and tokens.shape[1] > 4:
                break
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens[0, 1:].tolist()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        return self

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_or_create(path: Path = DECODER_WEIGHTS) -> SmallDecoder | None:
    """Carga el decoder si existe, None si no hay pesos entrenados."""
    model = SmallDecoder()
    if path.exists():
        try:
            model.load(path)
            print(f"[decoder] Pesos cargados desde {path} ({model.param_count()/1e6:.1f}M params)")
            return model
        except Exception as e:
            print(f"[decoder] Error cargando pesos: {e}")
            return None
    return None
