"""
Pre-entrenamiento del RSSM con datos de Stella + conocimiento del mundo real.

Fuentes de datos:
  - stella.chats.jsonl    → conversaciones reales de Stella con Arca
  - stella.thoughts.jsonl → pensamientos idle de Stella
  - stella.episodic       → episodios de memoria
  - stella.research.json  → steps de quests
  - Wikipedia ES          → conocimiento del mundo real (streaming)

El RSSM aprende a predecir el estado siguiente dado el actual:
  embedding(texto_t) + acción → predice embedding(texto_{t+1})

Uso:
  python -m worldmodel.pretrain
  python -m worldmodel.pretrain --epochs 20 --lr 3e-4 --wiki-articles 500
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent.parent))
from worldmodel.rssm import MinimalRSSM, load_or_create
from worldmodel.obs_encoder import encode_batch_gpu, OBS_DIM

STELLA_STORE = Path("D:/stella/memory/store")
WEIGHTS_PATH = Path("worldmodel/weights/rssm.pt")


# ─── Carga de datos de Stella ──────────────────────────────────────────────────

def load_chat_sequences() -> list[list[str]]:
    """Carga conversaciones como secuencias [user, stella, user, stella...]"""
    path = STELLA_STORE / "stella.chats.jsonl"
    if not path.exists():
        return []
    msgs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            msgs.append(json.loads(line))
        except Exception:
            pass

    sequences, current, prev_ts = [], [], None
    for m in msgs:
        ts_str = m.get("ts", "")
        content = m.get("content", "").strip()
        if not content:
            continue
        if prev_ts and ts_str:
            try:
                from datetime import datetime
                t1 = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if (t2 - t1).total_seconds() > 3600 and current:
                    sequences.append(current)
                    current = []
            except Exception:
                pass
        current.append(content)
        prev_ts = ts_str

    if current:
        sequences.append(current)
    return sequences


def load_thoughts() -> list[str]:
    path = STELLA_STORE / "stella.thoughts.jsonl"
    if not path.exists():
        return []
    thoughts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            t = json.loads(line)
            content = t.get("content", "").strip()
            if content and len(content) > 20:
                thoughts.append(content)
        except Exception:
            pass
    return thoughts


def load_episodes() -> list[str]:
    path = STELLA_STORE / "stella.episodic"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            e.get("content", "").strip()
            for e in data.get("episodes", [])
            if e.get("content", "").strip()
        ]
    except Exception:
        return []


def load_research_steps() -> list[str]:
    path = STELLA_STORE / "stella.research.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        steps = []
        for task in data.get("tasks", []):
            for p in task.get("progress", []):
                content = p.get("content", "").strip()
                if content and len(content) > 20:
                    steps.append(content)
        return steps
    except Exception:
        return []


# ─── Conocimiento del mundo real ───────────────────────────────────────────────

def load_world_knowledge_sequences(n_articles: int = 300, window: int = 5) -> list[list[str]]:
    """
    Carga fragmentos de Wikipedia ES como secuencias.
    El RSSM aprende física y realidad del mundo desde texto humano curado.
    Esto NO es conocimiento de LLM — son textos del mundo real, como leer libros.
    """
    sequences = []
    try:
        from datasets import load_dataset
        print(f"  Cargando Wikipedia ES (streaming, hasta {n_articles} artículos)...")
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.es",
            split="train", streaming=True, trust_remote_code=False
        )
        count = 0
        for article in ds:
            if count >= n_articles:
                break
            text = article.get("text", "")
            if not text or len(text) < 500:
                continue

            # Dividir en oraciones por puntuación
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                     if len(s.strip()) > 40]
            if len(sents) < 3:
                continue

            # Ventanas deslizantes
            step = max(1, window // 2)
            for i in range(0, len(sents) - window + 1, step):
                w = sents[i:i + window]
                if len(w) >= 3:
                    sequences.append(w)

            count += 1
            if count % 100 == 0:
                print(f"  {count}/{n_articles} artículos → {len(sequences)} secuencias...")

        print(f"  Wikipedia ES: {count} artículos → {len(sequences)} secuencias")
    except Exception as e:
        print(f"  [!] Wikipedia ES no disponible: {e}")

    return sequences


# ─── Training ─────────────────────────────────────────────────────────────────

def _kl_divergence(
    mu1: torch.Tensor, log_std1: torch.Tensor,
    mu2: torch.Tensor, log_std2: torch.Tensor,
) -> torch.Tensor:
    """KL(N(mu1,std1) || N(mu2,std2))"""
    std1 = torch.exp(log_std1.clamp(-4, 2))
    std2 = torch.exp(log_std2.clamp(-4, 2))
    kl = (torch.log(std2 / (std1 + 1e-8)) +
          (std1 ** 2 + (mu1 - mu2) ** 2) / (2 * std2 ** 2 + 1e-8) - 0.5)
    return kl.mean()


def train_on_sequence(
    model: MinimalRSSM,
    obs_pred: nn.Module,
    optimizer: torch.optim.Optimizer,
    seq_embs: np.ndarray,
) -> float:
    """
    Entrena el RSSM en una secuencia de embeddings.
    Loss = KL(prior || posterior_next) + 0.5 * MSE(reconstrucción obs)
    """
    if len(seq_embs) < 2:
        return 0.0

    model.train()
    obs_pred.train()
    h = model.zero_state()
    total_loss = 0.0
    steps = 0

    for i in range(len(seq_embs) - 1):
        obs_t  = torch.from_numpy(seq_embs[i]).float().unsqueeze(0)      # [1, OBS_DIM]
        obs_t1 = torch.from_numpy(seq_embs[i + 1]).float().unsqueeze(0)  # [1, OBS_DIM]

        action_idx = 0 if i % 2 == 0 else 6
        act = F.one_hot(torch.tensor([action_idx]), model.n_actions).float()

        gru_in = torch.cat([obs_t, act], dim=-1)
        h = model.gru(gru_in, h)

        # Posterior con obs_t
        post_in = torch.cat([h, obs_t], dim=-1)
        z_mean    = model.post_mean(post_in)
        z_log_std = model.post_log_std(post_in)
        z = z_mean + torch.exp(z_log_std.clamp(-4, 2)) * torch.randn_like(z_mean)

        # Prior desde h (lo que el RSSM predice sin ver la siguiente obs)
        prior_mean    = model.prior_mean(h)
        prior_log_std = model.prior_log_std(h)

        # Posterior del siguiente step (objetivo del prior)
        post_in_next   = torch.cat([h, obs_t1], dim=-1)
        z_next_mean    = model.post_mean(post_in_next)
        z_next_log_std = model.post_log_std(post_in_next)

        # Loss 1: el prior aprende a predecir el siguiente estado latente
        kl_loss = _kl_divergence(
            prior_mean, prior_log_std,
            z_next_mean.detach(), z_next_log_std.detach()
        )

        # Loss 2: el feat debe preservar info suficiente para reconstruir obs_t
        feat = torch.cat([h, z], dim=-1)
        recon_loss = F.mse_loss(obs_pred(feat), obs_t)

        loss = kl_loss + 0.5 * recon_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(obs_pred.parameters()), 1.0
        )
        optimizer.step()

        h = h.detach()
        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model: MinimalRSSM, obs_pred: nn.Module, sequences: list[np.ndarray]) -> float:
    model.eval()
    obs_pred.eval()
    total_loss, count = 0.0, 0
    for seq_embs in sequences:
        if len(seq_embs) < 2:
            continue
        h = model.zero_state()
        for i in range(len(seq_embs) - 1):
            obs_t  = torch.from_numpy(seq_embs[i]).float().unsqueeze(0)
            obs_t1 = torch.from_numpy(seq_embs[i + 1]).float().unsqueeze(0)
            action_idx = 0 if i % 2 == 0 else 6
            act = F.one_hot(torch.tensor([action_idx]), model.n_actions).float()
            h = model.gru(torch.cat([obs_t, act], dim=-1), h)
            post_in      = torch.cat([h, obs_t], dim=-1)
            z_mean       = model.post_mean(post_in)
            z_log_std    = model.post_log_std(post_in)
            z = z_mean + torch.exp(z_log_std.clamp(-4, 2)) * torch.randn_like(z_mean)
            prior_mean    = model.prior_mean(h)
            prior_log_std = model.prior_log_std(h)
            post_in_next  = torch.cat([h, obs_t1], dim=-1)
            z_next_mean   = model.post_mean(post_in_next)
            z_next_log_std= model.post_log_std(post_in_next)
            kl   = _kl_divergence(prior_mean, prior_log_std, z_next_mean, z_next_log_std)
            feat = torch.cat([h, z], dim=-1)
            recon = F.mse_loss(obs_pred(feat), obs_t)
            total_loss += (kl + 0.5 * recon).item()
            count += 1
            h = h.detach()
    return total_loss / max(count, 1)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-entrenamiento RSSM")
    parser.add_argument("--epochs",       type=int,   default=15)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--min-seq-len",  type=int,   default=3)
    parser.add_argument("--wiki-articles",type=int,   default=300,
                        help="Artículos de Wikipedia ES a cargar (0 = desactivar)")
    args = parser.parse_args()

    from worldmodel.training_logger import setup_logger
    import builtins
    _logger = setup_logger("pretrain_rssm")
    builtins.print = lambda *a, **k: _logger.info(" ".join(str(x) for x in a))

    print("=" * 60)
    print("  RSSM PRE-TRAINING — Stella + Conocimiento del Mundo Real")
    print("=" * 60)

    # ── 1. Datos de Stella ────────────────────────────────────
    print("\n[1/4] Cargando datos de Stella...")
    chat_seqs = load_chat_sequences()
    thoughts  = load_thoughts()
    episodes  = load_episodes()
    research  = load_research_steps()

    print(f"  Chats:      {len(chat_seqs)} sesiones ({sum(len(s) for s in chat_seqs)} msgs)")
    print(f"  Thoughts:   {len(thoughts)}")
    print(f"  Episodes:   {len(episodes)}")
    print(f"  Research:   {len(research)}")

    # ── 2. Conocimiento del mundo ─────────────────────────────
    world_seqs_raw: list[list[str]] = []
    if args.wiki_articles > 0:
        print("\n[2/4] Cargando conocimiento del mundo real (Wikipedia ES)...")
        world_seqs_raw = load_world_knowledge_sequences(args.wiki_articles)
    else:
        print("\n[2/4] Wikipedia ES desactivada (--wiki-articles 0)")

    # ── 3. Codificar en GPU ───────────────────────────────────
    print("\n[3/4] Codificando en GPU (batch)...")
    all_sequences: list[np.ndarray] = []

    # Stella: chats (ya son listas de strings)
    for seq in chat_seqs:
        if len(seq) >= args.min_seq_len:
            embs = encode_batch_gpu(seq)
            all_sequences.append(embs)

    # Stella: thoughts → ventanas de 4
    if len(thoughts) >= 4:
        t_embs = encode_batch_gpu(thoughts)
        print(f"  Thoughts codificados: {len(t_embs)}")
        for i in range(0, len(t_embs) - 3, 2):
            all_sequences.append(t_embs[i:i + 4])

    # Stella: episodes → ventanas de 4
    if len(episodes) >= 4:
        e_embs = encode_batch_gpu(episodes)
        print(f"  Episodes codificados: {len(e_embs)}")
        for i in range(0, len(e_embs) - 3, 3):
            all_sequences.append(e_embs[i:i + 4])

    # Stella: research → ventanas de 3
    if len(research) >= 3:
        r_embs = encode_batch_gpu(research)
        for i in range(0, len(r_embs) - 2, 2):
            all_sequences.append(r_embs[i:i + 3])

    n_stella = len(all_sequences)
    print(f"  Secuencias de Stella: {n_stella}")

    # Wikipedia ES → codificar todas las oraciones de golpe
    if world_seqs_raw:
        all_wiki_texts: list[str] = []
        wiki_seq_lens:  list[int] = []
        for seq in world_seqs_raw:
            all_wiki_texts.extend(seq)
            wiki_seq_lens.append(len(seq))

        print(f"  Codificando {len(all_wiki_texts)} oraciones de Wikipedia ES...")
        all_wiki_embs = encode_batch_gpu(all_wiki_texts, batch_size=128)

        idx = 0
        for seqlen in wiki_seq_lens:
            seq_embs = all_wiki_embs[idx : idx + seqlen]
            if len(seq_embs) >= 2:
                all_sequences.append(seq_embs)
            idx += seqlen

        n_wiki = len(all_sequences) - n_stella
        print(f"  Secuencias Wikipedia ES: {n_wiki}")

    n_total = len(all_sequences)
    n_wiki  = n_total - n_stella
    total_steps = sum(len(s) - 1 for s in all_sequences)
    print(f"\n  Total secuencias: {n_total}  ({n_stella} Stella + {n_wiki} Wikipedia)")
    print(f"  Total steps:      {total_steps}")

    if not all_sequences:
        print("\nERROR: No hay secuencias. Verifica rutas y conectividad HuggingFace.")
        sys.exit(1)

    # Val split (10%)
    random.shuffle(all_sequences)
    n_val    = max(1, int(n_total * 0.1))
    val_seqs   = all_sequences[:n_val]
    train_seqs = all_sequences[n_val:]
    print(f"  Train: {len(train_seqs)} | Val: {len(val_seqs)}")

    # ── 4. Entrenar ───────────────────────────────────────────
    print(f"\n[4/4] Entrenando RSSM ({args.epochs} épocas, lr={args.lr})...")
    model = load_or_create(WEIGHTS_PATH)

    feat_dim = model.hidden_dim + model.latent_dim  # 320
    obs_pred = nn.Linear(feat_dim, model.obs_dim)   # cabeza temporal, no se guarda

    all_params = list(model.parameters()) + list(obs_pred.parameters())
    optimizer  = Adam(all_params, lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_seqs)
        train_loss = sum(
            train_on_sequence(model, obs_pred, optimizer, seq)
            for seq in train_seqs
        ) / len(train_seqs)
        val_loss = evaluate(model, obs_pred, val_seqs)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            model.save(WEIGHTS_PATH)
            marker = " ← checkpoint"

        print(f"  Época {epoch:2d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f}{marker}")

    print(f"\n  Mejor val_loss: {best_val:.4f}")
    print(f"  Pesos guardados: {WEIGHTS_PATH}")
    print(f"  Tamaño: {WEIGHTS_PATH.stat().st_size / 1024:.1f} KB")
    print("\n¡Pre-entrenamiento completado! El RSSM ahora conoce el mundo.")
    print("Reinicia el demo para activar el RSSM entrenado.")
    print("=" * 60)


if __name__ == "__main__":
    main()
