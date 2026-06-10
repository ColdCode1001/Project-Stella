"""
Pre-entrenamiento del LargeRSSM (128.8M params) con conocimiento del mundo real.

Bugs corregidos vs v1:
  - GPU training: model.to(device), todos los tensores al device
  - Ventanas NO solapadas (step=window, no step=window//2) -> ~50K seqs, no 534K
  - Limite de secuencias totales (--max-sequences)
  - Cache de encodings en disco -> no re-encodes al relanzar
  - Progress log cada N batches dentro de cada época

Uso:
  python -m worldmodel.pretrain_large
  python -m worldmodel.pretrain_large --epochs 10 --max-articles 2000
  python -m worldmodel.pretrain_large --no-wiki   (solo datos de Stella)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))
from worldmodel.rssm import LargeRSSM
from worldmodel.obs_encoder import encode_batch_gpu

RSSM_LARGE_PATH  = Path("worldmodel/weights/rssm_large.pt")
ENCODE_CACHE     = Path("worldmodel/weights/pretrain_large_cache.npz")
STELLA_CHATS     = Path("D:/stella/memory/store/stella.chats.jsonl")
STELLA_THOUGHTS  = Path("D:/stella/memory/store/stella.thoughts.jsonl")
STELLA_EPISODIC  = Path("D:/stella/memory/store/stella.episodic")
STELLA_WEB       = Path("D:/stella/memory/store/stella.web.jsonl")


# ─── Datos ─────────────────────────────────────────────────────────────────────

def load_stella_sequences() -> list[list[str]]:
    import json
    sequences: list[list[str]] = []

    if STELLA_CHATS.exists():
        msgs = []
        for line in STELLA_CHATS.read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line)
                if m.get("content", "").strip():
                    msgs.append(m.get("content", "").strip())
            except Exception:
                pass
        for i in range(0, len(msgs), 20):
            seq = msgs[i:i + 20]
            if len(seq) >= 2:
                sequences.append(seq)
        print(f"  Chats: {len(msgs)} mensajes -> {len(sequences)} seqs")

    if STELLA_THOUGHTS.exists():
        n_before = len(sequences)
        for line in STELLA_THOUGHTS.read_text(encoding="utf-8").splitlines():
            try:
                c = json.loads(line).get("content", "").strip()
                if c and len(c.split()) >= 10:
                    sents = [s.strip() for s in c.split(".") if len(s.strip()) > 5]
                    if len(sents) >= 2:
                        sequences.append(sents[:10])  # max 10 oraciones por pensamiento
            except Exception:
                pass
        print(f"  Pensamientos: {len(sequences) - n_before} seqs")

    if STELLA_EPISODIC.exists():
        try:
            data = json.loads(STELLA_EPISODIC.read_text(encoding="utf-8"))
            eps = [e.get("content", "").strip() for e in data.get("episodes", [])
                   if len(e.get("content", "").split()) >= 5]
            if eps:
                sequences.append(eps)
                print(f"  Episodios: {len(eps)} items -> 1 seq")
        except Exception:
            pass

    if STELLA_WEB.exists():
        triggers = []
        for line in STELLA_WEB.read_text(encoding="utf-8").splitlines():
            try:
                t = json.loads(line).get("trigger", "").strip()
                if len(t.split()) >= 4:
                    triggers.append(t)
            except Exception:
                pass
        if triggers:
            sequences.append(triggers)
            print(f"  Web: {len(triggers)} triggers -> 1 seq")

    return sequences


def load_wiki_sequences(
    max_articles: int = 2000,
    window: int = 5,
    max_seqs_per_article: int = 3,
) -> list[list[str]]:
    """
    Ventanas NO solapadas (step=window) y limite por articulo.
    Con max_articles=2000 y max_seqs_per_article=3 -> ~6000 seqs de Wikipedia.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets no instalado -- sin Wikipedia")
        return []

    print(f"  Streaming Wikipedia ES ({max_articles} articulos, max {max_seqs_per_article} seqs/art)...")
    sequences: list[list[str]] = []
    n_articles = 0

    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.es",
                          split="train", streaming=True)
        for article in ds:
            if n_articles >= max_articles:
                break
            text = article.get("text", "").strip()
            if not text or len(text) < 200:
                continue

            sents = []
            for raw in text.replace("\n\n", "\n").split("\n"):
                for s in raw.split(". "):
                    s = s.strip()
                    if len(s.split()) >= 6:
                        sents.append(s)

            if len(sents) < window:
                continue

            # Ventanas NO solapadas: step=window
            art_seqs = 0
            for i in range(0, len(sents) - window + 1, window):
                if art_seqs >= max_seqs_per_article:
                    break
                sequences.append(sents[i:i + window])
                art_seqs += 1

            n_articles += 1
            if n_articles % 500 == 0:
                print(f"    {n_articles}/{max_articles} arts, {len(sequences)} seqs")

    except Exception as e:
        print(f"  ERROR Wikipedia: {e}")
        return []

    print(f"  Wikipedia: {n_articles} arts -> {len(sequences)} seqs")
    return sequences


# ─── Codificacion con cache ─────────────────────────────────────────────────────

def encode_or_load(
    sequences: list[list[str]],
    cache_path: Path,
    force: bool = False,
) -> list[list[np.ndarray]]:
    """Codifica secuencias o carga desde cache si el cache existe."""
    if cache_path.exists() and not force:
        print(f"  Cargando encodings desde cache ({cache_path})...")
        data = np.load(cache_path, allow_pickle=True)
        encoded = data["encoded"].tolist()
        print(f"  Cache: {len(encoded)} seqs")
        return encoded

    all_texts: list[str] = []
    seq_lens:  list[int] = []
    for seq in sequences:
        seq_lens.append(len(seq))
        all_texts.extend(seq)

    print(f"  Codificando {len(all_texts):,} textos en GPU...")
    all_embs = encode_batch_gpu(all_texts, batch_size=512)

    encoded: list[list[np.ndarray]] = []
    idx = 0
    for length in seq_lens:
        encoded.append([all_embs[idx + i] for i in range(length)])
        idx += length

    # Guardar cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), np.array(encoded, dtype=object))
    # npz necesita array de objetos
    np.savez(str(cache_path).replace(".npz", "_raw.npz"),
             encoded=np.array(encoded, dtype=object))
    print(f"  Cache guardado: {cache_path}")
    return encoded


# ─── Training ──────────────────────────────────────────────────────────────────

def train_rssm_large(
    model:      LargeRSSM,
    train_seqs: list[list[np.ndarray]],
    val_seqs:   list[list[np.ndarray]],
    device:     torch.device,
    epochs:     int   = 10,
    lr:         float = 1e-4,
    batch_size: int   = 32,
    log_every:  int   = 200,
) -> float:
    obs_proj = nn.Linear(model.feat_dim, model.obs_dim).to(device)
    optimizer = AdamW(
        list(model.parameters()) + list(obs_proj.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    n_steps   = epochs * max(1, len(train_seqs) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    best_val  = float("inf")
    step      = 0

    for epoch in range(1, epochs + 1):
        random.shuffle(train_seqs)
        model.train()
        obs_proj.train()
        epoch_loss, epoch_steps = 0.0, 0

        for i in range(0, len(train_seqs), batch_size):
            batch = train_seqs[i:i + batch_size]
            if not batch:
                continue

            total_loss = torch.tensor(0.0, device=device)
            n_pairs    = 0

            for seq in batch:
                h = model.zero_state().to(device)  # [n_layers, 1, hidden_dim]
                for j in range(len(seq) - 1):
                    obs_t   = torch.from_numpy(seq[j]).float().unsqueeze(0).to(device)
                    obs_tp1 = torch.from_numpy(seq[j + 1]).float().unsqueeze(0).to(device)

                    gru_in = torch.cat([obs_t.unsqueeze(0),
                                        torch.zeros(1, 1, model.n_actions, device=device)], dim=-1)
                    _, h_new = model.gru(gru_in, h)
                    h_top    = h_new[-1]

                    # Posterior
                    post_out = model.posterior(torch.cat([h_top, obs_t], dim=-1))
                    post_mean, post_log_std = post_out.chunk(2, dim=-1)
                    post_std = torch.exp(post_log_std.clamp(-4, 2))
                    z        = post_mean + post_std * torch.randn_like(post_mean)

                    # Prior
                    prior_out = model.prior(h_top)
                    prior_mean, prior_log_std = prior_out.chunk(2, dim=-1)
                    prior_std = torch.exp(prior_log_std.clamp(-4, 2))

                    # KL
                    kl = (
                        prior_log_std - post_log_std
                        + (post_std.pow(2) + (post_mean - prior_mean).pow(2))
                          / (2 * prior_std.pow(2) + 1e-8)
                        - 0.5
                    ).sum(-1).mean()

                    # Reconstruccion
                    feat  = torch.cat([h_top, z], dim=-1)
                    recon = F.mse_loss(obs_proj(feat), obs_tp1)

                    total_loss = total_loss + kl + 0.5 * recon
                    n_pairs   += 1
                    h          = h_new.detach()

            if n_pairs == 0:
                continue

            loss = total_loss / n_pairs
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_steps += 1
            step        += 1

            if step % log_every == 0:
                avg = epoch_loss / epoch_steps
                pct = 100 * i / len(train_seqs)
                print(f"    step {step} | ep{epoch} {pct:.0f}% | loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        avg_train = epoch_loss / max(epoch_steps, 1)
        avg_val   = eval_loss(model, obs_proj, val_seqs[:200], device)

        marker = ""
        if avg_val < best_val:
            best_val = avg_val
            RSSM_LARGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            model.save(RSSM_LARGE_PATH)
            marker = " <- checkpoint"

        print(f"  Epoca {epoch:2d}/{epochs} | train={avg_train:.4f} | val={avg_val:.4f}{marker}")

    return best_val


@torch.no_grad()
def eval_loss(
    model:    LargeRSSM,
    obs_proj: nn.Module,
    seqs:     list[list[np.ndarray]],
    device:   torch.device,
) -> float:
    model.eval()
    obs_proj.eval()
    total, steps = 0.0, 0
    for seq in seqs:
        h = model.zero_state().to(device)
        for j in range(len(seq) - 1):
            obs_t   = torch.from_numpy(seq[j]).float().unsqueeze(0).to(device)
            obs_tp1 = torch.from_numpy(seq[j + 1]).float().unsqueeze(0).to(device)
            gru_in  = torch.cat([obs_t.unsqueeze(0),
                                 torch.zeros(1, 1, model.n_actions, device=device)], dim=-1)
            _, h_new = model.gru(gru_in, h)
            h_top    = h_new[-1]
            post_out = model.posterior(torch.cat([h_top, obs_t], dim=-1))
            post_mean, post_log_std = post_out.chunk(2, dim=-1)
            post_std = torch.exp(post_log_std.clamp(-4, 2))
            z        = post_mean + post_std * torch.randn_like(post_mean)
            prior_out = model.prior(h_top)
            prior_mean, prior_log_std = prior_out.chunk(2, dim=-1)
            prior_std = torch.exp(prior_log_std.clamp(-4, 2))
            kl = (
                prior_log_std - post_log_std
                + (post_std.pow(2) + (post_mean - prior_mean).pow(2))
                  / (2 * prior_std.pow(2) + 1e-8)
                - 0.5
            ).sum(-1).mean()
            feat  = torch.cat([h_top, z], dim=-1)
            recon = F.mse_loss(obs_proj(feat), obs_tp1)
            total += (kl + 0.5 * recon).item()
            steps += 1
            h = h_new
    return total / max(steps, 1)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--max-articles",  type=int,   default=2000,
                        help="Articulos Wikipedia (default 2000 -> ~6000 seqs)")
    parser.add_argument("--max-seqs",      type=int,   default=50000,
                        help="Cap total de secuencias de entrenamiento")
    parser.add_argument("--no-wiki",       action="store_true")
    parser.add_argument("--from-scratch",  action="store_true")
    parser.add_argument("--no-cache",      action="store_true",
                        help="Ignorar cache y re-codificar")
    parser.add_argument("--log-every",     type=int,   default=200)
    args = parser.parse_args()

    from worldmodel.training_logger import setup_logger
    import builtins
    _logger = setup_logger("pretrain_large")
    builtins.print = lambda *a, **k: _logger.info(" ".join(str(x) for x in a))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 62)
    print("  PRETRAIN — LargeRSSM 128.8M params")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"  Wikipedia: {'NO' if args.no_wiki else f'{args.max_articles} arts'} | Max seqs: {args.max_seqs:,}")
    print("=" * 62)

    # ── 1. Modelo ─────────────────────────────────────────────────
    print("\n[1/4] LargeRSSM...")
    model = LargeRSSM().to(device)
    print(f"  {model.param_count()/1e6:.1f}M params en {device}")

    if not args.from_scratch and RSSM_LARGE_PATH.exists():
        try:
            model.load(RSSM_LARGE_PATH)
            print(f"  Pesos cargados desde {RSSM_LARGE_PATH}")
        except Exception as e:
            print(f"  No se pudo cargar ({e}) — desde cero")

    # ── 2. Datos Stella ───────────────────────────────────────────
    print("\n[2/4] Datos de Stella...")
    stella_seqs = load_stella_sequences()
    print(f"  Total Stella: {len(stella_seqs)} seqs")

    # ── 3. Wikipedia ES ───────────────────────────────────────────
    wiki_seqs: list[list[str]] = []
    if not args.no_wiki:
        print(f"\n[3/4] Wikipedia ES...")
        wiki_seqs = load_wiki_sequences(max_articles=args.max_articles)
    else:
        print("\n[3/4] Sin Wikipedia (modo imprinted)")

    # ── 4. Codificar + entrenar ───────────────────────────────────
    print(f"\n[4/4] Codificando y entrenando...")

    all_text_seqs = stella_seqs + wiki_seqs
    random.shuffle(all_text_seqs)

    # Cap de secuencias
    if len(all_text_seqs) > args.max_seqs:
        print(f"  Limitando a {args.max_seqs:,} seqs (de {len(all_text_seqs):,})")
        all_text_seqs = all_text_seqs[:args.max_seqs]

    print(f"  Total a codificar: {len(all_text_seqs):,} seqs")

    # Cache path incluye el hash del tamaño para invalidar si cambian los datos
    cache_tag = f"{len(all_text_seqs)}"
    cache_path = Path(f"worldmodel/weights/pretrain_large_cache_{cache_tag}.npy")

    if cache_path.exists() and not args.no_cache:
        print(f"  Cargando cache {cache_path}...")
        encoded = np.load(str(cache_path), allow_pickle=True).tolist()
        print(f"  {len(encoded):,} seqs desde cache")
    else:
        # Codificar todas en GPU
        all_texts: list[str] = []
        seq_lens:  list[int] = []
        for seq in all_text_seqs:
            seq_lens.append(len(seq))
            all_texts.extend(seq)
        print(f"  Codificando {len(all_texts):,} textos en GPU...")
        all_embs = encode_batch_gpu(all_texts, batch_size=512)
        encoded  = []
        idx = 0
        for length in seq_lens:
            encoded.append([all_embs[idx + i] for i in range(length)])
            idx += length
        np.save(str(cache_path), np.array(encoded, dtype=object))
        print(f"  Cache guardado: {cache_path}")

    n_val   = max(50, int(len(encoded) * 0.05))
    val_e   = encoded[:n_val]
    train_e = encoded[n_val:]
    print(f"  Train: {len(train_e):,} | Val: {len(val_e):,}")

    best = train_rssm_large(
        model, train_e, val_e, device,
        args.epochs, args.lr, args.batch_size, args.log_every,
    )

    print(f"\n  Mejor val_loss: {best:.4f}")
    print(f"  Guardado: {RSSM_LARGE_PATH}")
    if RSSM_LARGE_PATH.exists():
        print(f"  Tamano: {RSSM_LARGE_PATH.stat().st_size / 1e6:.0f} MB")
    print("\nLargeRSSM listo. Probar con: probe_wm.bat large")
    print("=" * 62)


if __name__ == "__main__":
    main()
