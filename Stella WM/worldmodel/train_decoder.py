"""
EXP-001 — Entrenamiento del SmallDecoder desde cero.

Pipeline:
  1. Carga DailyDialog (HuggingFace datasets) — fallback a chats de Stella
  2. Pre-computa vectores de contexto RSSM para cada turno de dialogo
  3. Entrena el decoder con teacher forcing (cross-entropy sobre tokens de respuesta)
  4. El RSSM esta CONGELADO — language_grads=False (patron Somniloquy / Niimi 2026)

Uso:
  python -m worldmodel.train_decoder
  python -m worldmodel.train_decoder --epochs 5 --batch-size 16 --lr 1e-4
  python -m worldmodel.train_decoder --cpu          # forzar CPU

Requiere:
  pip install datasets   (para DailyDialog)
  transformers           (tokenizer GPT-2, normalmente ya instalado)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))
from worldmodel.rssm import load_or_create as load_rssm
from worldmodel.obs_encoder import encode_observation
from worldmodel.decoder import SmallDecoder, DECODER_WEIGHTS, VOCAB_SIZE, MAX_SEQ, CTX_DIM

STELLA_CHATS = Path("D:/stella/memory/store/stella.chats.jsonl")
RSSM_WEIGHTS = Path("worldmodel/weights/rssm.pt")


# ─── Carga de datos ────────────────────────────────────────────────────────────

def load_dialogue_dataset() -> list[list[str]]:
    """
    Intenta cargar datasets de dialogo en cascada hasta encontrar uno disponible.
    Todos son Parquet nativo — sin scripts legacy.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[data] 'datasets' no instalado. Instala con: pip install datasets")
        return []

    candidates = [
        # (nombre_hf, split, campo_dialogos, formato)
        ("DavidVivancos/DailyDialog",  "train", "dialog",    "list"),
        ("Fraser/daily-dialog",        "train", "dialog",    "list"),
        ("blended_skill_talk",         "train", "free_messages", "interleave"),
        ("empathetic_dialogues",       "train", "utterance", "context_split"),
    ]

    for name, split, field, fmt in candidates:
        try:
            print(f"[data] Intentando: {name}...")
            ds = load_dataset(name, split=split)
            dialogues = []

            if fmt == "list":
                dialogues = [row[field] for row in ds if isinstance(row[field], list) and len(row[field]) >= 2]

            elif fmt == "interleave":
                # blended_skill_talk: alterna previous_utterance + free_messages
                for row in ds:
                    prev = row.get("previous_utterance", [])
                    free = row.get("free_messages", [])
                    combined = prev + free
                    if len(combined) >= 2:
                        dialogues.append(combined)

            elif fmt == "context_split":
                # empathetic_dialogues: conv_id agrupa turnos
                from collections import defaultdict
                convs = defaultdict(list)
                for row in ds:
                    cid = row.get("conv_id", "")
                    utt = row.get("utterance", "").strip()
                    if cid and utt:
                        convs[cid].append(utt)
                dialogues = [v for v in convs.values() if len(v) >= 2]

            if dialogues:
                print(f"[data] {name}: {len(dialogues)} dialogos cargados")
                return dialogues
        except Exception as e:
            print(f"[data]   -> fallo: {e}")

    return []


def load_stella_chats() -> list[list[str]]:
    """Fallback: usa las conversaciones historicas de Stella."""
    if not STELLA_CHATS.exists():
        return []
    msgs = []
    for line in STELLA_CHATS.read_text(encoding="utf-8").splitlines():
        try:
            msgs.append(json.loads(line))
        except Exception:
            pass

    sessions: list[list[str]] = []
    current: list[str] = []
    prev_ts = None

    for m in msgs:
        content = m.get("content", "").strip()
        if not content:
            continue
        ts_str = m.get("ts", "")
        if prev_ts and ts_str:
            try:
                from datetime import datetime
                t1 = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if (t2 - t1).total_seconds() > 3600 and current:
                    sessions.append(current)
                    current = []
            except Exception:
                pass
        current.append(content)
        prev_ts = ts_str

    if current:
        sessions.append(current)
    print(f"[data] Chats de Stella: {len(sessions)} sesiones cargadas")
    return sessions


# ─── Pre-computo de vectores de contexto ─────────────────────────────────────

def precompute_pairs(
    dialogues: list[list[str]],
    rssm,
    min_words: int = 4,
    encode_batch_size: int = 64,
) -> list[tuple[np.ndarray, str]]:
    """
    Para cada turno en cada dialogo, corre el RSSM sobre el historial
    y registra (context_vec [320], texto_respuesta) como par de entrenamiento.
    Usa batch encoding en GPU para acelerar el paso mas lento.
    """
    from worldmodel.obs_encoder import _hf_tokenizer, _hf_model, _hf_device, _st_projector, OBS_DIM
    import torch

    # Recopilar todos los enunciados primero para batch-encoding
    print("  Recopilando enunciados para batch encoding...")
    all_utts: list[str] = []
    dialogue_structure: list[list[int]] = []  # indices en all_utts para cada dialogo

    for dialogue in dialogues:
        if len(dialogue) < 2:
            dialogue_structure.append([])
            continue
        idxs = []
        for utt in dialogue:
            idxs.append(len(all_utts))
            all_utts.append(utt)
        dialogue_structure.append(idxs)

    print(f"  {len(all_utts)} enunciados totales — codificando en GPU...")

    # Batch encode todos los enunciados de una vez
    all_obs: list[np.ndarray] = []

    if _hf_model is not None and _hf_tokenizer is not None:
        # Batch encoding en GPU
        for i in range(0, len(all_utts), encode_batch_size):
            batch_texts = all_utts[i:i + encode_batch_size]
            enc = _hf_tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=128, padding=True
            )
            enc = {k: v.to(_hf_device) for k, v in enc.items()}
            with torch.no_grad():
                out = _hf_model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)  # [B, 384]
            embs = embs.cpu().numpy()
            for emb in embs:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                proj = emb @ _st_projector                               # [OBS_DIM]
                proj = proj / (np.linalg.norm(proj) + 1e-8)
                all_obs.append(proj)
            if i % 1000 == 0:
                print(f"  {min(i+encode_batch_size, len(all_utts))}/{len(all_utts)} encodings...", end="\r")
        print()
    else:
        # Fallback: encode_observation uno a uno
        for i, utt in enumerate(all_utts):
            all_obs.append(encode_observation(utt, session_length=0))
            if i % 200 == 0:
                print(f"  {i}/{len(all_utts)}...", end="\r")
        print()

    # Ahora construir pares pasando el RSSM por el historial
    pairs: list[tuple[np.ndarray, str]] = []
    for d_idx, (dialogue, idxs) in enumerate(zip(dialogues, dialogue_structure)):
        if len(idxs) < 2:
            continue
        h = None
        for turn_i, (utt_idx, utterance) in enumerate(zip(idxs[:-1], dialogue[:-1])):
            obs = all_obs[utt_idx]
            z_np, h, _, _, _ = rssm.step(obs, action_idx=turn_i % 2, h_prev=h)
            ctx_vec = np.concatenate([h.numpy().squeeze(), z_np])
            response = dialogue[turn_i + 1].strip()
            if len(response.split()) >= min_words:
                pairs.append((ctx_vec, response))

        if d_idx % 500 == 0:
            print(f"  RSSM: {d_idx}/{len(dialogues)} dialogos...", end="\r")
    print()

    print(f"[data] {len(pairs)} pares (contexto, respuesta) listos")
    return pairs


# ─── Tokenizacion ─────────────────────────────────────────────────────────────

def tokenize_batch(
    texts: list[str],
    tokenizer,
    max_len: int = MAX_SEQ,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convierte lista de textos en tensores input/target para teacher forcing.
    input:  [BOS, t1, t2, ..., tN-1]
    target: [t1, t2, ..., tN, EOS]   (-100 en padding = ignorado en CE loss)
    """
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    inputs, targets = [], []

    for text in texts:
        ids = tokenizer.encode(text, max_length=max_len - 1, truncation=True)
        inp = [bos] + ids
        tgt = ids + [eos]
        pad = max_len - len(inp)
        inp = inp + [eos] * pad
        tgt = tgt + [-100] * pad
        inputs.append(inp)
        targets.append(tgt)

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


# ─── Evaluacion ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(
    decoder: SmallDecoder,
    pairs: list[tuple[np.ndarray, str]],
    tokenizer,
    device: torch.device,
    n_batches: int = 20,
    batch_size: int = 16,
) -> float:
    decoder.eval()
    total, steps = 0.0, 0
    sample_pairs = pairs[:n_batches * batch_size]
    for i in range(0, len(sample_pairs), batch_size):
        batch = sample_pairs[i:i + batch_size]
        if not batch:
            break
        ctx_vecs, texts = zip(*batch)
        ctx = torch.tensor(np.stack(ctx_vecs), dtype=torch.float32).to(device)
        inp, tgt = tokenize_batch(list(texts), tokenizer)
        inp, tgt = inp.to(device), tgt.to(device)
        logits = decoder(inp, ctx)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=-100)
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EXP-001: Entrenar SmallDecoder desde cero")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--max-pairs",   type=int,   default=60000,
                        help="Maximo de pares de entrenamiento (0=sin limite)")
    parser.add_argument("--cpu",         action="store_true", help="Forzar CPU")
    parser.add_argument("--sample-every",type=int,   default=200,
                        help="Mostrar muestra generada cada N pasos")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 60)
    print("  EXP-001 -- SmallDecoder from scratch")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── 1. Tokenizer ──────────────────────────────────────────
    print("\n[1/5] Cargando tokenizer GPT-2...")
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # ── 2. Datos ──────────────────────────────────────────────
    print("\n[2/5] Cargando datos de dialogo...")
    dialogues = load_dialogue_dataset()
    if not dialogues:
        print("  Usando chats de Stella como fallback...")
        dialogues = load_stella_chats()
    if not dialogues:
        print("ERROR: Sin datos. Instala: pip install datasets")
        sys.exit(1)

    # ── 3. RSSM (congelado) ───────────────────────────────────
    print("\n[3/5] Cargando RSSM (congelado durante entrenamiento del decoder)...")
    rssm = load_rssm(RSSM_WEIGHTS)
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad_(False)
    print("  RSSM congelado. language_grads=False.")

    # ── 4. Pre-computar vectores de contexto ──────────────────
    print("\n[4/5] Pre-computando vectores de contexto RSSM...")
    all_pairs = precompute_pairs(dialogues, rssm)
    random.shuffle(all_pairs)
    if args.max_pairs > 0 and len(all_pairs) > args.max_pairs:
        all_pairs = all_pairs[:args.max_pairs]
        print(f"  Limitado a {args.max_pairs} pares")

    split     = int(0.95 * len(all_pairs))
    train_set = all_pairs[:split]
    val_set   = all_pairs[split:]
    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")

    # ── 5. Entrenar decoder ───────────────────────────────────
    print(f"\n[5/5] Entrenando SmallDecoder ({args.epochs} epocas, lr={args.lr})...")
    decoder   = SmallDecoder().to(device)
    n_params  = decoder.param_count()
    print(f"  Parametros: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01)
    n_steps   = args.epochs * (len(train_set) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_steps, 1))

    best_val  = float("inf")
    step      = 0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_set)
        decoder.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for i in range(0, len(train_set), args.batch_size):
            batch = train_set[i:i + args.batch_size]
            if not batch:
                continue
            ctx_vecs, texts = zip(*batch)
            ctx = torch.tensor(np.stack(ctx_vecs), dtype=torch.float32).to(device)
            inp, tgt = tokenize_batch(list(texts), tokenizer)
            inp, tgt = inp.to(device), tgt.to(device)

            logits = decoder(inp, ctx)
            loss   = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                tgt.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_steps += 1
            step        += 1

            if step % args.sample_every == 0:
                # Muestra de generacion
                decoder.eval()
                sample_ctx = torch.tensor(
                    val_set[0][0], dtype=torch.float32
                ).unsqueeze(0).to(device)
                sample_ids  = decoder.generate(sample_ctx, max_new=40, temperature=0.85)
                sample_text = tokenizer.decode(sample_ids, skip_special_tokens=True)
                print(f"\n  [step {step}] muestra: {sample_text[:120]!r}")
                decoder.train()

        avg_train = epoch_loss / max(epoch_steps, 1)
        avg_val   = eval_loss(decoder, val_set, tokenizer, device)

        print(f"  Epoca {epoch:2d}/{args.epochs} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            decoder.save(DECODER_WEIGHTS)
            print(f"  -> Checkpoint guardado (val_loss={best_val:.4f})")

    print(f"\nEntrenamiento completado! Mejor val_loss: {best_val:.4f}")
    print(f"Decoder guardado en: {DECODER_WEIGHTS}")
    print("\nActualiza EXPERIMENTS.md con los resultados.")
    print("=" * 60)


if __name__ == "__main__":
    main()
