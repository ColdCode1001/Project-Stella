"""
Entrenamiento del decoder en la voz única de Stella.

Filosofía:
  - Dataset: SOLO los 402 mensajes reales de Stella (stella.chats.jsonl)
  - RSSM: YA ENTRENADO (rssm.pt) — el cerebro tiene significado real
  - Decoder: fine-tune desde decoder.pt — aprende la VOZ de Stella, no inglés genérico
  - Resultado: decoder_stella.pt — la boca habla como Stella

Por qué fine-tune y no desde cero:
  El decoder ya sabe "cómo generar texto coherente desde un vector".
  Solo necesita adaptar ese conocimiento a la voz de Stella en español.
  Sin fine-tune, 402 pares son demasiado pocos para aprender articulación + voz a la vez.

Uso:
  python -m worldmodel.train_decoder_stella
  python -m worldmodel.train_decoder_stella --epochs 50 --lr 3e-5
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
from worldmodel.obs_encoder import encode_batch_gpu
from worldmodel.decoder import SmallDecoder, VOCAB_SIZE, MAX_SEQ, CTX_DIM

STELLA_CHATS        = Path("D:/stella/memory/store/stella.chats.jsonl")
RSSM_WEIGHTS        = Path("worldmodel/weights/rssm.pt")
DECODER_BASE        = Path("worldmodel/weights/decoder.pt")        # punto de partida
DECODER_STELLA      = Path("worldmodel/weights/decoder_stella.pt") # resultado final


# ─── Carga de datos ────────────────────────────────────────────────────────────

def load_raw_messages() -> list[dict]:
    """Carga todos los mensajes en orden cronológico."""
    if not STELLA_CHATS.exists():
        print(f"ERROR: No se encuentra {STELLA_CHATS}")
        sys.exit(1)
    msgs = []
    for line in STELLA_CHATS.read_text(encoding="utf-8").splitlines():
        try:
            m = json.loads(line)
            if m.get("content", "").strip():
                msgs.append(m)
        except Exception:
            pass
    return msgs


def group_into_sessions(msgs: list[dict], gap_seconds: float = 3600) -> list[list[dict]]:
    """Agrupa mensajes en sesiones separadas por gaps de silencio."""
    sessions: list[list[dict]] = []
    current:  list[dict]       = []
    prev_ts = None

    for m in msgs:
        ts_str = m.get("ts", "")
        if prev_ts and ts_str:
            try:
                from datetime import datetime
                t1 = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if (t2 - t1).total_seconds() > gap_seconds and current:
                    sessions.append(current)
                    current = []
            except Exception:
                pass
        current.append(m)
        prev_ts = ts_str

    if current:
        sessions.append(current)
    return sessions


def build_pairs(
    sessions: list[list[dict]],
    rssm,
    min_words: int = 3,
) -> list[tuple[np.ndarray, str]]:
    """
    Construye pares (ctx_vec [320D], texto_stella) para entrenamiento.
    El RSSM procesa el historial completo de cada sesión — contexto real.
    Solo Stella habla: las respuestas de Arca son solo contexto, nunca target.
    """
    # Recolectar todos los textos para batch encoding
    all_texts:   list[str]         = []
    sess_idxs:   list[list[int]]   = []
    sess_data:   list[list[dict]]  = []

    for session in sessions:
        if len(session) < 2:
            continue
        idxs = []
        for msg in session:
            idxs.append(len(all_texts))
            all_texts.append(msg["content"])
        sess_idxs.append(idxs)
        sess_data.append(session)

    if not all_texts:
        return []

    print(f"  Codificando {len(all_texts)} mensajes en GPU...")
    all_embs = encode_batch_gpu(all_texts, batch_size=64)

    # Construir pares reproduciendo el RSSM sesión por sesión
    pairs: list[tuple[np.ndarray, str]] = []

    for session, idxs in zip(sess_data, sess_idxs):
        h = None
        for i, (msg, idx) in enumerate(zip(session, idxs)):
            speaker    = msg.get("speaker", msg.get("role", ""))
            action_idx = 6 if speaker == "stella" else 0  # stella=idle, arca=responder
            obs        = all_embs[idx]

            z, h, _, _, _ = rssm.step(obs, action_idx=action_idx, h_prev=h)

            # Si el mensaje ACTUAL es de Arca y el SIGUIENTE es de Stella → par
            if speaker in ("arca", "user") and i + 1 < len(session):
                next_msg     = session[i + 1]
                next_speaker = next_msg.get("speaker", next_msg.get("role", ""))
                if next_speaker in ("stella", "assistant"):
                    response = next_msg["content"].strip()
                    if len(response.split()) >= min_words:
                        ctx_vec = np.concatenate([h.numpy().squeeze(), z])  # [320D]
                        pairs.append((ctx_vec, response))

    return pairs


def build_thought_pairs(rssm, chunk_words: int = 80) -> list[tuple[np.ndarray, str]]:
    """
    Carga stella.thoughts.jsonl como pares de entrenamiento adicionales.
    Cada pensamiento → chunks de ~80 palabras → target del decoder.
    Contexto RSSM: idle action sobre el embedding del propio pensamiento.
    """
    path = Path("D:/stella/memory/store/stella.thoughts.jsonl")
    if not path.exists():
        return []

    raw = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [t.get("content", "").strip() for t in raw if len(t.get("content","").split()) >= 5]
    if not texts:
        return []

    print(f"  Codificando {len(texts)} pensamientos en GPU...")
    embs = encode_batch_gpu(texts, batch_size=64)

    pairs: list[tuple[np.ndarray, str]] = []
    for emb, text in zip(embs, texts):
        z, h, _, _, _ = rssm.step(emb, action_idx=6)  # idle — pensamiento interno
        ctx_vec = np.concatenate([h.numpy().squeeze(), z])
        # Dividir pensamiento largo en chunks entrenables
        words = text.split()
        for i in range(0, len(words), chunk_words):
            chunk = " ".join(words[i:i + chunk_words])
            if len(chunk.split()) >= 5:
                pairs.append((ctx_vec, chunk))

    print(f"  Thoughts → {len(pairs)} chunks de entrenamiento")
    return pairs


def build_episode_pairs(rssm) -> list[tuple[np.ndarray, str]]:
    """Carga episodios de memoria como pares adicionales."""
    path = Path("D:/stella/memory/store/stella.episodic")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        episodes = [e.get("content","").strip() for e in data.get("episodes",[])
                    if len(e.get("content","").split()) >= 5]
    except Exception:
        return []

    if not episodes:
        return []

    embs = encode_batch_gpu(episodes, batch_size=64)
    pairs = []
    for emb, text in zip(embs, episodes):
        z, h, _, _, _ = rssm.step(emb, action_idx=4)  # guardar_episodio
        ctx_vec = np.concatenate([h.numpy().squeeze(), z])
        pairs.append((ctx_vec, text))

    print(f"  Episodes → {len(pairs)} pares")
    return pairs


def build_web_pairs(rssm, min_words: int = 4) -> list[tuple[np.ndarray, str]]:
    """
    Carga stella.web.jsonl: los 'triggers' son pensamientos espontáneos de Stella
    que inician una búsqueda web — voz real, modo semi-idle.
    """
    path = Path("D:/stella/memory/store/stella.web.jsonl")
    if not path.exists():
        return []

    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            trigger = entry.get("trigger", "").strip()
            if len(trigger.split()) >= min_words:
                texts.append(trigger)
        except Exception:
            pass

    if not texts:
        return []

    print(f"  Codificando {len(texts)} web-triggers en GPU...")
    embs = encode_batch_gpu(texts, batch_size=64)
    pairs = []
    for emb, text in zip(embs, texts):
        z, h, _, _, _ = rssm.step(emb, action_idx=6)  # idle — pensamiento espontáneo
        ctx_vec = np.concatenate([h.numpy().squeeze(), z])
        pairs.append((ctx_vec, text))

    print(f"  Web triggers → {len(pairs)} pares")
    return pairs


# ─── Tokenización ─────────────────────────────────────────────────────────────

def tokenize_batch(
    texts: list[str],
    tokenizer,
    max_len: int = MAX_SEQ,
) -> tuple[torch.Tensor, torch.Tensor]:
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


# ─── Evaluación ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(decoder, pairs, tokenizer, device, batch_size=8) -> float:
    decoder.eval()
    total, steps = 0.0, 0
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        if not batch:
            break
        ctx_vecs, texts = zip(*batch)
        ctx  = torch.tensor(np.stack(ctx_vecs), dtype=torch.float32).to(device)
        inp, tgt = tokenize_batch(list(texts), tokenizer)
        inp, tgt = inp.to(device), tgt.to(device)
        logits = decoder(inp, ctx)
        loss   = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=-100)
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrenar decoder en la voz de Stella")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=3e-5,
                        help="LR bajo: fine-tune, no olvidar articulación base")
    parser.add_argument("--sample-every",type=int,   default=50)
    parser.add_argument("--from-scratch",action="store_true",
                        help="Ignorar decoder.pt y empezar desde cero")
    args = parser.parse_args()

    from worldmodel.training_logger import setup_logger
    import builtins
    _logger = setup_logger("decoder_stella")
    _orig_print = builtins.print
    builtins.print = lambda *a, **k: _logger.info(" ".join(str(x) for x in a))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  DECODER — Voz de Stella")
    print(f"  Device: {device}")
    print(f"  LR: {args.lr} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print("=" * 60)

    # ── 1. Tokenizer ──────────────────────────────────────────
    print("\n[1/5] Cargando tokenizer GPT-2...")
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab: {tokenizer.vocab_size:,} tokens")

    # ── 2. RSSM entrenado (congelado) ─────────────────────────
    print("\n[2/5] Cargando RSSM entrenado (congelado)...")
    rssm = load_rssm(RSSM_WEIGHTS)
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad_(False)
    print("  RSSM congelado — language_grads=False")

    # ── 3. Dataset: solo Stella ───────────────────────────────
    print("\n[3/5] Cargando voz de Stella...")
    msgs     = load_raw_messages()
    sessions = group_into_sessions(msgs)
    print(f"  {len(msgs)} mensajes → {len(sessions)} sesiones")

    pairs = build_pairs(sessions, rssm)
    print(f"  {len(pairs)} pares de chats")

    pairs += build_thought_pairs(rssm)
    pairs += build_episode_pairs(rssm)
    pairs += build_web_pairs(rssm)
    print(f"  Total pares: {len(pairs)} (chats + thoughts + episodes + web)")

    if len(pairs) < 10:
        print("ERROR: Muy pocos pares. Verifica stella.chats.jsonl")
        sys.exit(1)

    random.shuffle(pairs)
    n_val      = max(1, int(len(pairs) * 0.1))
    val_set    = pairs[:n_val]
    train_set  = pairs[n_val:]
    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")

    # ── 4. Decoder — fine-tune desde base ────────────────────
    print("\n[4/5] Cargando decoder...")
    decoder = SmallDecoder().to(device)

    if not args.from_scratch and DECODER_BASE.exists():
        try:
            decoder.load(DECODER_BASE)
            print(f"  Fine-tune desde {DECODER_BASE} ({decoder.param_count()/1e6:.1f}M params)")
        except Exception as e:
            print(f"  No se pudo cargar decoder.pt ({e}) — iniciando desde cero")
    else:
        print(f"  Desde cero — {decoder.param_count()/1e6:.1f}M params")

    # ── 5. Entrenar ───────────────────────────────────────────
    print(f"\n[5/5] Entrenando en voz de Stella ({args.epochs} épocas)...")
    optimizer = AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01)
    n_steps   = args.epochs * max(1, len(train_set) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    best_val = float("inf")
    step     = 0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_set)
        decoder.train()
        epoch_loss, epoch_steps = 0.0, 0

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
                logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=-100
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
                decoder.eval()
                ctx_s = torch.tensor(
                    val_set[0][0], dtype=torch.float32
                ).unsqueeze(0).to(device)
                ids  = decoder.generate(ctx_s, max_new=60, temperature=0.85, top_k=50)
                text = tokenizer.decode(ids, skip_special_tokens=True)
                print(f"\n  [step {step}] {text[:140]!r}")
                decoder.train()

        avg_train = epoch_loss / max(epoch_steps, 1)
        avg_val   = eval_loss(decoder, val_set, tokenizer, device, args.batch_size)

        marker = ""
        if avg_val < best_val:
            best_val = avg_val
            DECODER_STELLA.parent.mkdir(parents=True, exist_ok=True)
            decoder.save(DECODER_STELLA)
            marker = " ← checkpoint"

        print(f"  Época {epoch:2d}/{args.epochs} | train={avg_train:.4f} | val={avg_val:.4f}{marker}")

    print(f"\n  Mejor val_loss: {best_val:.4f}")
    print(f"  Guardado: {DECODER_STELLA}")
    print(f"  Tamaño:   {DECODER_STELLA.stat().st_size / 1e6:.1f} MB")
    print("\n¡Decoder en voz de Stella completado!")
    print("  Copia decoder_stella.pt → decoder.pt para activarlo en el demo.")
    print("=" * 60)


if __name__ == "__main__":
    main()
