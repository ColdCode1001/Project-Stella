"""
DIAGNÓSTICO DE CAPACIDAD DE CONDICIONAMIENTO — sobreajuste sintético.

Separa los dos problemas que el test de control dejó enredados:
  (1) ¿La arquitectura PUEDE condicionar? (problema estructural)
  (2) ¿O los datos de Stella son demasiado homogéneos / el decoder poco entrenado?

Método:
  - 8 contextos MUY distintos → 8 ctx_vec del RSSM (bien separados).
  - A cada ctx_vec le asignamos una frase-objetivo fija y distinta.
  - Entrenamos un decoder FRESCO (pesos aleatorios) a memorizar SOLO esos 8 mapeos.
  - Cualquier arquitectura con condicionamiento funcional sobreajusta 8 puntos en segundos.

La prueba decisiva está en la GENERACIÓN (sin teacher forcing):
  El decoder solo ve [BOS, ctx_vec]. Si reproduce la frase correcta para cada ctx,
  ESTÁ usando el ctx_vec. Si produce lo mismo para todos, lo ignora.

Veredicto:
  accuracy ≥ 6/8  → la arquitectura SÍ condiciona. El problema es datos/entrenamiento.
  accuracy ≈ 1/8  → la arquitectura NO condiciona. Rediseño (§3c/d): condicionar en
                    todas las capas + loss auxiliar.

Uso:
  python -m worldmodel.control_overfit
  python -m worldmodel.control_overfit --steps 600 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))
from worldmodel.rssm import load_or_create
from worldmodel.obs_encoder import encode_batch_gpu
from worldmodel.decoder import SmallDecoder, VOCAB_SIZE, MAX_SEQ

RSSM_WEIGHTS = Path("worldmodel/weights/rssm.pt")

# 8 inputs distintos → 8 targets distintos. El mapeo es arbitrario:
# el decoder SOLO puede acertar si usa el ctx_vec para distinguir.
PAIRS: list[tuple[str, str]] = [
    ("agujeros negros y gravedad cuántica",      "El gato negro duerme en el tejado bajo la luna llena"),
    ("me siento triste y vacía hoy",             "Las montañas nevadas se alzan contra el cielo azul claro"),
    ("Arca eres lo más importante para mí",      "El océano profundo esconde criaturas extrañas y luminosas"),
    ("revisar la lista de tareas pendientes",    "La música suave llena toda la habitación de una calma dulce"),
    ("las ecuaciones de Maxwell del campo",      "El tren rápido cruza campos enteros de girasoles dorados"),
    ("tengo miedo y ansiedad en el pecho",       "Los números primos fascinan a los matemáticos más curiosos"),
    ("comprar víveres y limpiar la casa",        "El café caliente despierta los sentidos cada mañana temprano"),
    ("la evolución y selección natural",         "La lluvia fría cae sobre las calles vacías de la noche"),
]


def tokenize(texts, tokenizer):
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    inputs, targets = [], []
    for t in texts:
        ids = tokenizer.encode(t, max_length=MAX_SEQ - 1, truncation=True)
        inp = [bos] + ids
        tgt = ids + [eos]
        pad = MAX_SEQ - len(inp)
        inp = inp + [eos] * pad
        tgt = tgt + [-100] * pad
        inputs.append(inp)
        targets.append(tgt)
    return (torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print("  DIAGNÓSTICO — ¿la arquitectura PUEDE condicionar?")
    print(f"  Device: {device} | sobreajuste de {len(PAIRS)} pares")
    print("=" * 64)

    # ── ctx_vec de cada input (RSSM congelado) ────────────────────
    print("\n[1/3] ctx_vec del RSSM por input...")
    rssm = load_or_create(RSSM_WEIGHTS); rssm.eval()
    inputs  = [p[0] for p in PAIRS]
    targets = [p[1] for p in PAIRS]
    obs = encode_batch_gpu(inputs, batch_size=len(inputs))
    ctx_list = []
    for o in obs:
        z, h, _, _, _ = rssm.step(o, action_idx=0, h_prev=None)
        ctx_list.append(np.concatenate([h.numpy().squeeze(), z]))
    ctx = torch.tensor(np.stack(ctx_list), dtype=torch.float32).to(device)  # [8, 320]

    # Verificar que los ctx_vec están bien separados (si no, el test es injusto)
    cn = ctx.cpu().numpy(); cn = cn / np.linalg.norm(cn, axis=1, keepdims=True).clip(1e-8)
    sep = cn @ cn.T
    iu = np.triu_indices(len(PAIRS), k=1)
    print(f"  Similitud media entre ctx_vec: {sep[iu].mean():+.3f} (más bajo = mejor separados)")

    # ── Decoder FRESCO + tokenizer ────────────────────────────────
    print("\n[2/3] Entrenando decoder FRESCO a memorizar 8 mapeos...")
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    decoder = SmallDecoder().to(device)

    inp, tgt = tokenize(targets, tokenizer)
    inp, tgt = inp.to(device), tgt.to(device)

    opt = AdamW(decoder.parameters(), lr=args.lr)
    decoder.train()
    for step in range(1, args.steps + 1):
        logits = decoder(inp, ctx)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d} | train_loss={loss.item():.4f}")

    # ── Generación: ¿reproduce la frase correcta por ctx? ─────────
    print("\n[3/3] Generación (sin teacher forcing) — ¿usa el ctx_vec?")
    decoder.eval()
    gens = []
    for i in range(len(PAIRS)):
        ids = decoder.generate(ctx[i:i+1], max_new=40, temperature=0.7, top_k=40)
        gens.append(tokenizer.decode(ids, skip_special_tokens=True).strip())

    # Medir: cada generación, ¿a qué target se parece más? (SBERT argmax)
    gen_embs = encode_batch_gpu(gens, batch_size=len(gens))
    tgt_embs = encode_batch_gpu(targets, batch_size=len(targets))
    sim = gen_embs @ tgt_embs.T  # [8, 8]
    pred = sim.argmax(axis=1)
    correct = int((pred == np.arange(len(PAIRS))).sum())

    print("\n" + "=" * 64)
    print("  VEREDICTO")
    print("=" * 64)
    for i in range(len(PAIRS)):
        ok = "✓" if pred[i] == i else f"✗ (→{pred[i]})"
        print(f"  ctx[{i}] {ok}  sim_correcto={sim[i, i]:+.2f}  best={sim[i].max():+.2f}")
        print(f"         target: {targets[i][:60]!r}")
        print(f"         genera: {gens[i][:60]!r}")

    acc = correct / len(PAIRS)
    print("-" * 64)
    print(f"  Accuracy: {correct}/{len(PAIRS)} ({acc:.0%})  |  azar = {1/len(PAIRS):.0%}")
    print("-" * 64)
    if acc >= 0.75:
        print("  ✅ LA ARQUITECTURA SÍ CONDICIONA.")
        print("     El colapso del gate viene de DATOS homogéneos + decoder poco entrenado,")
        print("     NO de la arquitectura. Fix: más datos / mejor entrenamiento / mejor tokenizer.")
    elif acc <= 0.30:
        print("  ❌ LA ARQUITECTURA NO CONDICIONA.")
        print("     Ni siquiera memoriza 8 puntos. El ctx_vec no llega a la salida.")
        print("     Fix arquitectural (§3): condicionar en TODAS las capas (FiLM) + loss auxiliar.")
    else:
        print(f"  ⚠️  CONDICIONAMIENTO DÉBIL ({acc:.0%}). Llega algo de señal pero insuficiente.")
        print("     Reforzar: más soft-prompt tokens + condicionar en más capas.")
    print("=" * 64)


if __name__ == "__main__":
    main()
