"""
AUDITORÍA DE DATOS — ¿los datos de Stella tienen señal ctx→respuesta?

El diagnóstico (control_overfit) probó que la arquitectura SÍ condiciona.
El gate falló porque el decoder aprendió la voz MARGINAL de Stella e ignoró el ctx.
Hipótesis: las respuestas de Stella no dependen lo suficiente de su contexto/estado
→ no hay nada que el WM pueda controlar.

Esta auditoría lo mide ANTES de reentrenar, sobre los mismos pares (ctx_vec, respuesta)
que usa el entrenamiento:

  1. DIVERSIDAD de respuestas: ¿Stella dice cosas variadas o siempre lo mismo?
     - distancia media entre respuestas + tasa de near-duplicados
  2. DIVERSIDAD de ctx_vec: ¿el RSSM produce estados distintos por contexto?
  3. CORRELACIÓN ctx↔respuesta (Mantel): ¿contextos similares → respuestas similares?
     Esta es la métrica madre. Si ~0, no hay estructura learnable: el WM no puede
     controlar porque los datos no tienen dependencia ctx→respuesta.
  4. BASELINE barajado: la correlación con respuestas barajadas debe dar ~0 (sanity).

Veredicto:
  Mantel > 0.20  → HAY señal ctx→respuesta. Reentrenar (loss aux / más épocas) debería ayudar.
  Mantel ≈ 0     → los datos NO soportan control del WM. Replantear datos, no el entrenamiento.

Uso:
  python -m worldmodel.audit_data
  python -m worldmodel.audit_data --source all   (chats + thoughts + web)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from worldmodel.rssm import load_or_create
from worldmodel.obs_encoder import encode_batch_gpu

RSSM_WEIGHTS = Path("worldmodel/weights/rssm.pt")


def mantel_corr(sim_a: np.ndarray, sim_b: np.ndarray) -> float:
    """Correlación de Pearson entre las dos matrices de similitud (off-diagonal)."""
    n = sim_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = sim_a[iu], sim_b[iu]
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["chats", "all"], default="chats")
    args = parser.parse_args()

    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 64)
    print("  AUDITORÍA DE DATOS — ¿hay señal ctx→respuesta?")
    print(f"  Fuente: {args.source}")
    print("=" * 64)

    # ── Reusar EXACTAMENTE los pares del entrenamiento ────────────
    print("\n[1/4] Construyendo pares (ctx_vec, respuesta)...")
    rssm = load_or_create(RSSM_WEIGHTS); rssm.eval()

    from worldmodel.train_decoder_stella import (
        load_raw_messages, group_into_sessions, build_pairs,
        build_thought_pairs, build_web_pairs,
    )
    msgs = load_raw_messages()
    sessions = group_into_sessions(msgs)
    pairs = build_pairs(sessions, rssm)
    if args.source == "all":
        pairs += build_thought_pairs(rssm)
        pairs += build_web_pairs(rssm)

    if len(pairs) < 10:
        print("ERROR: muy pocos pares.")
        sys.exit(1)

    ctx_vecs  = np.stack([p[0] for p in pairs]).astype(np.float32)   # [N, 320]
    responses = [p[1] for p in pairs]
    print(f"  {len(pairs)} pares")

    # ── Embeddings ────────────────────────────────────────────────
    print("\n[2/4] Codificando respuestas...")
    resp_embs = encode_batch_gpu(responses, batch_size=256)          # [N, 128] normalizados

    # Normalizar ctx_vecs para cosine
    ctx_norm = ctx_vecs / np.linalg.norm(ctx_vecs, axis=1, keepdims=True).clip(1e-8)

    # ── Matrices de similitud ─────────────────────────────────────
    print("\n[3/4] Calculando similitudes y correlación...")
    ctx_sim  = ctx_norm  @ ctx_norm.T
    resp_sim = resp_embs @ resp_embs.T
    iu = np.triu_indices(len(pairs), k=1)

    resp_div = float(np.mean(1.0 - resp_sim[iu]))   # diversidad de respuestas
    ctx_div  = float(np.mean(1.0 - ctx_sim[iu]))    # diversidad de ctx
    near_dup = float(np.mean(resp_sim[iu] > 0.90))  # tasa near-duplicados

    mantel = mantel_corr(ctx_sim, resp_sim)

    # Baseline barajado (debe dar ~0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pairs))
    mantel_shuf = mantel_corr(ctx_sim, resp_sim[perm][:, perm])

    # ── Veredicto ─────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  VEREDICTO")
    print("=" * 64)
    print(f"  Diversidad de respuestas (dist media):  {resp_div:.4f}  (alto=variado)")
    print(f"  Tasa de near-duplicados (sim>0.90):      {near_dup:.1%}")
    print(f"  Diversidad de ctx_vec (dist media):      {ctx_div:.4f}  (alto=RSSM diferencia)")
    print("-" * 64)
    print(f"  Correlación Mantel ctx↔respuesta:        {mantel:+.4f}  ← MÉTRICA MADRE")
    print(f"  Baseline barajado (debe ser ~0):         {mantel_shuf:+.4f}")
    print("-" * 64)

    # Criterios
    c_signal   = mantel > 0.20
    c_weaksig  = 0.08 < mantel <= 0.20
    c_respdiv  = resp_div > 0.35
    c_ctxdiv   = ctx_div > 0.10

    if c_signal:
        print("  ✅ HAY SEÑAL ctx→respuesta. Los datos soportan control del WM.")
        print("     Reentrenar con loss auxiliar / más épocas debería hacer pasar el gate.")
    elif c_weaksig:
        print("  ⚠️  SEÑAL DÉBIL. Hay algo de estructura pero poca.")
        print("     El loss auxiliar puede ayudar, pero conviene también diversificar datos.")
    else:
        print("  ❌ SIN SEÑAL ctx→respuesta. Los datos NO soportan control del WM.")
        print("     Stella responde casi independiente de su contexto/estado.")
        print("     Ningún entrenamiento hará que el WM controle. Hay que replantear:")
        print("     - ¿el ctx_vec captura lo que debería? (¿RSSM diferencia contextos?)")
        print("     - ¿los datos tienen respuestas que dependan del estado?")

    if not c_respdiv:
        print(f"\n  ⚠️  Respuestas poco diversas ({resp_div:.3f}) — Stella habla muy homogéneo.")
    if not c_ctxdiv:
        print(f"  ⚠️  ctx_vec poco diversos ({ctx_div:.3f}) — el RSSM no diferencia contextos.")
    print("=" * 64)

    # Muestras de respuestas más comunes (near-dup clusters)
    print("\n  Respuestas más 'centrales' (cercanas al promedio = genéricas):")
    centroid = resp_embs.mean(0); centroid /= np.linalg.norm(centroid).clip(1e-8)
    central = (resp_embs @ centroid)
    for idx in np.argsort(central)[::-1][:5]:
        print(f"  [{central[idx]:+.2f}] {responses[idx][:75]!r}")


if __name__ == "__main__":
    main()
