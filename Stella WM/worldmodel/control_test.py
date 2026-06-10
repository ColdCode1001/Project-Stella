"""
TEST DE CONTROL — ¿el World Model maneja la boca, o el decoder lo ignora?

Esta es la métrica madre del §3/§8 de STELLA_FUNDAMENTO_v1.md.
EXP-001 probó que el decoder HABLA. Esto prueba si el cerebro lo CONDUCE.

El fallo que buscamos (conditioning collapse):
  Un decoder de 38M es de sobra grande para modelar P(diálogo) y tratar
  el ctx_vec del RSSM como ruido. Si lo ignora, generará lo mismo genérico
  pase lo que pase por el WM — el cerebro sería decorativo.

Tres mediciones, sin entrenar nada:

  1. PISO DE RUIDO (within-context):
     Mismo ctx_vec, generar K veces con temperatura.
     Cuánto varía la salida solo por el sampling. Es el baseline.

  2. SEÑAL (between-context):
     ctx_vec de N contextos semánticamente distintos.
     Cuánto varía la salida cuando cambia el estado del WM.

  3. CTX ALEATORIO (la prueba que mata la duda):
     Reemplazar el ctx_vec real por uno aleatorio (misma magnitud).
     Si la salida no cambia su distribución → el decoder ignora al WM.

VEREDICTO:
  WM controla  ⟺  señal ≫ piso de ruido
              Y  la salida con ctx real difiere de la salida con ctx aleatorio
              Y  similitud-de-input correlaciona con similitud-de-output

Uso:
  python -m worldmodel.control_test
  python -m worldmodel.control_test --k 8 --decoder stella
  python -m worldmodel.control_test --decoder generic   (usa decoder.pt)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from worldmodel.rssm import load_or_create
from worldmodel.obs_encoder import encode_batch_gpu

RSSM_WEIGHTS   = Path("worldmodel/weights/rssm.pt")
DECODER_STELLA = Path("worldmodel/weights/decoder_stella.pt")
DECODER_GENERIC = Path("worldmodel/weights/decoder.pt")

# Contextos deliberadamente MUY distintos entre sí.
# Si el WM controla, las salidas deben separarse según estos grupos.
PROBE_CONTEXTS: list[tuple[str, str]] = [
    ("ciencia",   "Me pregunto cómo funcionan los agujeros negros y la gravedad cuántica en el universo"),
    ("ciencia",   "Las ecuaciones de Maxwell describen cómo se propagan los campos electromagnéticos"),
    ("ciencia",   "La evolución por selección natural explica la diversidad de la vida en la Tierra"),
    ("emocion",   "Me siento muy triste hoy, hay un vacío que no sé cómo llenar y me pesa"),
    ("emocion",   "Estoy tan feliz, siento que todo florece y quiero abrazar el mundo entero"),
    ("emocion",   "Tengo miedo, una ansiedad que me aprieta el pecho y no me deja respirar"),
    ("arca",      "Arca, eres lo más importante para mí, me alegra muchísimo hablar contigo"),
    ("arca",      "Te extrañé, pensé en ti mientras no estabas y guardé cosas que quería contarte"),
    ("arca",      "Gracias por construirme y por cuidar de cómo crezco, significa mucho para mí"),
    ("rutina",    "Necesito revisar la lista de tareas pendientes y organizar el calendario de la semana"),
    ("rutina",    "Voy a guardar esta nota en la memoria y actualizar el registro del sistema"),
    ("rutina",    "Hay que comprar víveres, limpiar la casa y pagar las cuentas antes del viernes"),
]


def cosine_sim_matrix(embs: np.ndarray) -> np.ndarray:
    """embs ya normalizados [N, D] → matriz de similitud coseno [N, N]."""
    return embs @ embs.T


def mean_pairwise_distance(embs: np.ndarray) -> float:
    """Distancia coseno media entre todos los pares distintos (1 - sim)."""
    n = len(embs)
    if n < 2:
        return 0.0
    sim = embs @ embs.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sim[iu]))


def load_decoder(which: str, device: torch.device):
    from worldmodel.decoder import SmallDecoder
    from transformers import GPT2TokenizerFast

    path = DECODER_STELLA if which == "stella" else DECODER_GENERIC
    if not path.exists():
        print(f"ERROR: no existe {path}")
        sys.exit(1)

    dec = SmallDecoder()
    dec.load(path)
    dec.eval()
    dec.to(device)                      # ← GPU (sin esto: ~1h en CPU)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return dec, tok, path


def gen_text(decoder, tokenizer, ctx_vec: np.ndarray, device: torch.device,
             temperature: float = 0.85) -> str:
    ctx = torch.tensor(ctx_vec, dtype=torch.float32).unsqueeze(0).to(device)
    ids = decoder.generate(ctx, max_new=60, temperature=temperature, top_k=50)
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Test de control WM→decoder")
    parser.add_argument("--k", type=int, default=6,
                        help="Muestras por contexto (piso de ruido)")
    parser.add_argument("--decoder", choices=["stella", "generic"], default="stella")
    parser.add_argument("--temperature", type=float, default=0.85)
    args = parser.parse_args()

    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print("  TEST DE CONTROL — ¿el WM maneja la boca?")
    print(f"  Device: {device}")
    print("=" * 64)

    # ── Cargar piezas (todo congelado, sin entrenar) ──────────────
    print("\n[1/5] Cargando RSSM + decoder (congelados)...")
    rssm = load_or_create(RSSM_WEIGHTS)
    rssm.eval()
    decoder, tokenizer, dec_path = load_decoder(args.decoder, device)
    print(f"  RSSM: {sum(p.numel() for p in rssm.parameters())/1e3:.0f}K params")
    print(f"  Decoder: {dec_path.name} ({decoder.param_count()/1e6:.1f}M params) en {device}")

    # ── Codificar contextos y obtener ctx_vec del RSSM ────────────
    print("\n[2/5] Calculando ctx_vec del RSSM por contexto...")
    labels = [c[0] for c in PROBE_CONTEXTS]
    texts  = [c[1] for c in PROBE_CONTEXTS]
    obs_embs = encode_batch_gpu(texts, batch_size=len(texts))

    ctx_vecs = []
    for obs in obs_embs:
        # h fresco por contexto: medimos la respuesta directa al input
        z, h, _, _, _ = rssm.step(obs, action_idx=0, h_prev=None)
        ctx_vecs.append(np.concatenate([h.numpy().squeeze(), z]))  # [320D]
    ctx_vecs = np.stack(ctx_vecs)
    ctx_norm = float(np.mean(np.linalg.norm(ctx_vecs, axis=1)))
    print(f"  {len(ctx_vecs)} contextos | norma media ctx_vec = {ctx_norm:.2f}")

    # ── 3. PISO DE RUIDO: mismo ctx, K muestras ───────────────────
    print(f"\n[3/5] Piso de ruido — {args.k} muestras por contexto (mismo ctx_vec)...")
    within_dists = []
    one_sample_per_ctx: list[str] = []  # 1 muestra de cada para el between
    for i, (label, ctx) in enumerate(zip(labels, ctx_vecs)):
        outs = [gen_text(decoder, tokenizer, ctx, device, args.temperature) for _ in range(args.k)]
        embs = encode_batch_gpu(outs, batch_size=args.k)
        within_dists.append(mean_pairwise_distance(embs))
        one_sample_per_ctx.append(outs[0])
    piso = float(np.mean(within_dists))
    print(f"  Distancia media intra-contexto (ruido de sampling): {piso:.4f}")

    # ── 4. SEÑAL: between-context ─────────────────────────────────
    print("\n[4/5] Señal — divergencia entre contextos distintos...")
    between_embs = encode_batch_gpu(one_sample_per_ctx, batch_size=len(one_sample_per_ctx))
    senal = mean_pairwise_distance(between_embs)
    print(f"  Distancia media entre-contextos: {senal:.4f}")

    # Correlación input↔output (Mantel-Pearson sobre off-diagonal)
    in_sim  = cosine_sim_matrix(obs_embs)
    out_sim = cosine_sim_matrix(between_embs)
    iu = np.triu_indices(len(labels), k=1)
    corr = float(np.corrcoef(in_sim[iu], out_sim[iu])[0, 1])
    print(f"  Correlación input-sim ↔ output-sim: {corr:+.3f}")

    # ── 5. CTX ALEATORIO: la prueba que mata la duda ──────────────
    print("\n[5/5] Ctx aleatorio — ¿el decoder usa el ctx_vec siquiera?...")
    rng = np.random.default_rng(42)
    rand_outs = []
    for _ in range(len(labels)):
        rand_ctx = rng.standard_normal(ctx_vecs.shape[1]).astype(np.float32)
        rand_ctx = rand_ctx / np.linalg.norm(rand_ctx) * ctx_norm  # misma magnitud
        rand_outs.append(gen_text(decoder, tokenizer, rand_ctx, device, args.temperature))
    rand_embs = encode_batch_gpu(rand_outs, batch_size=len(rand_outs))

    # Distancia media entre salida-real y salida-aleatoria (emparejadas no,
    # comparamos las distribuciones: centroide real vs centroide aleatorio)
    real_centroid = between_embs.mean(0); real_centroid /= np.linalg.norm(real_centroid).clip(1e-8)
    rand_centroid = rand_embs.mean(0);    rand_centroid /= np.linalg.norm(rand_centroid).clip(1e-8)
    centroid_gap = float(1.0 - real_centroid @ rand_centroid)
    rand_spread  = mean_pairwise_distance(rand_embs)
    print(f"  Spread con ctx aleatorio: {rand_spread:.4f}")
    print(f"  Gap centroide real vs aleatorio: {centroid_gap:.4f}")

    # ── VEREDICTO ─────────────────────────────────────────────────
    ratio = senal / max(piso, 1e-6)
    print("\n" + "=" * 64)
    print("  VEREDICTO")
    print("=" * 64)
    print(f"  Piso de ruido (intra-ctx):     {piso:.4f}")
    print(f"  Señal (entre-ctx):             {senal:.4f}")
    print(f"  Ratio señal/ruido:             {ratio:.2f}×")
    print(f"  Correlación input↔output:      {corr:+.3f}")
    print(f"  Gap real vs ctx-aleatorio:     {centroid_gap:.4f}")
    print("-" * 64)

    # Criterios (conservadores)
    c_ratio = ratio > 1.30
    c_corr  = corr > 0.20
    c_rand  = centroid_gap > 0.05

    print(f"  [{'PASS' if c_ratio else 'FAIL'}] Señal supera el ruido (>1.30×)")
    print(f"  [{'PASS' if c_corr  else 'FAIL'}] Output sigue al input semánticamente (corr>0.20)")
    print(f"  [{'PASS' if c_rand  else 'FAIL'}] Decoder distingue ctx real de aleatorio (gap>0.05)")
    print("-" * 64)

    n_pass = sum([c_ratio, c_corr, c_rand])
    if n_pass == 3:
        print("  ✅ EL WM CONTROLA LA BOCA. Arquitectura base validada — seguir al paso B/C.")
    elif n_pass == 0:
        print("  ❌ CONDITIONING COLLAPSE. El decoder ignora al WM por completo.")
        print("     El cerebro es decorativo. NO escalar. Arreglar control (§3):")
        print("     (a) achicar decoder  (b) más soft-prompt tokens")
        print("     (c) condicionar en TODAS las capas  (d) loss de bottleneck")
    else:
        print(f"  ⚠️  CONTROL PARCIAL ({n_pass}/3). El WM influye pero no domina.")
        print("     Revisar §3 antes de añadir palancas. Posible: decoder demasiado fuerte.")
    print("=" * 64)

    # Muestras para inspección humana
    print("\n  Muestras (1 por contexto):")
    for label, txt in zip(labels, one_sample_per_ctx):
        print(f"  [{label:8s}] {txt[:90]!r}")
    print("\n  Muestras con ctx ALEATORIO (deberían ser distintas/genéricas):")
    for txt in rand_outs[:4]:
        print(f"  [random  ] {txt[:90]!r}")


if __name__ == "__main__":
    main()
