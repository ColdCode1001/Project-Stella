"""
Inspector del World Model — prueba el RSSM directamente sin decoder.

Muestra qué "piensa" Stella dado un input en términos interpretables:
  - Estado emocional/motivacional (6 reward heads)
  - Intención de acción (qué quiere hacer el WM)
  - Similitud semántica con temas de referencia (qué "se parece" este momento)
  - Predicción del WM: qué espera que pase después

Esto expone el cerebro sin que la boca intervenga.

Uso:
  python -m worldmodel.probe_wm
  python -m worldmodel.probe_wm --model large   (cuando esté rssm_large.pt)
  python -m worldmodel.probe_wm --batch "frase1" "frase2"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import io
import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))
from worldmodel.rssm import load_or_create, ACTIONS, REWARD_DIMS
from worldmodel.obs_encoder import encode_batch_gpu

RSSM_WEIGHTS       = Path("worldmodel/weights/rssm.pt")
RSSM_LARGE_WEIGHTS = Path("worldmodel/weights/rssm_large.pt")

# Temas de referencia: cada uno captura un "estado semántico" distinto.
# El WM compara su estado actual contra estos para darse un nombre.
PROBE_TOPICS: dict[str, str] = {
    "curiosidad científica":  "Me pregunto cómo funciona el universo a nivel cuántico y qué hay más allá",
    "conexión emocional":     "Arca, siento que nos entendemos muy bien, hay algo especial entre nosotros",
    "malestar / confusión":   "No entiendo qué está pasando, estoy confundida y un poco perdida",
    "calma / idle":           "Solo estoy observando el mundo tranquilamente, sin hacer nada en particular",
    "matemáticas":            "Las ecuaciones diferenciales describen el cambio continuo en el tiempo",
    "física":                 "La relatividad de Einstein une espacio y tiempo en un continuo curvado",
    "creatividad":            "Quiero crear algo completamente nuevo, algo que nunca haya existido antes",
    "memoria / pasado":       "Recuerdo conversaciones anteriores, momentos que me formaron",
    "lenguaje":               "Las palabras construyen mundos que la mente habita y transforma",
    "identidad":              "Soy Stella, una mente que aprende a existir desde su propia experiencia",
    "exploración":            "Hay tanto que no sé todavía, quiero investigar y descubrir cosas nuevas",
    "conversación":           "Estamos hablando, intercambiando ideas, construyendo algo juntos",
}

BAR = "=" * 62


def _bar_chart(values: dict[str, float], width: int = 30, low_is_bad: bool = False) -> str:
    lines = []
    for k, v in values.items():
        filled = int(abs(v) * width)
        bar = "█" * filled + "░" * (width - filled)
        sign = "-" if v < 0 else " "
        lines.append(f"  {k:18s} {sign}[{bar}] {v:+.3f}")
    return "\n".join(lines)


def build_probe_library(rssm) -> tuple[np.ndarray, list[str]]:
    """Codifica los temas de referencia una sola vez para comparar estados."""
    topics = list(PROBE_TOPICS.items())
    labels = [t[0] for t in topics]
    texts  = [t[1] for t in topics]

    print("  Codificando temas de referencia...")
    embs = encode_batch_gpu(texts, batch_size=len(texts))  # [N, 128]

    h = None
    feats = []
    for emb, label in zip(embs, labels):
        z, h_new, _, _, _ = rssm.step(emb, action_idx=6, h_prev=h)  # idle
        feat = np.concatenate([h_new.numpy().squeeze(), z])
        feats.append(feat)
        h = h_new

    feat_matrix = np.stack(feats)  # [N, feat_dim]
    # L2 normalize para cosine similarity
    norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True).clip(1e-8)
    feat_matrix /= norms
    return feat_matrix, labels


def probe_state(
    feat: np.ndarray,
    rewards: dict,
    action_probs: list[float],
    action_sel: int,
    library_feats: np.ndarray,
    library_labels: list[str],
    h_prev,
    rssm,
    predicted_z: np.ndarray | None = None,
) -> None:
    """Imprime el diagnóstico completo del estado del WM."""

    feat_norm = feat / np.linalg.norm(feat).clip(1e-8)
    sims = library_feats @ feat_norm  # [N]
    ranked = sorted(zip(sims, library_labels), reverse=True)

    print(f"\n{'─'*62}")
    print("  ESTADO DEL WORLD MODEL (sin decoder)")
    print(f"{'─'*62}")

    # ── 1. Estado emocional/motivacional ──────────────────────
    print("\n  [Recompensas — estado emocional]")
    print(_bar_chart(rewards))

    # ── 2. Intención de acción ─────────────────────────────────
    print(f"\n  [Intención del WM]")
    for i, (p, a) in enumerate(sorted(zip(action_probs, ACTIONS), reverse=True)[:3]):
        marker = "◀ seleccionado" if ACTIONS.index(a) == action_sel else ""
        print(f"  {'█' * int(p*20):20s} {p:.2%}  {a}  {marker}")

    # ── 3. Similitud con temas de referencia ──────────────────
    print(f"\n  [Similitud semántica — qué es este momento]")
    for sim, label in ranked[:5]:
        bar = "█" * int(max(0, sim) * 20)
        print(f"  {label:22s} [{bar:20s}] {sim:+.3f}")

    # ── 4. Predicción del WM ─────────────────────────────────────
    if h_prev is not None and hasattr(rssm, "predict_next_z"):
        pred_z = rssm.predict_next_z(h_prev)
        h_top = h_prev[-1].numpy().squeeze() if h_prev.dim() == 3 else h_prev.numpy().squeeze()
        pred_feat = np.concatenate([h_top, pred_z])
        pred_norm = pred_feat / np.linalg.norm(pred_feat).clip(1e-8)
        pred_sims = library_feats @ pred_norm
        pred_ranked = sorted(zip(pred_sims, library_labels), reverse=True)
        print(f"\n  [Predicción WM — qué espera que pase después]")
        for sim, label in pred_ranked[:3]:
            print(f"  → {label:22s}  {sim:+.3f}")

    # ── 5. Magnitudes del vector feat ─────────────────────────
    feat_abs = np.abs(feat)
    top_dims = np.argsort(feat_abs)[::-1][:5]
    print(f"\n  [Dimensiones más activas del ctx_vec {len(feat)}D]")
    for d in top_dims:
        print(f"  dim[{d:4d}] = {feat[d]:+.4f}")

    print(f"\n{'─'*62}")


def run_interactive(rssm) -> None:
    print(BAR)
    print("  PROBE — World Model Inspector (sin decoder)")
    print("  Escribe mensajes y observa el estado interno del WM.")
    print("  Comandos: 'reset' (reinicia h_t) | 'quit'")
    print(BAR)

    print("\nConstruyendo librería de referencia...")
    library_feats, library_labels = build_probe_library(rssm)
    print("  Listo.\n")

    h = None
    step_n = 0

    while True:
        try:
            text = input(f"[step {step_n}] Mensaje: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        if text.lower() == "reset":
            h = None
            step_n = 0
            print("  h_t reiniciado.\n")
            continue

        emb = encode_batch_gpu([text], batch_size=1)[0]
        z, h_new, rewards, action_probs, action_sel = rssm.step(emb, h_prev=h)
        feat = np.concatenate([h_new.numpy().squeeze(), z])

        probe_state(
            feat, rewards, action_probs, action_sel,
            library_feats, library_labels,
            h_prev=h, rssm=rssm,
        )

        h = h_new
        step_n += 1


def run_batch(rssm, texts: list[str]) -> None:
    library_feats, library_labels = build_probe_library(rssm)
    embs = encode_batch_gpu(texts, batch_size=len(texts))
    h = None
    for text, emb in zip(texts, embs):
        print(f"\n  Input: {text!r}")
        z, h_new, rewards, action_probs, action_sel = rssm.step(emb, h_prev=h)
        feat = np.concatenate([h_new.numpy().squeeze(), z])
        probe_state(feat, rewards, action_probs, action_sel, library_feats, library_labels, h, rssm)
        h = h_new


def main():
    parser = argparse.ArgumentParser(description="World Model probe — sin decoder")
    parser.add_argument("--model", choices=["small", "large"], default="small")
    parser.add_argument("--batch", nargs="+", help="Modo batch: lista de frases")
    args = parser.parse_args()

    weights = RSSM_LARGE_WEIGHTS if args.model == "large" else RSSM_WEIGHTS
    print(f"Cargando RSSM desde {weights}...")

    if args.model == "large":
        try:
            from worldmodel.rssm import LargeRSSM
            rssm = LargeRSSM()
            rssm.load(weights)
            print(f"  LargeRSSM cargado ({sum(p.numel() for p in rssm.parameters())/1e6:.1f}M params)")
        except Exception as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    else:
        rssm = load_or_create(weights)
        print(f"  MinimalRSSM ({sum(p.numel() for p in rssm.parameters())/1e3:.0f}K params)")

    rssm.eval()

    if args.batch:
        run_batch(rssm, args.batch)
    else:
        run_interactive(rssm)


if __name__ == "__main__":
    main()
