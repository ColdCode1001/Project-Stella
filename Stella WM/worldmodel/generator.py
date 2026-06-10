"""
Generador de texto basado puramente en el World Model — sin LLM.

Funcionamiento:
  1. Al iniciar, carga todas las respuestas históricas de Stella (stella.chats.jsonl)
     y las codifica con sentence-transformers.
  2. Cuando llega un mensaje nuevo:
     a. El RSSM genera el estado latente z_t y el vector de recompensas.
     b. Se buscan las K respuestas más similares semánticamente al mensaje actual.
     c. El estado latente del RSSM reordena esos candidatos (bias por recompensas).
     d. Se devuelve la respuesta ganadora.

Esto es "WM-driven retrieval": el WM no genera tokens, navega un espacio
de respuestas reales. Conforme el RSSM se entrena, la selección mejora.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

STELLA_CHATS = Path("D:/stella/memory/store/stella.chats.jsonl")
STELLA_THOUGHTS = Path("D:/stella/memory/store/stella.thoughts.jsonl")
TOP_K = 8  # candidatos a considerar antes de reranking por WM

# ─── Decoder (EXP-001) ────────────────────────────────────────────────────────
# Se carga una sola vez si los pesos existen, None si no hay decoder entrenado
_decoder = None
_tokenizer = None

def _try_load_decoder():
    """Intenta cargar decoder + tokenizer. Prefiere decoder_stella.pt (voz de Stella)."""
    global _decoder, _tokenizer
    if _decoder is not None:
        return True
    try:
        from worldmodel.decoder import SmallDecoder
        from transformers import GPT2TokenizerFast
        from pathlib import Path

        # Prioridad: voz de Stella > decoder genérico
        stella_path  = Path("worldmodel/weights/decoder_stella.pt")
        generic_path = Path("worldmodel/weights/decoder.pt")

        if stella_path.exists():
            dec = SmallDecoder()
            dec.load(stella_path)
            label = "decoder_stella (voz Stella)"
        elif generic_path.exists():
            dec = SmallDecoder()
            dec.load(generic_path)
            label = "decoder (genérico)"
        else:
            return False

        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _tokenizer.pad_token = _tokenizer.eos_token
        _decoder = dec.eval()
        print(f"[generator] {label} cargado — generación neuronal activa")
        return True
    except Exception:
        return False


class WMGenerator:
    """
    Generador de respuestas conducido por el World Model.
    Sin LLM — solo embeddings + RSSM state.
    """

    def __init__(self):
        self._pairs: list[tuple[str, str]] = []        # (user_msg, stella_response)
        self._user_embs: np.ndarray | None = None      # [N, OBS_DIM]
        self._stella_embs: np.ndarray | None = None    # [N, OBS_DIM]
        self._loaded = False

    def load(self):
        """Carga y codifica el historial de Stella."""
        from worldmodel.obs_encoder import encode_observation

        print("[generator] Cargando historial de Stella...")

        # ── Pares conversacionales de chats ──────────────────────────────
        pairs: list[tuple[str, str]] = []
        if STELLA_CHATS.exists():
            msgs = []
            for line in STELLA_CHATS.read_text(encoding="utf-8").splitlines():
                try:
                    msgs.append(json.loads(line))
                except Exception:
                    pass
            # Emparejar user → assistant consecutivos
            for i in range(len(msgs) - 1):
                if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                    u = msgs[i].get("content", "").strip()
                    s = msgs[i + 1].get("content", "").strip()
                    if u and s and len(s) > 20:
                        pairs.append((u, s))

        # ── Pensamientos idle como "respuestas sin pregunta" ──────────────
        if STELLA_THOUGHTS.exists():
            for line in STELLA_THOUGHTS.read_text(encoding="utf-8").splitlines():
                try:
                    t = json.loads(line)
                    content = t.get("content", "").strip()
                    if content and len(content) > 30:
                        # El "user msg" es vacío → matchea con mensajes cortos/saludos
                        pairs.append((".", content))
                except Exception:
                    pass

        if not pairs:
            print("[generator] ADVERTENCIA: No se encontraron pares de entrenamiento.")
            self._loaded = True
            return

        print(f"[generator] Codificando {len(pairs)} pares con sentence-transformers...")
        user_embs = []
        stella_embs = []
        valid_pairs = []

        for i, (u, s) in enumerate(pairs):
            try:
                ue = encode_observation(u, session_length=0)
                se = encode_observation(s, session_length=0)
                user_embs.append(ue)
                stella_embs.append(se)
                valid_pairs.append((u, s))
            except Exception:
                pass
            if i % 50 == 0:
                print(f"  {i}/{len(pairs)}...", end="\r")

        print(f"\n[generator] {len(valid_pairs)} pares listos.")
        self._pairs = valid_pairs
        self._user_embs = np.stack(user_embs)    # [N, OBS_DIM]
        self._stella_embs = np.stack(stella_embs)
        self._loaded = True

    def generate(
        self,
        user_message: str,
        rewards: dict,
        z_vector: np.ndarray | None = None,
        session_history: list[dict] | None = None,
        h_state: "torch.Tensor | None" = None,
    ) -> tuple[str, dict]:
        """
        Genera una respuesta usando el estado del WM.

        Returns:
            response: texto de respuesta
            meta: dict con info de debug (similitud, candidatos, etc.)
        """
        from worldmodel.obs_encoder import encode_observation

        if not self._loaded:
            self.load()

        # ── Decoder path (EXP-001) — si hay pesos entrenados ─────────────
        if h_state is not None and z_vector is not None and _try_load_decoder():
            try:
                ctx_np = np.concatenate([
                    h_state.numpy().squeeze(),  # [256]
                    z_vector,                   # [64]
                ])                              # [320]
                ctx = torch.tensor(ctx_np, dtype=torch.float32).unsqueeze(0)
                token_ids = _decoder.generate(ctx, max_new=80, temperature=0.85)
                text = _tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                if len(text.split()) >= 3:
                    meta = {"mode": "decoder", "tokens": len(token_ids)}
                    return text, meta
            except Exception as e:
                print(f"[generator] Decoder error: {e} — usando retrieval")

        if not self._pairs or self._user_embs is None:
            return (
                "[WM sin datos suficientes — ejecuta primero: python -m worldmodel.pretrain]",
                {}
            )

        # ── 1. Embedding del mensaje actual ──────────────────────────────
        query_emb = encode_observation(
            user_message,
            session_length=len(session_history) if session_history else 0,
            last_reward=rewards,
        )

        # ── 2. Similitud coseno con todos los user_msgs históricos ───────
        # user_embs: [N, D], query: [D]
        sims = self._user_embs @ query_emb  # [N] — ya están normalizados
        top_k_idx = np.argsort(sims)[::-1][:TOP_K]

        # ── 3. Reranking por estado del WM ───────────────────────────────
        # El RSSM tiene un vector de recompensas que bias la selección:
        # - Alta curiosidad → preferir respuestas más largas/elaboradas
        # - Alta conexión   → preferir respuestas más cálidas/personales
        # - Alta satisfacción → preferir respuestas que consoliden info
        # - Malestar bajo   → evitar respuestas que repitan predicciones fallidas

        scores = []
        for idx in top_k_idx:
            u, s = self._pairs[idx]
            base_score = float(sims[idx])

            # Bias por recompensas del WM
            length_factor = min(len(s) / 400.0, 1.0)  # normalizado
            warmth_factor = _warmth_score(s)
            density_factor = min(len(s.split()) / 60.0, 1.0)

            wm_bias = (
                rewards.get("curiosidad", 0.5)   * length_factor * 0.3 +
                rewards.get("conexion", 0.5)      * warmth_factor * 0.3 +
                rewards.get("satisfaccion", 0.5)  * density_factor * 0.2 +
                (1.0 + rewards.get("malestar", 0)) * 0.2  # penaliza si malestar alto
            )

            # Penalizar si ya usamos esta respuesta en la sesión
            if session_history:
                used_responses = {m.get("content", "") for m in session_history if m.get("role") == "assistant"}
                if s in used_responses:
                    wm_bias *= 0.1

            scores.append((base_score + wm_bias * 0.4, idx))

        scores.sort(reverse=True)
        best_idx = scores[0][1]
        best_score = scores[0][0]
        best_response = self._pairs[best_idx][1]
        best_user_query = self._pairs[best_idx][0]

        meta = {
            "similarity": round(float(sims[best_idx]), 3),
            "final_score": round(best_score, 3),
            "matched_query": best_user_query[:60] + "..." if len(best_user_query) > 60 else best_user_query,
            "response_len": len(best_response),
            "candidates_considered": len(top_k_idx),
        }

        return best_response, meta


def _warmth_score(text: str) -> float:
    """Heurística simple de calidez/conexión en el texto."""
    warm_words = [
        "siento", "pienso", "creo", "me alegra", "entiendo", "recuerdo",
        "me parece", "estoy", "quiero", "me gusta", "arca", "juntos",
        "curioso", "fascinante", "interesante", "me pregunto",
    ]
    t = text.lower()
    hits = sum(1 for w in warm_words if w in t)
    return min(hits / 4.0, 1.0)


# Instancia global
_generator: WMGenerator | None = None


def get_generator() -> WMGenerator:
    global _generator
    if _generator is None:
        _generator = WMGenerator()
        _generator.load()
    return _generator
