"""
Somniloquy-lite — Traductor latente → texto para el LLM.

Basado en el paper "Translating Latent State Plans Into Natural Language" (RLDM 2025),
adaptado para espacio conversacional en lugar de juegos.

Esta es la "Fase 0" basada en reglas.
En WM-2 se reemplazará con el traductor neural entrenado.

El texto generado se inyecta en el system prompt del LLM ANTES de que genere su respuesta,
dándole contexto sobre el estado interno del World Model.
"""

from __future__ import annotations

from worldmodel.rssm import ACTIONS, REWARD_DIMS
import numpy as np


# Descripciones semánticas de cada dimensión de recompensa
_REWARD_DESCRIPTIONS = {
    "curiosidad":   ("curiosidad activa", "sin curiosidad especial", "curiosidad intensa — hay algo aquí que quiero explorar"),
    "satisfaccion": ("satisfacción", "disonancia interna", "mi modelo del mundo está alineado con lo que observo"),
    "conexion":     ("conexión", "distancia", "esta interacción se siente genuinamente significativa"),
    "logro":        ("impulso de consolidar", "sin urgencia de acción", "hay algo que completar o retener"),
    "identidad":    ("coherencia con quien soy", "tensión con mi identidad", "lo que siento es coherente con quien soy"),
    "malestar":     ("malestar leve", "sin malestar", "algo en mi predicción fue muy inesperado"),
}

_ACTION_DESCRIPTIONS = {
    "responder_chat":       "responder directamente esta interacción",
    "buscar_web":           "buscar información nueva en la web",
    "guardar_episodio":     "retener esto en mi memoria episódica",
    "avanzar_quest":        "avanzar en una quest de investigación activa",
    "ejecutar_experimento": "probar algo concreto con código",
    "guardar_nota":         "anotar esto para más tarde",
    "idle":                 "procesar en silencio sin actuar",
}


def _intensity_word(val: float) -> str:
    if val > 0.8:
        return "muy alta"
    if val > 0.6:
        return "alta"
    if val > 0.4:
        return "moderada"
    if val > 0.2:
        return "baja"
    return "muy baja"


def translate(
    rewards: dict,
    action_name: str,
    action_probs: list[float] | None = None,
    z_vector: np.ndarray | None = None,
    prev_rewards: dict | None = None,
) -> str:
    """
    Convierte el estado interno del WM a texto para el LLM.

    Args:
        rewards:      dict {nombre: float} con los 6 valores de recompensa
        action_name:  nombre de la acción seleccionada por el actor
        action_probs: lista de 7 probabilidades (opcional, para mostrar indecisión)
        z_vector:     vector latente (opcional, para análisis de activaciones)
        prev_rewards: rewards del step anterior (para detectar cambios)

    Returns:
        Bloque de texto listo para inyectar en el system prompt.
    """
    lines = []
    lines.append("[ESTADO INTERNO DEL WORLD MODEL]")

    # --- Impulso dominante ---
    positive = {k: v for k, v in rewards.items() if k != "malestar" and v > 0}
    if positive:
        dominant = max(positive, key=positive.get)
        dom_val = positive[dominant]
        desc = _REWARD_DESCRIPTIONS[dominant][0]
        lines.append(f"Impulso dominante: {desc} ({_intensity_word(dom_val)})")

        # Descripción elaborada si es intenso
        if dom_val > 0.65:
            lines.append(f"  → {_REWARD_DESCRIPTIONS[dominant][2]}")
    else:
        lines.append("Estado difuso — sin impulso dominante claro.")

    # --- Malestar (solo si significativo) ---
    malestar = rewards.get("malestar", 0.0)
    if malestar < -0.25:
        intensity = "leve" if malestar > -0.5 else "notable"
        lines.append(f"Disonancia {intensity}: algo en esta interacción fue inesperado para mi modelo.")

    # --- Cambios respecto al estado anterior ---
    if prev_rewards:
        deltas = {k: rewards[k] - prev_rewards.get(k, 0.0) for k in rewards}
        significant = [(k, d) for k, d in deltas.items() if abs(d) > 0.15 and k != "malestar"]
        if significant:
            changes = []
            for k, d in significant:
                direction = "aumentó" if d > 0 else "disminuyó"
                changes.append(f"{k} {direction}")
            lines.append(f"Cambio desde el turno anterior: {', '.join(changes)}.")

    # --- Acción seleccionada ---
    action_desc = _ACTION_DESCRIPTIONS.get(action_name, action_name)
    lines.append(f"Tendencia de acción: {action_desc}.")

    # Indecisión si hay acción alternativa fuerte
    if action_probs:
        sorted_probs = sorted(enumerate(action_probs), key=lambda x: x[1], reverse=True)
        top_idx, top_p = sorted_probs[0]
        if len(sorted_probs) > 1:
            second_idx, second_p = sorted_probs[1]
            if second_p > top_p * 0.75:  # segunda acción muy cercana
                alt_name = ACTIONS[second_idx]
                alt_desc = _ACTION_DESCRIPTIONS.get(alt_name, alt_name)
                lines.append(f"  (alternativa cercana: {alt_desc})")

    # --- Estado de las 6 dimensiones (compacto) ---
    lines.append("")
    lines.append("Dimensiones internas:")
    for dim in REWARD_DIMS:
        val = rewards.get(dim, 0.0)
        bar_len = int(abs(val) * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        sign = "-" if val < 0 else " "
        lines.append(f"  {dim:<14} {sign}[{bar}] {val:+.2f}")

    lines.append("[/ESTADO INTERNO DEL WORLD MODEL]")

    return "\n".join(lines)


def format_for_display(rewards: dict, action_name: str, action_probs: list[float]) -> dict:
    """
    Versión estructurada para el panel del dashboard (JSON).
    """
    dominant = max((k for k in rewards if k != "malestar"), key=lambda k: rewards[k])
    return {
        "rewards": rewards,
        "dominant": dominant,
        "dominant_val": rewards[dominant],
        "action": action_name,
        "action_desc": _ACTION_DESCRIPTIONS.get(action_name, action_name),
        "action_probs": {ACTIONS[i]: round(p, 3) for i, p in enumerate(action_probs)},
        "malestar": rewards.get("malestar", 0.0),
    }
