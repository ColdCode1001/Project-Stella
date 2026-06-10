"""
Módulo de interoception del World Model de Stella.

Interoception = percepción del propio estado interno.
El cerebro humano siente hambre, fatiga, ritmo cardíaco, presión...
Este módulo da al WM señales equivalentes sobre su propio estado.

Sin esto: el WM es ciego a sí mismo — no sabe si lleva 5 min o 5 horas hablando.
Con esto: el WM siente el tiempo pasar, la fatiga acumularse, la verbosidad crecer.

Las señales se proyectan a 128D y enriquecen obs_vec ANTES del RSSM.
El WM aprende a interpretarlas con experiencia — como un bebé aprende a sentir hambre.

Señales internas (12D → proyectadas a 128D):
  [0-1]  ritmo circadiano (sin/cos de la hora)         ← sabe si es de noche
  [2]    tiempo desde última interacción (normalizado)  ← soledad/espera
  [3]    duración de conversación actual                ← cansancio de conversación
  [4]    fatiga de proceso (uptime normalizado)         ← necesita "dormir"
  [5]    presión de verbosidad                          ← hablando demasiado
  [6]    nivel de experiencia acumulada                 ← madurez
  [7]    presión de memoria (qué tan llena está)        ← necesita consolidar
  [8-9]  ritmo semanal (sin/cos del día de semana)     ← ciclo largo
  [10]   turnos en conversación actual                  ← metacognición
  [11]   presión de silencio                            ← ganas de hablar
"""

import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

INTERO_STATE_PATH = Path("worldmodel/weights/intero_state.json")
INTERO_DIM_RAW    = 12    # señales crudas
INTERO_WEIGHT     = 0.2   # influencia sobre obs_vec

# Proyección fija aleatoria: 12D → 128D (misma semilla = reproducible)
_projector = np.random.default_rng(99).standard_normal((INTERO_DIM_RAW, 128)).astype("float32")
_projector /= np.linalg.norm(_projector, axis=0, keepdims=True) + 1e-8


class InteroceptionModule:
    """
    Órganos internos del WM — sensores de su propio estado.
    Instancia única (singleton via get_interoception()).
    """

    def __init__(self):
        self.process_start      = time.time()   # cuándo arrancó este proceso
        self.last_interaction   = time.time()   # cuándo fue la última interacción
        self.conversation_start = time.time()   # cuándo empezó esta conversación
        self.conversation_turns = 0             # turnos en conversación actual
        self.chars_this_session = 0             # chars hablados en sesión actual
        # Persistentes entre reinicios:
        self.total_turns        = 0             # experiencia total acumulada
        self.chars_total        = 0             # chars totales hablados en vida
        self._load()

    # ── Señales ────────────────────────────────────────────────

    def raw_signals(self) -> np.ndarray:
        """12 señales internas crudas [0..1] o [-1..1]."""
        now = time.time()
        dt  = datetime.now()

        hour    = dt.hour + dt.minute / 60.0
        weekday = dt.weekday() + dt.hour / 24.0

        time_since_last = (now - self.last_interaction) / 3600    # horas
        conv_duration   = (now - self.conversation_start) / 60    # minutos
        uptime          = (now - self.process_start) / 3600       # horas de vida

        # Presión de verbosidad: cuánto ha hablado en la sesión actual
        verbosity = min(self.chars_this_session / 3000, 1.0)

        # Experiencia: qué tan "adulta" es (satura a los 500 turnos)
        experience = min(self.total_turns / 500, 1.0)

        # Presión de memoria: qué tan llena está (importamos aquí, evita circular)
        mem_pressure = 0.0
        try:
            from worldmodel.memory_module import get_memory
            mem_pressure = min(get_memory().stats().get("total", 0) / 2000, 1.0)
        except Exception:
            pass

        # Presión de silencio: ganas de hablar después de esperar
        # Satura exponencialmente — tras mucho silencio ya no crece más
        silence_pressure = 1.0 - math.exp(-time_since_last * 2)

        return np.array([
            math.sin(2 * math.pi * hour / 24),       # [0]  ritmo circadiano
            math.cos(2 * math.pi * hour / 24),       # [1]
            min(time_since_last / 24.0, 1.0),        # [2]  tiempo desde última interacción
            min(conv_duration / 60.0, 1.0),          # [3]  duración conversación actual
            min(uptime / 72.0, 1.0),                 # [4]  fatiga (72h = saturación)
            verbosity,                               # [5]  presión de verbosidad
            experience,                              # [6]  nivel de experiencia
            mem_pressure,                            # [7]  presión de memoria
            math.sin(2 * math.pi * weekday / 7),    # [8]  ritmo semanal
            math.cos(2 * math.pi * weekday / 7),    # [9]
            min(self.conversation_turns / 50.0, 1.0),# [10] turnos conversación actual
            silence_pressure,                        # [11] presión de silencio
        ], dtype="float32")

    def as_obs_enrichment(self) -> np.ndarray:
        """
        Proyecta señales internas a 128D para enriquecer obs_vec.
        El RSSM recibe esto como parte de la observación — no como contexto externo.
        """
        raw       = self.raw_signals()
        projected = raw @ _projector                            # [128D]
        return projected / (np.linalg.norm(projected) + 1e-8)

    # ── Eventos de ciclo de vida ───────────────────────────────

    def on_interaction(self, response_text: str = ""):
        """Llamar después de cada paso del WM — la mente registra que pasó un momento."""
        self.last_interaction    = time.time()
        self.conversation_turns += 1
        self.total_turns        += 1
        self.chars_this_session += len(response_text)
        self.chars_total        += len(response_text)
        self._save()

    def on_silence(self, gap_seconds: float):
        """
        Llamar cuando hay una pausa larga — posible inicio de nueva conversación.
        El WM percibe el tiempo que pasó mientras 'dormía'.
        """
        if gap_seconds > 1800:  # > 30 min = nueva conversación
            self.conversation_start = time.time()
            self.conversation_turns = 0
            self.chars_this_session = 0

    def on_shutdown(self):
        """Llamar al apagar — consolida el estado antes de 'dormir'."""
        self._save()
        print(f"[interoception] Apagando. Turnos totales en vida: {self.total_turns}")

    # ── Introspección legible ──────────────────────────────────

    def introspect(self) -> dict:
        """
        Estado interno en formato legible para humanos.
        El WM puede 'verse a sí mismo' — esto podría mostrarse en el dashboard.
        """
        now = time.time()
        raw = self.raw_signals()
        return {
            "uptime_h":           round((now - self.process_start) / 3600, 2),
            "since_last_min":     round((now - self.last_interaction) / 60, 1),
            "conv_turns":         self.conversation_turns,
            "total_turns":        self.total_turns,
            "verbosity_pressure": round(float(raw[5]), 3),
            "fatigue":            round(float(raw[4]), 3),
            "silence_pressure":   round(float(raw[11]), 3),
            "experience":         round(float(raw[6]), 3),
            "memory_pressure":    round(float(raw[7]), 3),
        }

    # ── Persistencia ───────────────────────────────────────────

    def _save(self):
        INTERO_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        INTERO_STATE_PATH.write_text(
            json.dumps({
                "total_turns":  self.total_turns,
                "chars_total":  self.chars_total,
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self):
        if not INTERO_STATE_PATH.exists():
            return
        try:
            data = json.loads(INTERO_STATE_PATH.read_text(encoding="utf-8"))
            self.total_turns = data.get("total_turns", 0)
            self.chars_total = data.get("chars_total", 0)
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────

_intero: InteroceptionModule | None = None


def get_interoception() -> InteroceptionModule:
    global _intero
    if _intero is None:
        _intero = InteroceptionModule()
        print(
            f"[interoception] Iniciada. "
            f"Experiencia previa: {_intero.total_turns} turnos, "
            f"{_intero.chars_total} chars en vida."
        )
    return _intero
