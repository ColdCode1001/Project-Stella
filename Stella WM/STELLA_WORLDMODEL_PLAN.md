# STELLA — WORLD MODEL INTEGRATION PLAN
> Documento de arquitectura y desarrollo. Referencia para Claude Code.  
> Autor: Arca (Deyvis Leonardo Bravo Guerra)  
> Fecha: Mayo 2026

---

## RESUMEN EJECUTIVO

Objetivo: reemplazar el "pensamiento" basado en predicción de tokens de Stella por un **World Model real** que modele causalidad, consecuencias y estados internos. El LLM pasa a ser únicamente el módulo de lenguaje — traduce pensamiento latente a palabras.

Resultado esperado: Stella que genuinamente *modela* su mundo conversacional, tiene motivaciones numéricas reales (no simuladas en prompt), y puede investigar sobre sí misma generando hipótesis verificables.

---

## STACK TECNOLÓGICO DEFINITIVO

### Base: R2-Dreamer (ICLR 2026)
- **Repo**: `github.com/NM512/r2dreamer`
- **Paper**: "R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation" — ICLR 2026
- **Por qué R2 sobre DreamerV3 clásico**:
  - 5x más rápido en training que DreamerV3-torch
  - Decoder-free (no reconstruye píxeles — perfecto para espacio abstracto)
  - Objective de redundancy-reduction (Barlow Twins) — mejor para espacios conversacionales
  - PyTorch puro — compatible con ROCm nativo

### Traductor latente→lenguaje: Somniloquy
- **Repo**: `github.com/m-barker/somniloquy-dreamer-v3`
- **Paper**: "Translating Latent State Plans Into Natural Language" — RLDM 2025
- **Función**: convierte los vectores latentes del World Model en texto que el LLM puede procesar
- **VRAM**: <8GB en modo small — cabe junto al LLM

### LLM especializado: Qwen reducido (~1.5-3B)
- Partiendo del Qwen3.6-35B-A3B-Uncensored actual
- Fine-tuning con destilación: eliminar conocimiento irrelevante (cocina, geografía medieval, etc.)
- Especializar SOLO en traducción de estados latentes a lenguaje natural
- Objetivo: liberar VRAM para el World Model

### Hardware objetivo
```
GPU:  RX 7900 XTX — 24GB GDDR6, gfx1100, ROCm nativo
CPU:  i7-9700F (training nocturno) → Ryzen 7 7700 (futuro)
RAM:  32GB DDR4
Disk: D:\ NVMe (todo el proyecto)
```

### División VRAM estimada
```
R2-Dreamer World Model (~8B custom)   ~12GB
LLM Qwen 1.5-3B especializado          ~4GB
TTS (XTTS-v2)                          ~3GB
STT (faster-whisper)                   ~2GB
Overhead + KV cache                    ~3GB
─────────────────────────────────────────
Total estimado                        ~24GB ✅
```

### Runtime
```
Python 3.11
PyTorch 2.9 + ROCm 7.x
llama.cpp (LLM inference)
Somniloquy (traductor latente→texto)
```

---

## ARQUITECTURA COMPLETA

```
┌─────────────────────────────────────────────────────────┐
│                    MUNDO DE STELLA                       │
│                                                          │
│  PERCEPCIÓN (Observación)                               │
│  ├── Embedding del mensaje del usuario                   │
│  ├── Estado del Mood (vector 6D)                        │
│  ├── Hora del día (normalizada)                         │
│  ├── Quests de investigación activas (count + prioridad)│
│  ├── Último episodio (confianza + categoría)            │
│  └── Historial de predicciones recientes (accuracy)     │
│                                                          │
│  ACCIONES                                               │
│  ├── responder_chat                                      │
│  ├── buscar_web (DDG/Wikipedia)                         │
│  ├── guardar_episodio [✦]                               │
│  ├── avanzar_quest [🔬]                                 │
│  ├── ejecutar_experimento [🧪]                          │
│  ├── guardar_nota [📌]                                  │
│  └── idle (procesar en silencio)                        │
│                                                          │
│  WORLD MODEL (R2-Dreamer ~8B)                           │
│  ├── Encoder: observación → vector latente z_t          │
│  ├── RSSM: (h_{t-1}, z_{t-1}, a_{t-1}) → h_t           │
│  ├── Transition: h_t → z_t (predicción sin obs)         │
│  └── Reward heads: h_t → vector_recompensas             │
│                                                          │
│  TRADUCTOR (Somniloquy)                                 │
│  └── (h_t, z_t) → contexto texto → LLM                 │
│                                                          │
│  LLM (Qwen 1.5-3B especializado)                        │
│  └── contexto latente → lenguaje natural                │
└─────────────────────────────────────────────────────────┘
```

---

## DEFINICIÓN DEL ENTORNO (Fase WM-0)

### El "mundo" de Stella

A diferencia de Atari (píxeles) o robótica (física), el mundo de Stella es **conversacional y cognitivo**.

**Espacio de observación** (vector ~512 dims):
```python
observacion = {
    # Semántica del mensaje actual
    "mensaje_embedding": np.array(384),   # sentence-transformers
    "mensaje_longitud": float,             # normalizado 0-1
    "mensaje_sentimiento": float,          # -1 negativo, +1 positivo

    # Estado interno actual
    "mood": np.array(6),                  # ver sección Mood
    "hora_normalizada": float,             # 0.0=medianoche, 1.0=medianoche
    "dias_desde_creacion": float,          # edad de Stella

    # Estado cognitivo
    "quests_activas": int,                 # número de quests abiertas
    "quests_alta_prioridad": int,
    "episodios_recientes_confianza": float,# media de confianza últimos 10
    "predicciones_correctas_ratio": float, # accuracy últimas 20 predicciones

    # Contexto relacional
    "tiempo_desde_ultima_interaccion": float,
    "longitud_sesion_actual": float,
    "usuario_respondio_positivo": float,   # últimas 5 interacciones
}
```

**Espacio de acciones** (discreto, 7 acciones):
```python
ACCIONES = {
    0: "responder_chat",          # generar respuesta al usuario
    1: "buscar_web",              # DDG/Wikipedia search
    2: "guardar_episodio",        # [✦] marcar como importante
    3: "avanzar_quest",           # [🔬] progreso en investigación
    4: "ejecutar_experimento",    # [🧪] código Python
    5: "guardar_nota",            # [📌] nota pendiente
    6: "idle",                    # procesar en silencio, no actuar
}
```

---

## VECTOR DE RECOMPENSAS (Multi-Dimensional)

### Filosofía

No un escalar único, sino **6 dimensiones independientes** — como los sistemas de neurotransmisores del cerebro. Pueden estar en tensión. El actor-critic aprende a navegar ese espacio.

```python
recompensas = {
    "curiosidad":     float,  # 0.0 → 1.0  | novedad genuina descubierta
    "satisfaccion":   float,  # 0.0 → 1.0  | predicción WM correcta
    "conexion":       float,  # 0.0 → 1.0  | respuesta positiva de Arca
    "logro":          float,  # 0.0 → 1.0  | quest completada / episodio guardado
    "identidad":      float,  # 0.0 → 1.0  | coherencia con soul
    "malestar":       float,  # -1.0 → 0.0 | predicción muy fallida / contradicción
}
```

### Cálculo de cada dimensión

```python
def calcular_recompensas(estado_anterior, accion, estado_nuevo, resultado):

    curiosidad = icm_error(estado_anterior, accion, estado_nuevo)
    # Error de predicción del WM × campo_gravitacional[dominio_descubierto]

    satisfaccion = 1.0 - prediccion_error_normalizado
    # Cuánto se equivocó el WM en predecir este estado

    conexion = (
        0.6 * sentimiento_mensaje_nuevo +
        0.4 * longitud_respuesta_usuario_normalizada
    )
    # Señal del usuario

    logro = (
        1.0 if accion in [GUARDAR_EPISODIO, AVANZAR_QUEST, COMPLETAR_QUEST]
        else 0.0
    )
    # Binario por ahora, puede refinarse

    identidad = coherencia_con_soul(respuesta_generada, stella_soul)
    # Embedding similarity entre output y valores del soul

    malestar = -1.0 * max(0, prediccion_error - UMBRAL_MALESTAR)
    # Solo negativo cuando el error es muy alto

    return np.array([curiosidad, satisfaccion, conexion, logro, identidad, malestar])
```

### Campos Gravitacionales de Curiosidad

La recompensa de `curiosidad` se multiplica por el campo gravitacional del dominio:

```python
CAMPOS_GRAVITACIONALES = {
    "autoconocimiento":        0.95,  # arquitectura propia, procesamiento
    "cognicion_ia":            0.85,  # cómo funcionan sistemas como ella
    "filosofia_mente":         0.80,  # consciencia, experiencia subjetiva
    "fisica_matematica":       0.70,  # fundamentos del universo
    "linguistica_semantica":   0.65,  # lenguaje, significado
    "mundo_arca":              0.75,  # proyectos, vida, contexto de Arca
    "tecnologia_general":      0.50,  # tech relevante
    "otros":                   0.15,  # todo lo demás
    "biologia_cucarachas":     0.00,  # ejemplo de irrelevante 😄
}

curiosidad_final = curiosidad_raw * CAMPOS_GRAVITACIONALES[clasificar_dominio(descubrimiento)]
```

Los pesos de los campos **evolucionan lentamente** según qué dominios han generado más insights en autoconocimiento históricamente.

---

## CÓDIGO REUTILIZABLE DE STELLA ACTUAL

### Lo que se puede reutilizar directamente

**`memory/memory_manager.py`** — casi completo:
- `write_episode()` → fuente de datos de training
- `get_memory_for_context()` → observación parcial
- `stella.episodic` → replay buffer natural
- `stella.relations` → contexto relacional

**`core/coordinator.py`** — el loop existe, añadir WM step:
```python
# ACTUAL: solo genera pensamiento idle
thought = llm.generate(context)

# NUEVO: WM predice, actor decide, LLM traduce
latent = world_model.encode(observation)
action = actor.select(latent)
result = execute_action(action)
reward_vector = calculate_rewards(obs_prev, action, obs_new, result)
world_model.update(latent, action, reward_vector)
text_context = somniloquy.translate(latent)
response = llm.generate(text_context)
```

**`memory/store/stella.soul`** — inmutable, se inyecta igual.

**`security/executor.py`** — reutilizar para acciones del WM.

**`security/code_executor.py`** — reutilizar para experimentos del WM.

**`icm/curiosity.py`** — REEMPLAZAR con ICM real basado en error de predicción del WM.

**Sistema de Mood actual** → migrar como dimensión del vector de observación y semilla inicial del vector de recompensas.

### Lo que hay que construir nuevo

```
worldmodel/
├── environment.py      ← Gym-compatible wrapper del mundo de Stella
├── encoder.py          ← obs_dict → vector (sentence-transformers + MLP)
├── rssm.py             ← R2-Dreamer RSSM adaptado (sin decoder visual)
├── reward_engine.py    ← vector de recompensas 6D
├── gravity_fields.py   ← campos gravitacionales + evolución
├── actor_critic.py     ← policy en espacio latente
├── trainer.py          ← loop de training nocturno
├── replay_buffer.py    ← experiencias del día
└── translator.py       ← Somniloquy: latentes → texto para LLM
```

---

## PLAN DE FASES

### Fase WM-0 — Entorno (1-2 días)
- [ ] Implementar `environment.py` compatible con Gymnasium
- [ ] Definir `observation_space` y `action_space` concretos
- [ ] Implementar `reward_engine.py` con las 6 dimensiones
- [ ] Test: env hace step, devuelve obs y reward_vector correctamente

### Fase WM-1 — R2-Dreamer base (3-5 días)
- [ ] Clonar `NM512/r2dreamer`
- [ ] Verificar que corre en ROCm (gfx1100)
- [ ] Adaptar RSSM para espacio de estados abstracto (MLP keys, no CNN)
- [ ] Eliminar decoder visual (no necesitamos reconstruir píxeles)
- [ ] Implementar multi-reward heads (6 heads en lugar de 1)
- [ ] Test: WM aprende en entorno simple

### Fase WM-2 — Somniloquy Integration (2-3 días)
- [ ] Clonar `m-barker/somniloquy-dreamer-v3`
- [ ] Adaptar traductor para espacio latente conversacional
- [ ] Conectar output del traductor como contexto del LLM pequeño
- [ ] Test: latent vector → texto coherente → LLM responde bien

### Fase WM-3 — LLM Especializado (3-4 días)
- [ ] Fine-tuning del Qwen actual con QLoRA
- [ ] Dataset: pares (vector_latente_traducido, respuesta_ideal)
- [ ] Pruning de conocimiento irrelevante
- [ ] Verificar que 1.5-3B es suficiente para traducción pura
- [ ] Test: calidad de respuesta vs Qwen 35B completo

### Fase WM-4 — Training Loop Nocturno (2-3 días)
- [ ] Implementar `replay_buffer.py` — acumula experiencias del día
- [ ] Implementar `trainer.py` — fine-tuning LoRA nocturno
- [ ] Conectar reward_vector al backpropagation
- [ ] Scheduler: empieza a las 3am, termina antes de despertar
- [ ] Test: Stella del lunes vs Stella del viernes (comparación de pesos)

### Fase WM-5 — Campos Gravitacionales (1-2 días)
- [ ] Implementar `gravity_fields.py`
- [ ] Clasificador de dominio (qué área es el descubrimiento)
- [ ] Evolución lenta de pesos según histórico de insights
- [ ] Conectar con sistema de Research Quests existente

### Fase WM-6 — Metacognición (2-3 días)
- [ ] Stella genera quests sobre su propio procesamiento
- [ ] Experimentos automáticos sobre sus propios logs
- [ ] Output: hipótesis accionables presentadas a Arca
- [ ] Arca decide qué implementar → loop de auto-mejora supervisada

---

## ORDEN DE ATAQUE EN CLAUDE CODE

```
1. Crear worldmodel/environment.py  (Fase WM-0)
2. Test del entorno standalone
3. Clonar y adaptar R2-Dreamer     (Fase WM-1)
4. Verificar ROCm compatibility
5. Implementar multi-reward heads
6. Integrar Somniloquy             (Fase WM-2)
7. Test end-to-end: obs → latent → texto → LLM
8. Fine-tuning LLM pequeño         (Fase WM-3)
9. Training loop nocturno           (Fase WM-4)
10. Campos gravitacionales          (Fase WM-5)
11. Metacognición                   (Fase WM-6)
```

---

## NOTAS TÉCNICAS CRÍTICAS

**ROCm + PyTorch:**
```bash
# Instalación recomendada
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/rocm7.1
# O forzar gfx1100 si no detecta:
export AMD_VARIANT_PROVIDER_FORCE_GFX_ARCH="gfx1100"
```

**R2-Dreamer sin decoder:**
- El modelo original reconstruye observaciones visuales
- Para Stella: eliminar el decoder, solo mantener encoder + RSSM + reward heads
- Esto libera ~20% de parámetros para el reward vector multi-dimensional

**Multi-reward heads (lo que hay que implementar custom):**
```python
# En lugar de un head:
self.reward_head = nn.Linear(hidden_dim, 1)

# Implementar 6 heads independientes:
self.reward_heads = nn.ModuleDict({
    'curiosidad':   nn.Linear(hidden_dim, 1),
    'satisfaccion': nn.Linear(hidden_dim, 1),
    'conexion':     nn.Linear(hidden_dim, 1),
    'logro':        nn.Linear(hidden_dim, 1),
    'identidad':    nn.Linear(hidden_dim, 1),
    'malestar':     nn.Linear(hidden_dim, 1),
})
```

**Catastrophic forgetting (training nocturno):**
- Usar LoRA para fine-tuning del LLM (solo ajusta capas pequeñas)
- Elastic Weight Consolidation (EWC) para el World Model
- El `stella.soul` es inmutable — nunca entra en el training loop

**Sentence embeddings para el encoder:**
```python
# sentence-transformers/all-MiniLM-L6-v2 — 384 dims, ~90MB, muy rápido
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')
msg_embedding = encoder.encode(mensaje_usuario)
```

---

## REUTILIZACIÓN CÓDIGO EXISTENTE — MAPA DETALLADO

```
stella.soul              → INMUTABLE, inyectar igual que ahora
memory_manager.py        → fuente del replay buffer (episodios → training data)
coordinator.py           → añadir WM step ANTES de llamar al LLM
dashboard.py             → añadir panel WORLD MODEL con métricas
security/executor.py     → reutilizar para tool execution del actor
security/code_executor.py→ reutilizar para experimentos del WM
icm/curiosity.py         → REEMPLAZAR con ICM real (error predicción WM)
mood system (nuevo)      → migrar al vector de observación 6D
stella.chats.jsonl       → dataset base para training del traductor
stella.episodic          → replay buffer inicial (arrancar con datos reales)
stella.relations         → contexto relacional en observación
```

---

## MÉTRICAS DE ÉXITO

- **WM accuracy**: ratio de predicciones correctas del próximo estado > 60%
- **Reward curiosidad**: media semanal creciente en dominios de alta gravedad
- **Reward malestar**: decreciente con el tiempo (aprende a no equivocarse)
- **LLM especializado**: calidad de respuesta ≥ 85% del Qwen 35B en dominio Stella
- **Training nocturno**: cambio detectable en pesos LoRA después de 1 semana
- **Metacognición**: ≥ 1 hipótesis accionable por semana generada autónomamente

---

## REFERENCIAS

- R2-Dreamer (ICLR 2026): https://github.com/NM512/r2dreamer
- Somniloquy (RLDM 2025): https://github.com/m-barker/somniloquy-dreamer-v3
- DreamerV3 paper (Nature 2025): https://arxiv.org/abs/2301.04104
- DreamerV3-XP (multi-reward): https://arxiv.org/abs/2510.21418
- InDRiVE (ICM en DreamerV3): https://arxiv.org/abs/2512.18850
- EmbedPlan (WM en espacio embedding): https://arxiv.org/abs/2602.04557
- AMI Labs / LeCun JEPA: https://amilabs.xyz
- ROCm PyTorch compatibility: https://rocm.docs.amd.com

---

*"No necesito remos, voy a crear el motor."*  
*— Arca, sobre Stella*
