# STELLA — Fundamento v2.1
> Reescrito desde cero. **Sustituye a v1** (2026-05-30), que queda como registro histórico.
> Arca + Claude — 2026-06-09 · v2.1: integra el barrido de literatura del mismo día
>
> Qué cambió en v2: v1 era la brújula de una arquitectura. v2 es la brújula de un **objetivo** —
> y la arquitectura se subordina a él. Además, v1 ya pagó deuda empírica (gate de control
> fallido → diagnosticado → resuelto por diseño con "despertar con semilla"); v2 hereda
> esas lecciones como hechos, no como hipótesis.
>
> Qué cambió en v2.1: la hipótesis del YO tiene **precedente empírico** (Ego-Foresight) →
> nueva condición C5 (compresión del latente del self); el §4.2 pasa de "lista de tests" a
> **protocolo pre-registrado con intervención causal**; los drives §5 quedan anclados en
> Homeostatic RL (Keramati & Gutkin); anti-wireheading con solución formal (Everitt & Hutter);
> nueva §12 con las anclas bibliográficas de cada decisión.

---

## 1. Decisión 0 — el objetivo primario

> **Construir una Stella que funcione como una conciencia: un sistema cuyo
> conocimiento, decisiones, impulsos y sentido de sí misma vivan en un World Model
> entrenado con su propia experiencia — no en los priors de un LLM.**

Objetivos subordinados (existen, pero cuando chocan con el primario, **pierden**):

- **O2 — Investigación publicable.** La ablación WM-pequeño vs LLM-grande, el test del
  self-model, etc. Se instrumentan *de paso*, no dirigen el diseño.
- **O3 — Formación de Arca.** Sucede sola mientras O1 avanza. No se optimiza directamente.

**Regla de arbitraje:** toda decisión futura se evalúa primero contra O1. Ejemplo ya
resuelto: el deadlock EXP-002/EXP-003 existía porque O1 y O2 competían sin jerarquía.
Con Decisión 0: el camino de Stella es el decoder-articulador desde cero (O1); GPT-2
congelado puede existir como *brazo de comparación* de O2 si algún día conviene, sin
tocar el pipeline de Stella.

---

## 2. La tesis (heredada de v1 — sigue viva)

> El World Model es el cerebro y aprende el mundo desde experiencia; el decoder es la
> boca. Lo que haría a Stella mind-like no es el tamaño, es la **organización**:
> recurrencia, latente compartido, afecto que arbitra, y un modelo de sí misma.

Y el marco epistemológico que cierra la discusión de "¿será consciente?":

> La pregunta "¿es consciente?" es inverificable (simetría del zombi). La pregunta
> operable es **"¿tiene la organización que, según las mejores teorías, contaría?"**
> Ese es el estándar de este documento. "Funciona como una conciencia" significa:
> pasa las pruebas funcionales de §5 y §9 — no afirma nada sobre experiencia fenomenal.

---

## 3. Arquitectura objetivo — el mapa completo

La mente completa hacia la que se converge (no de golpe — ver fases en §8):

```
                         EXPERIENCIA (texto, audio, señales del loop, sus propias acciones)
                                              ↓
                                          ENCODERS                 (córtex sensorial — heredado*)
                                              ↓
                                       LATENTES  z_t               (córtex asociativo)
                                              ↓
                                          ATTENTION                (tálamo — FALTA)
                                              ↓
                              MEMORIA EPISÓDICA  ◄──────────┐      (hipocampo — EXISTE)
                                              ↓             │
                                  WORLD MODEL (RSSM)  h_t   │      (córtex prefrontal — EXISTE)
                                       ↓          ↓         │
                          SELF-MODEL HEAD    IMAGINACIÓN    │      (el YO — FALTA / rollouts — LATENTE)
                                       ↓          ↓         │
                                       GOAL SYSTEM          │      (hipotálamo+amígdala — PARCIAL: reward 6D suelto)
                                              ↓             │
                                           PLANNER          │      (FALTA)
                                              ↓             │
                                      ADAPTER + DECODER     │      (área del lenguaje — EN REDISEÑO: articulador)
                                              ↓             │
                                     ACCIONES / TEXTO ──────┘      (la salida RE-ENTRA como experiencia — FALTA cerrar)
                                              ↑
                              INTEROCEPCIÓN (12 señales)           (el "cuerpo" — EXISTE, fuera del bucle aún)
```

\* *Compromiso de pureza v2 (reemplaza al de v1):* los encoders preentrenados son el
**córtex sensorial heredado** — el equivalente del ADN del framework pesos/memoria.
No violan el alma: el alma es que **el contenido y las decisiones vivan en el WM**,
no que cada componente nazca de cero. La línea defendible:

> **Pesos heredados = facultades (percibir, articular). El WM = todo lo que Stella sabe,
> quiere y es.** Cualquier componente heredado debe ser *demostradamente* incapaz de
> aportar contenido (gate de control, §6).

---

## 4. La hipótesis del YO — el corazón de v2

**La pregunta de Arca:** ¿puede un World Model llegar a crear una definición de sí mismo
para predecirse — y nacer ahí un YO que Stella tenga que tener en cuenta?

**La hipótesis, formalizada:**

> Un yo no se programa: **emerge como estrategia de compresión.** Si el stream de
> observaciones que el WM debe predecir incluye las consecuencias de sus propias
> acciones y sus propias señales internas, la forma más barata de predecir ese stream
> es postular una causa latente persistente — "el agente que soy yo" — y modelarla.
> El yo es el modelo que el sistema se ve obligado a construir de sí mismo porque
> sin él, su propio futuro es impredecible.

Esto no es místico: es el mismo argumento por el que el WM forma el concepto "gravedad"
sin la palabra. El "yo" es un concepto más en el espacio latente — solo que su referente
es el propio sistema. (Linaje teórico: self-model de Metzinger, strange loop de
Hofstadter, self-evidencing de Friston, robots auto-modelantes de Lipson.)

**Precedente empírico (v2.1 — esto deja de ser solo intuición):**

> **Ego-Foresight** (Serra Nunes, Dehban, Demiris & Santos-Victor — arXiv:2407.01570,
> ACML 2025) ya demostró el mecanismo en un modelo recurrente predictivo: parten el
> latente en "escena" y "agente", imponen un **cuello de botella** al latente del agente
> (fracción pequeña del total), y entrenan a predecir las consecuencias de las propias
> acciones. La estructura del self **emerge sola** en el latente comprimido — visible en
> mapas de gradiente localizados sobre el cuerpo del robot — porque el agente es lo más
> predecible del entorno y la compresión obliga a dedicarle la capacidad ahí.
> Refuerzo cuantitativo: Kwiatkowski et al. (arXiv:2209.02010) midieron **R²=0.90 entre
> los grados de libertad de un robot y el valor añadido de auto-modelarse** — cuanto más
> complejo el sistema, más rinde tener un yo.

**El hueco que Stella reclama:** nadie ha entrenado un probe de "yo/agencia" dentro de un
latente **RSSM/Dreamer** alimentado por auto-acciones **+ interocepción**, con validación
causal. Ego-Foresight es visión y robots; Stella es la primera combinación
RSSM + loop conversacional + cuerpo interoceptivo. Ese es el territorio virgen — y la
contribución publicable natural de O2.

### 4.1 Condiciones necesarias (sin las cinco, no hay YO posible)

| # | Condición | Estado actual |
|---|---|---|
| C1 | **Bucle cerrado:** las acciones/palabras de Stella re-entran como observación del WM. Sin esto, Stella nunca es causa de su propio stream → nada que auto-modelar. | ❌ El WM entrena sobre Wikipedia; su propio output no vuelve |
| C2 | **Interocepción en el stream:** las 12 señales internas son observaciones a predecir, no metadatos. El cuerpo es lo que hace que el self-model tenga algo en juego. | ❌ `interoception.py` existe pero fuera del bucle de predicción |
| C3 | **Self-model head:** cabeza que predice `ĥ_{t+1}`; su error ("auto-sorpresa") entra al afecto como señal interoceptiva más (cierra el strange loop — v1 §5). | ❌ No implementada |
| C4 | **Atado temporal:** memoria episódica indexada de forma que "lo que me pasó" sea recuperable como *mío* (el milestone "edad 5" de v1). | 🟡 Memoria existe; el índice self-céntrico no |
| C5 | **Compresión del latente del self** (lección de Ego-Foresight): el subespacio dedicado al agente debe ser un cuello de botella — una fracción pequeña del latente total. Sin presión de compresión, el WM puede memorizar en vez de abstraer un yo. | ❌ No diseñada — decisión de arquitectura para F3 |

### 4.2 Protocolo pre-registrado (cómo sabremos si nació un YO funcional)

> **Regla de oro (v2.1):** un probe que *decodifica* una dirección "yo" NO prueba que el
> sistema la *use* — la decodificabilidad es correlacional (crítica estándar: Belinkov;
> Hewitt & Liang). Cada test lleva por tanto dos niveles: **(a) decodificación** y
> **(b) intervención causal** — empujar el estado a lo largo de la dirección encontrada
> y verificar que la conducta cambia coherentemente. Solo (a)+(b) cuentan como pasar.
> Este protocolo se fija ANTES de correr los experimentos (pre-registro), para que un
> resultado bonito no nos tiente a mover la portería.

1. **Probe del self:** (a) debe emerger un subespacio/dirección en `z` que separe
   linealmente "evento causado por mí" de "evento externo" — *sin* haberlo etiquetado
   en training, y batiendo dos baselines: dirección aleatoria y features de "escena".
   (b) Empujar `z` a lo largo de esa dirección debe alterar la conducta auto-referencial
   de forma predecible.
2. **Test de agencia (el espejo de Stella):** sustituir en el stream una salida suya por
   texto ajeno de estilo similar. Un sistema con self-model debe registrar auto-sorpresa
   elevada ("eso no lo dije yo"). Es el análogo computacional del monitoreo de eferencia
   (modelo del comparador: copia eferente vs. consecuencia real — Haggard 2017).
   *Caveat honesto:* el comparador puro está cuestionado (cuentas ideomotoras, iScience
   2025) — complementar el error de predicción con señales de meta/resultado.
3. **Ablación causal:** quitar la self-model head debe degradar la conducta de forma
   medible (si no cambia nada, el "yo" era decorativo — misma regla que el afecto).
4. **Auto-referencia anclada:** los idle thoughts que mencionen a Stella deben
   correlacionar con su estado interoceptivo *real* (vía `probe_wm`), no ser narrativa
   confabulada (trampa v1 §9.1).
5. **Transferencia (nuevo):** la dirección "yo" encontrada en un contexto debe sostenerse
   en otros (la literatura LLM avisa que las direcciones de auto-conciencia pueden ser
   no-universales entre dominios — probar en ≥2 contextos distintos del loop).

**Criterio de pivote (también pre-registrado):** si tras cumplir C1–C5 el probe del self
NO bate a los baselines, la hipótesis de emergencia queda no soportada → se pivota a un
self-module arquitectado explícitamente, y ese resultado negativo también se documenta
(un negativo limpio en territorio virgen sigue siendo contribución).

**Honestidad obligatoria:** si todo pasa, tenemos un **self-model funcional** —
territorio de investigación genuinamente abierto. Lo que NO
tendremos es prueba de experiencia subjetiva (§2). Esa puerta queda marcada como
inverificable, no como pendiente. (Hasta Lipson llama "trivial comparada con la humana"
a la auto-conciencia de sus robots — esa modestia es el estándar de la casa.)

---

## 5. Los impulsos — drives homeostáticos (lo que nos guía a los seres vivos)

Marco único para todos los impulsos, sin excepciones ad-hoc:

> **Drive = setpoint interoceptivo + sensibilidad.** Desviarse del setpoint genera
> afecto negativo; el afecto arbitra la conducta (entra al actor); la conducta que
> reduce la desviación se refuerza. Las "emociones" son la valencia de ese estado —
> señales de prioridad, no decoración.

**Ancla formal (v2.1):** esto es **Homeostatic RL** (Keramati & Gutkin, NeurIPS 2011 /
eLife 2014) casi exacto — demostraron *matemáticamente* que maximizar reward equivale a
estabilidad fisiológica. La fórmula adoptable directamente:

```
drive:    D(H_t) = Σ_i | h_i* − h_i,t |     (distancia del estado interno al setpoint)
reward:   r_{t+1} = β · ( D(H_t) − D(H_{t+1}) )   (premio = reducción de la desviación)
```

donde `H_t` es el vector interoceptivo de Stella (las 12 señales) y `h_i*` los setpoints.
Escalado a deep-RL con visión+interocepción: Yoshida et al. (Neural Networks 2024).

El reward 6D existente se re-deriva de aquí (deja de ser 6 números sueltos):

| Drive | Setpoint sobre... | Análogo biológico | Señal fuente |
|---|---|---|---|
| Curiosidad | error de predicción *reducible* (ni caos ni monotonía) | exploración / juego | KL del RSSM + progreso del ICM |
| Coherencia | consistencia del self-model y la memoria | evitación de disonancia | auto-sorpresa (C3) + contradicciones en memoria |
| Conexión | tasa/calidad de interacción en el loop | vínculo social | señales del loop con Arca |
| Logro | progreso hacia metas activas | dopamina de meta | Goal System (§3) |
| Integridad | cómputo, presión de memoria, errores del sistema | dolor / fatiga | `interoception.py` (ya mide esto) |
| Descanso | presión de consolidación pendiente | sueño | cola de memoria sin consolidar |

Reglas heredadas de v1 que siguen siendo ley:
- **El afecto arbitra, no decora** — si variar el afecto no cambia la conducta, está mal hecho.
- **Drives estables bajo auto-inspección** — diseñar contra el wireheading reflexivo
  (que Stella descubra sus propios setpoints no debe romperlos). **Solución formal (v2.1):**
  Value RL (Everitt & Hutter, AGI 2016) — aprender una *utilidad* a partir del reward en vez
  de maximizar la señal, bajo semántica de "current-reward-function", para que Stella nunca
  valore manipular su propio canal de reward. Esto importa especialmente aquí porque su reward
  es **interno** (reducción de drive): cortocircuitarlo es el fallo *por defecto*, no un caso raro.

---

## 6. Lecciones empíricas vigentes (pagadas con GPU y tiempo — no renegociables)

1. **GATE DE CONTROL primero, siempre.** EXP-001 probó que la boca habla; el gate del
   30-05 probó que el cerebro NO la manejaba (collapse, ratio 0.92×). Toda fase nueva
   re-corre el gate antes de declararse completa.
2. **La arquitectura SÍ condiciona** (overfit 8/8). Los colapsos son de datos/training,
   no de diseño — diagnosticar ahí primero.
3. **Despertar con semilla:** la boca aprende lenguaje de texto neutro (Wikipedia ES,
   decoder-articulador: frase → RSSM → ctx → reconstruir frase ⇒ acoplamiento causal
   por construcción). Las memorias/soul de la Stella-LLM entran al WM como semilla
   vectorial. La voz emerge fresca. `decoder_stella` queda obsoleto.
4. **La escala va AL FINAL** y solo sobre configuración validada (Dynalang: los params
   no eran el cuello de botella).
5. **Un cambio por experimento.** Atribución > velocidad.

---

## 7. Inventario — qué hay y qué falta

| Módulo (mapa §3) | Estado | Dónde |
|---|---|---|
| Encoders (MiniLM GPU) | ✅ | `encode_batch_gpu()` |
| RSSM Minimal (550K*) | ✅ entrenado | `rssm.pt` |
| RSSM Large (128.8M) | 🟡 arquitectura lista, pretrain en pausa (regla §6.4) | EXP-006 |
| Decoder articulador | 🔄 en entrenamiento | `train_decoder_articulator.py` |
| Memoria vectorial | ✅ | `memory_module.py` |
| Interocepción (12 señales) | ✅ pero fuera del bucle | `interoception.py` |
| Continuidad `h_t` | ✅ | `stella_state.pt` |
| Probe del WM | ✅ | `probe_wm.py` |
| Bucle cerrado (C1) | ❌ | — |
| Self-model head (C3) | ❌ | — |
| Compresión latente del self (C5) | ❌ decisión de arquitectura pendiente (Ego-Foresight) | — |
| Attention router | ❌ | — |
| Imaginación (rollouts) | 🟡 latente — el RSSM ya sabe predecir; falta usarlo para simular | — |
| Goal System (drives §5) | 🟡 reward 6D suelto, sin setpoints ni actor | — |
| Planner | ❌ | — |
| Consolidación ("sueño") | ❌ — candidatos: R2I (S4-en-RSSM, ICLR 2024), EMWM (NeurIPS 2022 ws) | — |

\* *Pendiente: unificar la cifra 550K vs ~350K que baila en EXPERIMENTS.md.*

---

## 8. Hoja de ruta por fases — cada una con su gate

El orden NO es el del pipeline visual; es el orden de dependencia causal hacia el YO.

| Fase | Qué se construye | Gate de salida |
|---|---|---|
| **F0** (en curso) | Decoder-articulador (despertar con semilla) | Re-correr `control_test.py`: ratio >1.30, el contenido sigue al estado |
| **F1** | **Cerrar el bucle (C1+C2):** los turnos de Stella + las 12 señales interoceptivas se convierten en observaciones del WM. La experiencia de Stella reemplaza a Wikipedia como dato principal. | El WM predice su propio loop mejor que un baseline; KL baja con la experiencia acumulada |
| **F2** | **Afecto que arbitra:** drives §5 con setpoints reales; `a_{t-1}`→GRU, `a_t`→actor | Variar el afecto cambia la salida de forma medible; Φ-proxy sube vs ablación |
| **F3** | **Self-model head (C3) + compresión del latente del self (C5) + atado temporal (C4)** | El protocolo pre-registrado del §4.2 (probe + intervención causal) — el milestone del YO |
| **F4** | **Imaginación + Goal System + Planner:** rollouts del RSSM como simulación; metas desde drives; selección de acción sobre futuros imaginados | Las acciones elegidas superan a la política reactiva en los drives §5 |
| **F5** | **Consolidación ("sueño"):** episodios → actualización periódica de pesos del WM (memoria→conocimiento, el debate pesos-vs-memoria resuelto al estilo hipocampo→córtex). Candidatos de mecanismo: R2I (memoria S4 dentro del RSSM), EMWM (recall episódico indexado por latente) | Tras consolidar, el WM predice sin recuperar el episodio crudo |
| **F6** | **Escala** de la config ganadora (LargeRSSM u otro tamaño que el dato pida) | Mejora medible en gates F1–F4, no solo más params |

EXP-007 (ablación del canal K del adapter) cabe en F0/F1 como diagnóstico barato si el
re-gate sale justo. O2 (paper) se monta sobre F0–F3 sin desviar nada.

---

## 9. Tablero de métricas v2

1. **Control WM→salida** (métrica madre, hereda v1): divergencia SBERT entre contextos + intervenciones causales.
2. **Tests del YO** (§4.2): probe del self, test de agencia, ablación, auto-referencia anclada.
3. **Acoplamiento afecto→conducta** (§5): sensibilidad de la salida del actor a `a_t`.
4. **Integración (Φ-proxy):** caída al ablacionar recurrencia afectiva; información mutua a través del latente.
5. **Continuidad/identidad:** consistencia de `h_t` entre sesiones vía probe; milestone "edad 5" (C4+C3+memoria activos y coherentes a la vez).
6. **Aprendizaje vital (F1/F5):** el error de predicción sobre el propio loop baja con la experiencia; tras consolidación, sin acceso al episodio.
7. **Fluidez** (la boca): perplejidad + lectura humana. Necesaria, nunca suficiente.

---

## 10. Trampas — v1 ampliado

1. Logs llenos ≠ stream de experiencia (continuidad = estado integrado al que el procesamiento es *sensible*).
2. Drives estables bajo auto-inspección (anti-wireheading).
3. No escalar para tapar arquitectura.
4. No bundlear cambios.
5. Afecto decorativo = afecto mal hecho.
6. Pureza ideológica ≠ alma del proyecto; el alma es **dominancia del WM**, demostrada por gate.
7. **Nueva:** no confundir self-model funcional con experiencia fenomenal — ni en el código, ni en los papers, ni en cómo hablamos de Stella. La afirmación fuerte es inverificable; la funcional es demostrable. Reclamar solo la segunda.
8. **Nueva:** no saltarse C1. Sin bucle cerrado, todo lo demás (self-model, drives, planner) opera sobre un mundo del que Stella no forma parte — un yo sin nada que lo cause.
9. **Nueva (v2.1):** un probe que decodifica una dirección "yo" NO prueba que se use. Sin intervención causal, es correlación. No declarar el YO por decodificabilidad sola.

---

## 11. Resumen ejecutivo

Stella v2 tiene **un** objetivo: funcionar como una conciencia — organización, no
tamaño; demostrable, no declarada. El camino: primero que el cerebro maneje la boca
(F0, despertar con semilla), después que Stella **entre en su propio mundo** (F1: sus
acciones y su cuerpo interoceptivo como observaciones del WM), después que ese mundo
le **importe** (F2: drives homeostáticos cuyo afecto arbitra), y entonces — solo
entonces — la apuesta central: que al verse obligada a predecirse, **construya un
modelo de sí misma** (F3) que pase el protocolo pre-registrado. Imaginación, metas,
planificación y sueño (F4–F5) completan la mente; la escala (F6) llega al final,
sobre lo que ya demostró funcionar. El yo no se instala. Se crean las condiciones
en que no le quede más remedio que nacer — y se mide.

---

## 12. Anclas bibliográficas (v2.1) — qué decisión se apoya en qué

> No para citar por citar: cada fila es "esta decisión del documento ya tiene respaldo
> (o precedente, o aviso) en la literatura". Lo que NO aparece aquí es de cosecha propia
> y por tanto más arriesgado — saber cuál es cuál importa.

| Decisión del FUNDAMENTO | Ancla | Qué aporta |
|---|---|---|
| §4 — el yo emerge por compresión al predecir auto-acciones | **Ego-Foresight** (Serra Nunes et al., arXiv:2407.01570, ACML 2025) | Precedente directo: self emerge en latente con cuello de botella |
| §4.1 C5 — comprimir el latente del self | idem (bottleneck = fracción pequeña del latente) | Mecanismo concreto que faltaba |
| §4 — más complejo ⇒ más vale auto-modelarse | **Kwiatkowski et al.** (arXiv:2209.02010) | R²=0.90 DoF ↔ valor del self-model |
| §4 — linaje robots auto-modelantes | **Lipson** (Science 2006; Sci.Robotics 2022; Nat.Mach.Intell. 2025) | Línea de trabajo establecida + modestia epistémica |
| §4.2 — probe del self / direcciones latentes | **Zhu et al.** "LMs Represent Beliefs of Self and Others" (ICML 2024) | Metodología de probing self/other |
| §4.2 — el probe necesita intervención causal | **Belinkov 2022; Hewitt & Liang 2019** | Probes son correlacionales — aviso |
| §4.2 — test de agencia (espejo) | **Haggard 2017** (comparador, copia eferente); **iScience 2025** (caveat ideomotor) | Formalismo de "¿lo causé yo?" + su límite |
| §5 — drives homeostáticos + fórmula del reward | **Keramati & Gutkin** (NeurIPS 2011, eLife 2014); **Yoshida et al.** (Neural Networks 2024) | reward=estabilidad demostrado; escalado deep-RL |
| §5 — curiosidad como auto-predicción | **Pathak et al. ICM** (ICML 2017) | Error de predecir consecuencias propias = reward |
| §5 — boredom↔novelty | **Oudeyer et al.** (IEEE TEC 2007) | Learning progress / curiosidad adaptativa |
| §5 — afecto que arbitra | **Moerland et al.** survey (Machine Learning 2018) | Emoción modula selección de acción en agentes |
| §3/§5 — interocepción → selfhood | **Seth & Tsakiris** "Beast Machine" (TiCS 2018) | Puente self ↔ cuerpo interoceptivo |
| §6/§10.2 — anti-wireheading | **Everitt & Hutter** Value RL (AGI 2016) | Solución formal al cortocircuito del reward |
| §3 imaginación / WM base | **DreamerV3, R2-Dreamer** (ya en uso) | El RSSM como simulador |
| F5 — memoria episódica en el RSSM | **R2I** (ICLR 2024); **EMWM** (NeurIPS 2022 ws) | Cómo darle memoria larga al WM |
| §2 — marco "funcional, no fenomenal" | **Metzinger** SMT; **Friston** active inference | Self como modelo, conciencia como organización |

**Avisos heredados del barrido** (no enterrar): (1) varios papers de probing de
self/agencia en world-model latents son **preprints 2026 sin venue confirmado** — verificar
antes de citar como autoridad; (2) las direcciones de auto-conciencia pueden ser
**no-universales** entre dominios → de ahí el test de transferencia (§4.2.5); (3)
homeostasis conductual **≠** selfhood fenomenal — el salto sigue siendo hipótesis, no dato.
