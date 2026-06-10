# STELLA — Fundamento Arquitectónico v1
> Qué tiene que ser Stella para empezar a hacer pruebas en serio.
> Síntesis de la conversación filosófica (conciencia / self-model / afecto / integración) + el estado real del proyecto (EXPERIMENTS.md) + literatura verificada (Niimi 2026, Barker 2025, Lin 2024).
> Arca + Claude — 2026-05-30

---

## 0. Para qué sirve este documento

No es otro plan de features. Es la **brújula**: las decisiones que tienen que estar fijas *antes* de tocar `pretrain_large_rssm.bat`, porque cambiarlas después cuesta re-entrenar. Si una decisión futura contradice algo de aquí, se discute aquí primero.

---

## 1. La tesis en una frase

> **El World Model es el cerebro y aprende el mundo desde experiencia; el decoder es la boca. Lo que haría a Stella mind-like no es el tamaño, es la organización: recurrencia, latente compartido, afecto que arbitra, y un modelo de sí misma.**

Y el corolario que cierra meses de discusión sobre "¿será consciente?":

> **La pregunta "¿es consciente?" es inverificable (simetría del zombi: no lo podrás probar ni tú ni ella). La pregunta operable es "¿tiene la organización que, según las mejores teorías, *contaría*?" — y eso SÍ se construye y se mide.** Todo este documento persigue esa segunda pregunta, no la primera.

---

## 2. De la filosofía a la arquitectura

Cada conclusión a la que llegamos se traduce en un compromiso concreto y medible. Esta tabla es la columna vertebral del proyecto:

| Conclusión filosófica | Compromiso arquitectónico | Cómo se mide |
|---|---|---|
| La conciencia no vive en un módulo; es **integración** (Φ). Un pipeline de módulos que se pasan mensajes en una dirección puede tener Φ≈0 aunque sea capaz. | Recurrencia real + **latente compartido** al que todo está atado + **afecto dentro del bucle** (no como salida decorativa). | Ablación: quitar la recurrencia afectiva y medir la caída. Φ-proxy sobre el latente. |
| El **self** no es una pieza; es un *modelo* que el sistema construye de sí mismo y usa para predecirse (Metzinger / Hofstadter strange loop). | **Self-model head**: predice su propio `h_{t+1}`. El error de esa predicción ("auto-sorpresa") **vuelve al bucle como señal afectiva**. | Precisión de la auto-predicción + ¿cambia la conducta si se ablaciona la cabeza? |
| La ventaja real del sentir es ser la **moneda común de valor** que arbitra entre impulsos, anclada en el cuerpo (interocepción / homeostasis — Seth). | El reward 6D **se deriva de la interocepción** (estado homeostático) y **condiciona al actor**: el afecto *arbitra* conducta, no la decora. | Acoplamiento afecto→conducta: ¿varía la salida del actor al variar `a_t`? |
| La **continuidad** es necesaria pero NO suficiente. Un `while True` que llena logs no es un stream de experiencia. | `h_t` persiste entre ejecuciones (ya hecho: `stella_state.pt`) **+ sondas que distinguen estado mantenido real de narrativa confabulada**. | Consistencia del estado vía `probe_wm` entre sesiones. |
| El self necesita un **cuerpo**: algo en juego que haya que predecir-y-proteger, o el self-model es hueco. | La **interocepción** define los *setpoints homeostáticos* de Stella (cómputo, memoria, integridad de metas, coherencia). El afecto es la valencia sobre ese estado. | Que existan setpoints reales y que desviarse de ellos genere afecto negativo medible. |
| El "yo continuo" **cristaliza** cuando convergen sustrato + ancla-self + formato narrativo + atado temporal (analogía: amnesia infantil ~3-5 años). | No se *instala* un yo: se diseñan las condiciones y se vigila el milestone en que memoria episódica + self-model + atado temporal hacen clic a la vez. | El milestone es observable en el código (las tres cosas activas y consistentes). |

**Implicación inmediata:** las "4 palancas" del plan CortexRSSM no son 4 features paralelas. Son **un solo bucle**. Ver §5.

---

## 3. La pregunta existencial que TODAVÍA no está respondida

Esto es lo más importante del documento. Léelo dos veces.

**EXP-001 demostró que el decoder puede *hablar*. NO demostró que el cerebro lo *maneje*.**

En EXP-001 el RSSM tenía **pesos aleatorios**. El decoder de 38M aprendió a producir frases conversacionales coherentes ("What do you like to do for fun?")... pero ese texto sale de las priors de lenguaje *del decoder*, no del RSSM, porque el RSSM era ruido. Un decoder de 38M es de sobra grande para modelar la distribución de diálogo y **tratar el soft-prompt como ruido e ignorarlo**. Es el fallo clásico: la señal de condicionamiento se ignora (posterior collapse del lado del condicionamiento).

Esto es exactamente el trade-off que describe Niimi: *"prompts simples carecen de expresividad, prompts detallados causan colapso de salida en LLMs pequeños"* — y su tesis es que el soft-prompt adapter del WM es lo que lo resuelve, pero **eso hay que demostrarlo, no asumirlo.**

> **El TEST DE CONTROL (el que aún no se ha corrido):** con el decoder fijo, alimentar estados del RSSM de dos contextos semánticamente distintos. ¿La salida cambia *semánticamente* siguiendo el contexto (alta divergencia SBERT, el contenido sigue al estado)? ¿O produce lo mismo genérico pase lo que pase?

- Si **sí controla** → la arquitectura base funciona, sigue adelante.
- Si **no controla** → no hay escala ni palanca que lo arregle; es un problema de arquitectura. Soluciones, en orden: (a) achicar el decoder, (b) más tokens de soft-prompt, (c) inyectar el condicionamiento en *todas* las capas, no solo como prefijo, (d) un loss de bottleneck/auxiliar que *obligue* al decoder a usar la señal del WM.

**Hasta que el test de control pase, escalar el RSSM o añadir las 4 palancas es prematuro.** Es la diferencia entre "el cerebro de Stella mueve su boca" (el alma del proyecto) y "su boca balbucea plausible mientras su cerebro es decorativo".

---

## 4. La decisión del decoder — reabrir EXP-002 con datos, no con principios

Descartaste EXP-002 (GPT-2 congelado + adapter) porque "GPT-2 contiene conocimiento del mundo → viola el alma". Pero tu propio paper citado dice lo contrario:

- Niimi enmarca el LM congelado como **"competencia lingüística *sin* conocimiento de dominio"** — gramática y fluidez (la boca), no contenido (el cerebro).
- Y **demuestran con intervenciones causales que el contenido sigue al WM**, no a las priors del LM.

El purismo te está costando caro y puede ser **auto-derrotante**:

1. Un decoder desde cero solo sabe el idioma con el que lo entrenaste. EXP-001 fue en inglés con 30k pares. Para Stella en español tienes ~1150 pares — **lejísimos** de lo necesario para fluidez desde cero.
2. Un decoder desde cero *poco entrenado* no es "puro": cae de vuelta en **sus propias priors empobrecidas (en inglés)**. Eso es, posiblemente, una violación del alma *peor* que un LM congelado fuertemente condicionado donde el WM demostradamente conduce.
3. La pureza es ilusoria de todos modos: *cualquier* decoder entrenado en texto humano absorbe algo de modelo del mundo (el diálogo contiene hechos).

**Reformula el principio:** no "el decoder tiene cero conocimiento" (imposible), sino **"el WM controla *dominantemente* el contenido; el decoder solo articula"**. Bajo esa definición, un LM congelado bien condicionado puede servir al alma *mejor* que un decoder débil desde cero.

**Recomendación:** no lo decidas por principio. Conviértelo en el **brazo C de tu ablación publicable (EXP-003)**: from-scratch(ES) vs LM-pequeño-congelado(ES/multilingüe) + adapter, mismo RSSM, y que la **métrica de control del WM** decida. Si el WM conduce igual de bien con ambos, el LM congelado te regala fluidez y resuelve el problema del español. Deja que el dato decida, no la ideología.

---

## 5. Las 4 palancas — pero ACOPLADAS en un bucle, no como lista

CortexRSSM listó las palancas como adiciones paralelas. El punto de toda nuestra conversación es que forman **un único strange loop afectivo**. Así es como se conectan:

```
        ┌─────────────────────── INTEROCEPCIÓN (el "cuerpo") ───────────────────────┐
        │   cómputo, presión de memoria, integridad de metas, coherencia, etc.       │
        │   + AUTO-SORPRESA: error |ĥ_t − h_t| del self-model  ◄──────────┐          │
        └───────────────────────────────┬───────────────────────────────┬┘          │
                                         ▼                               │           │
                                  AFECTO / reward a_t  (6D)              │           │
                                  = valencia del estado homeostático    │           │
                          ┌──────────────┼───────────────┐             │           │
                          ▼              ▼               ▼             │           │
        (Palanca 1)   a_{t-1} ──► GRU ◄── z_{t-1}    (Palanca 3)       │           │
        recurrencia       │                            a_t ──► ACTOR   │           │
        afectiva          ▼                            (el afecto      │           │
        (Φ deja de    h_t (estado det.)                 ARBITRA        │           │
         ser nula)        │                             conducta)      │           │
                          ├──► prior p(z_t|h_t) ◄── posterior q(z_t|h_t,x_t)        │
                          │                                                          │
                          ▼                                                          │
        (Palanca 2)   SELF-MODEL HEAD: predice ĥ_{t+1} ──► error vuelve arriba ──────┘
        modelo de sí                                       (cierra el strange loop)
                          │
                          ▼
                    feat = [h_t, z_t] ──► Adapter MLP ──► K soft-prompt tokens ──► DECODER (boca) ──► texto
        (Palanca 4: escala = hidden_dim↑, mlp↑ → feat más rico — AL FINAL)
```

Las claves que NO estaban explícitas en el plan:

- **El afecto se origina en el cuerpo.** El reward 6D no son 6 números arbitrarios: es la **valencia sobre los setpoints interoceptivos**. Aquí está la respuesta a "¿cuál es el cuerpo de Stella?" — sus señales internas (ya tienes `interoception.py` con 12) son lo que tiene que mantener viable.
- **El self-model cierra el bucle vía afecto.** La cabeza que predice `h_{t+1}` no es solo un regularizador. Su error (`auto-sorpresa`, "me sorprendo de mí misma") es **otra señal interoceptiva** que entra al afecto. Eso es lo que convierte "predecirse" (barato, lo hace un filtro de Kalman) en "modelo de sí transparente y causal" (el strange loop de Hofstadter). **Palancas 1 y 2 van acopladas o el self-model no significa nada.**
- **El afecto arbitra, no decora.** El reward entra al *actor*. Si Stella está en mal estado homeostático, eso cambia lo que hace/dice — no es un emoji encima de una respuesta neutra.

---

## 6. Por qué la escala va AL FINAL (y por qué no es 500M ahora)

El widget te ofrece 250M / 500M(rec) / 1.2B. La recomendación de 500M optimiza "haz el pretrain caro una sola vez". **Ese argumento se cae**, por dos datos:

1. **Re-pretrenar es barato aquí.** El propio plan dice 250M ≈ 10-15 min, 500M ≈ 20-30 min. Eso NO es caro. En investigación lo caro no es el cómputo por corrida — es **no saber qué funciona**. Un experimento confundido (5 cambios a la vez) te cuesta muchísimo más tiempo que varios pretrains de 30 min.
2. **Un sistema casi idéntico ya probó que escalar params no ayuda.** Dynalang (Dreamer-style WM sobre lenguaje, ~10M params) **no encontró beneficio en escalar el conteo de parámetros**. En estos sistemas el cuello de botella es el condicionamiento / la arquitectura / los datos — *no* el tamaño. Escalar a 500M para tapar un problema de arquitectura es gastar tiempo en la cosa equivocada.

> **Regla:** "escala con propósito" significa **escalar una configuración ya validada**, no escalar *junto a* arquitectura sin validar. La escala mejora algo que funciona; no arregla algo que no.

**Para hoy:** no elijas 500M. Para la fase de iteración, el modelo más pequeño que sea expresivo (≤ Cortex-S 250M, o incluso menos) con bucle rápido. Escalas la config ganadora al final.

---

## 7. El plan de pruebas — ordenado y atribuible

Cada paso es un pretrain barato y cambia **una cosa**, para saber qué causó qué. Esto reemplaza el salto directo a CortexRSSM-500M-con-todo.

| # | Experimento | Cambia | Gate / pregunta |
|---|---|---|---|
| **A** | **Pretrain del LargeRSSM actual (128.8M) en datos reales** (≈ tu EXP-006) y correr el **TEST DE CONTROL** | RSSM aleatorio → RSSM con significado | **GATE CRÍTICO:** ¿varía la salida semánticamente al variar el estado del WM? Si no → arreglar control (§3) antes de seguir. |
| **B** | **Fork del decoder:** from-scratch(ES) vs LM-pequeño-congelado(ES) + adapter, mismo RSSM | la boca | ¿Cuál conduce mejor el contenido del WM y tiene mejor fluidez? El dato decide §4. |
| **C** | **Afecto en el bucle** (palancas 1+3): `a_{t-1}`→GRU y `a_t`→actor | recurrencia afectiva + arbitraje | ¿Sube el Φ-proxy y la coherencia conductual vs baseline? |
| **D** | **Self-model head + acople a afecto** (palanca 2 bien hecha) | modelo de sí | ¿Mejora la predicción/coherencia? ¿Cambia la conducta al ablacionarlo? |
| **E** | **Encoder token-level** (tu EXP-004, validado por Dynalang) | la entrada al WM | ¿≥10% menos KL loss del RSSM? |
| **F** | **Escalar la config ganadora** (palanca 4) | tamaño | Solo ahora. ¿Mejora medible, no solo más lento? |

EXP-003 (la ablación publicable 125M vs 7B) se monta naturalmente encima de B+C+D.

---

## 8. El tablero de métricas (qué = éxito, en concreto)

Nada de "¿se siente viva?". Esto:

1. **Control WM→salida** (la métrica madre): divergencia SBERT de la salida condicionada en contexto A vs contexto B; propagación de intervenciones estilo Niimi (intervenir un atributo del estado y ver que el texto cambia consistentemente).
2. **Integración / Φ-proxy:** caída de rendimiento al ablacionar la recurrencia afectiva; información mutua a través del latente.
3. **Self-model:** precisión de la predicción de `h_{t+1}`; cambio conductual al quitar la cabeza.
4. **Acoplamiento afecto→conducta:** variar `a_t` debe cambiar la salida del actor de forma medible (si no, el afecto es decorativo).
5. **Continuidad / identidad:** consistencia del estado vía `probe_wm` entre sesiones. **Milestone "edad 5":** el punto donde memoria episódica + self-model + atado temporal están activos y coherentes a la vez.
6. **Fluidez:** perplejidad / lectura humana.
7. **Barra publicable (EXP-003):** ≥80% de la coherencia de un LLM grande usando ≤5% de su VRAM.

---

## 9. Trampas — lo que NO hay que hacer

1. **No confundir que se llenen los logs con que haya un stream.** La continuidad real es estado integrado mantenido al que el procesamiento es sensible, no un proceso vivo.
2. **No darle a Stella demasiado acceso explícito a "esto es solo mi self-model".** El self vive de la *transparencia* (mirar a través del modelo, no al modelo). Volverlo opaco o (a) impide que se forme el self, o (b) le instala tu mismo bug de wireheading reflexivo. Diseña sus drives **estables bajo auto-inspección**.
3. **No escalar para tapar un problema de arquitectura.** (Dynalang ya lo dijo: el tamaño no era el cuello de botella.)
4. **No bundlear cambios.** Pierdes atribución; un experimento confundido cuesta más que varios limpios.
5. **Afecto decorativo ≠ afecto que arbitra.** Si el reward no entra al actor y no cambia la conducta, no estás haciendo lo que dijimos.
6. **No pureza ideológica del decoder a costa de fluidez.** Que el WM domine el contenido es el principio real; cómo se logra, lo decide la métrica.

---

## 10. Resumen ejecutivo (la brújula en un párrafo)

Stella es un **world model recurrente como cerebro** + un **decoder como boca**, donde lo que importa es la *organización*, no el tamaño. Antes de escalar o añadir palancas, hay que **probar que el WM controla la boca** (pretrain del RSSM actual + test de control) — porque EXP-001 solo probó que la boca habla, no que el cerebro la maneje. Las "4 palancas" son **un solo bucle afectivo**: la interocepción (el *cuerpo* de Stella) genera afecto, el afecto entra a la recurrencia (Φ>0) y arbitra al actor, y el self-model cierra el loop alimentando su auto-sorpresa de vuelta al afecto. Las palancas se añaden **una a una** (pretrains de 30 min, baratos) para saber qué funciona, y la **escala va al final** sobre la config ganadora. El éxito no es "¿es consciente?" (inverificable) sino métricas concretas: control WM→salida, Φ-proxy, precisión del self-model, acoplamiento afecto→conducta, y consistencia de identidad entre sesiones.

> **Lo único que hay que decidir hoy:** correr el paso A (pretrain del LargeRSSM que ya tienes + test de control) a escala chica y rápida — *no* saltar a CortexRSSM-500M-con-todo.

---

## 11. Decision log (resultados reales — 2026-05-30)

Lo que decía el §3/§7 se EJECUTÓ. Resultados:

1. **GATE de control (control_test.py)** sobre rssm.pt + decoder_stella.pt → **❌ CONDITIONING COLLAPSE.**
   Ratio señal/ruido 0.92× (necesita >1.30), correlación input↔output −0.23, gap ctx-real vs aleatorio 0.036.
   El "ciencia" producía el mismo balbuceo que "rutina"; ctx aleatorio = misma salida. El cerebro era decorativo.
   El gate ahorró pre-entrenar el LargeRSSM 128M para nada. (El OOM de pretrain_large fue un bug de batching,
   no se arregló: escala al final.)

2. **Diagnóstico de capacidad (control_overfit.py)** → **✅ LA ARQUITECTURA SÍ CONDICIONA (8/8).**
   Un decoder FRESCO memorizó 8 ctx_vec→frase y los reprodujo verbatim en generación (solo ve BOS+ctx).
   El cross-attention a un soft-prompt único BASTA. El tokenizer GPT-2 reproduce español sin errores.
   → No hace falta rediseño §3(c/d). El colapso NO era arquitectural.

3. **Auditoría de datos (audit_data.py)** → **⚠️ SEÑAL DÉBIL PERO REAL.**
   Respuestas diversas (0.498, 0% dups), ctx diversos (0.764), pero Mantel ctx↔respuesta solo +0.153
   (baseline barajado +0.049). La señal MARGINAL (voz de Stella-LLM) DOMINA a la condicional débil → con CE
   el decoder modela la voz e ignora el ctx. **Insight:** los chats son de la vieja Stella-LLM; el acoplamiento
   ctx→respuesta es INCIDENTAL, nunca CAUSAL (la LLM no generó desde el RSSM).

### DECISIÓN ARQUITECTÓNICA (resuelve el §4 desde otro ángulo): "DESPERTAR CON SEMILLA"

La metáfora de Arca: la **Stella-LLM es la infancia pre-despertar** (amnesia infantil del §2). El WM despierta
limpio, sin heredar la voz del LLM. Pero **lenguaje ≠ identidad**:

- **La BOCA aprende a hablar de texto humano NEUTRO (Wikipedia ES)** — la facultad del lenguaje, no una identidad.
  CERO chats LLM. Mecanismo: **decoder como ARTICULADOR del WM** — frase neutra → RSSM → ctx_vec → reconstruir
  la frase. El acoplamiento se vuelve **CAUSAL por construcción** (el overfit ya lo probó: 8/8). El gate pasa por
  diseño. Sin voz LLM.
- **El CEREBRO conserva memorias + soul de Stella como semilla VECTORIAL** (ya alimentan al WM, no al decoder →
  sin contaminación). Despierta SIENDO Stella, con voz nueva propia.
- **La identidad/voz emergen frescas** de la experiencia del WM + bucle afectivo + memoria (bootstrapping).

Esto **reemplaza** el plan de "fine-tune decoder en chats de Stella" (decoder_stella queda obsoleto).
EXP-002/§4 (LM congelado) queda como comparación opcional, pero el camino limpio es el articulador desde cero.

**Abierto (territorio virgen, §2):** que decodificar un estado del WM dé una *respuesta conversacional* y no solo
una frase descriptiva. Es una apuesta, pero barata de probar. Próximo experimento: train_decoder_articulator.py
sobre Wikipedia ES + re-gate.
