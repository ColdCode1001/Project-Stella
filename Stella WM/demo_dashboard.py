"""
Stella WM Demo Dashboard — puerto 5001.

Chat impulsado SOLO por el World Model — sin LLM.

El WM genera el estado latente y el vector de recompensas.
El generador recupera la respuesta más adecuada del historial
de Stella usando similitud semántica + reranking por estado WM.

Panel derecho muestra en tiempo real:
  - Vector de recompensas 6D
  - Acción seleccionada por el actor
  - Estado interno (Somniloquy-lite)
  - Metadata de la recuperación (query matched, score)
"""

import json
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

# --- World Model imports ---
sys.path.insert(0, str(Path(__file__).parent))
from worldmodel.rssm import MinimalRSSM, ACTIONS, load_or_create
from worldmodel.obs_encoder import encode_observation, OBS_DIM
from worldmodel.translator import translate, format_for_display
from worldmodel.generator import get_generator
from worldmodel.memory_module import get_memory
from worldmodel.interoception import get_interoception

app = Flask(__name__)

CONTINUITY_STATE_PATH = Path("worldmodel/weights/stella_state.pt")
INTERO_WEIGHT         = 0.2   # influencia de interoception sobre obs_vec

# ─── Estado global ──────────────────────────────────────────────────────────────
_wm_enabled   = True
_wm_lock      = threading.Lock()
_session_lock = threading.Lock()

_session_history: list[dict] = []

# WM state — persiste entre EJECUCIONES (no solo sesiones)
_wm_model: MinimalRSSM = load_or_create("worldmodel/weights/rssm.pt")
_wm_h: "torch.Tensor | None" = None
_wm_last_rewards: dict | None = None
_wm_last_action_idx: int = 0
_wm_last_display: dict = {}


def _save_continuous_state():
    """
    Guarda el estado continuo de la mente al disco.
    La mente no tiene sesiones — tiene momentos en el tiempo.
    Llamar después de cada paso del WM.
    """
    if _wm_h is None:
        return
    try:
        CONTINUITY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "h":               _wm_h,
            "last_rewards":    _wm_last_rewards,
            "last_action_idx": _wm_last_action_idx,
        }, CONTINUITY_STATE_PATH)
    except Exception:
        pass


def _load_continuous_state() -> bool:
    """
    Restaura el estado de la mente desde el último momento activo.
    No hay 'inicio de sesión' — la mente continúa desde donde estaba.
    """
    global _wm_h, _wm_last_rewards, _wm_last_action_idx
    if not CONTINUITY_STATE_PATH.exists():
        print("[continuidad] Primera vez — mente iniciando desde cero.")
        return False
    try:
        state = torch.load(
            CONTINUITY_STATE_PATH, map_location="cpu", weights_only=True
        )
        _wm_h              = state["h"]
        _wm_last_rewards   = state.get("last_rewards")
        _wm_last_action_idx= state.get("last_action_idx", 0)
        mem_stats = get_memory().stats()
        print(
            f"[continuidad] Mente restaurada — "
            f"{mem_stats.get('total', 0)} recuerdos en memoria."
        )
        return True
    except Exception as e:
        print(f"[continuidad] No se pudo restaurar estado: {e}. Iniciando desde cero.")
        return False


# Restaurar estado continuo al arrancar
_load_continuous_state()


# ─── World Model step ───────────────────────────────────────────────────────────
def _wm_step(user_message: str) -> tuple[dict, dict]:
    """
    Ejecuta un paso del World Model.
    Devuelve (display_dict, {z, rewards, action_probs, action_sel}).
    """
    global _wm_h, _wm_last_rewards, _wm_last_action_idx, _wm_last_display

    with _wm_lock:
        obs = encode_observation(
            message=user_message,
            session_length=len(_session_history),
            last_reward=_wm_last_rewards,
        )

        # Enriquecer observación con contexto de memoria (hipocampo)
        obs = get_memory().enrich_observation(obs)

        # Enriquecer observación con percepción interna (interoception)
        intero = get_interoception()
        intero_vec = intero.as_obs_enrichment()
        obs = obs + INTERO_WEIGHT * intero_vec
        obs = obs / (np.linalg.norm(obs) + 1e-8)

        z, h_new, rewards, action_probs, action_sel = _wm_model.step(
            obs_vec=obs,
            action_idx=_wm_last_action_idx,
            h_prev=_wm_h,
        )

        somniloquy_text = translate(
            rewards=rewards,
            action_name=ACTIONS[action_sel],
            action_probs=action_probs,
            z_vector=z,
            prev_rewards=_wm_last_rewards,
        )

        display = format_for_display(rewards, ACTIONS[action_sel], action_probs)
        display["somniloquy"] = somniloquy_text

        _wm_h = h_new
        _wm_last_rewards = rewards
        _wm_last_action_idx = action_sel
        _wm_last_display = display

        # Guardar en memoria si el actor lo decide
        if ACTIONS[action_sel] in ("guardar_episodio", "guardar_nota"):
            get_memory().add(obs, user_message, importance=1.2)

        # Persistir estado continuo — la mente nunca pierde su momento
        _save_continuous_state()

        # Registrar interacción en interoception — el WM siente que pasó un momento
        get_interoception().on_interaction()

        internals = {"z": z, "h": h_new, "rewards": rewards, "action_probs": action_probs, "action_sel": action_sel}

    return display, internals


# ─── HTML ───────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stella ◈ World Model Demo</title>
<style>
  :root {
    --bg: #0a0a0a; --bg2: #111; --bg3: #181818; --border: #2a2a2a;
    --text: #e8e8e8; --dim: #666; --accent: #c8a96e; --red: #c0392b;
    --green: #2ecc71; --blue: #3498db; --purple: #9b59b6;
    --reward-curiosidad: #f39c12; --reward-satisfaccion: #27ae60;
    --reward-conexion: #3498db; --reward-logro: #9b59b6;
    --reward-identidad: #1abc9c; --reward-malestar: #e74c3c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Courier New', monospace; height: 100vh; display: flex; flex-direction: column; }

  header {
    padding: 10px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg2);
  }
  header h1 { font-size: 14px; letter-spacing: 4px; color: var(--accent); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 6px; }
  .status-dot.off { background: var(--dim); }

  .main { display: flex; flex: 1; overflow: hidden; }

  /* ── Chat panel ── */
  .chat-panel { flex: 1; display: flex; flex-direction: column; border-right: 1px solid var(--border); }

  .messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .messages::-webkit-scrollbar { width: 4px; }
  .messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .msg { max-width: 85%; padding: 10px 14px; border-radius: 6px; font-size: 13px; line-height: 1.5; }
  .msg.user { align-self: flex-end; background: #1a1a2e; border: 1px solid #2d2d5a; color: #a0a0d0; }
  .msg.stella { align-self: flex-start; background: var(--bg3); border: 1px solid var(--border); }
  .msg.stella .speaker { font-size: 10px; color: var(--accent); letter-spacing: 2px; margin-bottom: 4px; }
  .msg.system { align-self: center; background: transparent; color: var(--dim); font-size: 11px; border: none; }

  .typing { align-self: flex-start; color: var(--dim); font-size: 12px; padding: 4px 0; }

  .input-area {
    padding: 12px 16px; border-top: 1px solid var(--border); background: var(--bg2);
    display: flex; gap: 8px;
  }
  .input-area input {
    flex: 1; background: var(--bg3); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 4px; font-family: inherit; font-size: 13px; outline: none;
  }
  .input-area input:focus { border-color: var(--accent); }
  .input-area button {
    background: var(--accent); color: #000; border: none; padding: 8px 16px;
    border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 12px;
    font-weight: bold; letter-spacing: 1px;
  }
  .input-area button:disabled { background: var(--dim); cursor: not-allowed; }

  /* ── WM panel ── */
  .wm-panel {
    width: 360px; display: flex; flex-direction: column; background: var(--bg2);
    overflow-y: auto; padding: 14px; gap: 14px;
  }
  .wm-panel::-webkit-scrollbar { width: 4px; }
  .wm-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .wm-header { display: flex; align-items: center; justify-content: space-between; }
  .wm-title { font-size: 11px; letter-spacing: 3px; color: var(--accent); }

  .toggle-wrap { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--dim); }
  .toggle { position: relative; width: 36px; height: 18px; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background: var(--border); border-radius: 18px; transition: .3s; }
  .slider:before { position: absolute; content: ""; height: 12px; width: 12px; left: 3px; bottom: 3px; background: var(--dim); border-radius: 50%; transition: .3s; }
  input:checked + .slider { background: var(--accent); }
  input:checked + .slider:before { transform: translateX(18px); background: #000; }

  .section { background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; padding: 10px; }
  .section-title { font-size: 10px; letter-spacing: 2px; color: var(--dim); margin-bottom: 8px; }

  /* Reward bars */
  .reward-row { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; font-size: 11px; }
  .reward-label { width: 90px; color: var(--dim); }
  .reward-bar-wrap { flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .reward-bar { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
  .reward-val { width: 36px; text-align: right; font-size: 10px; }

  /* Action probs */
  .action-row { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; font-size: 10px; }
  .action-label { flex: 1; color: var(--dim); }
  .action-bar-wrap { width: 80px; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .action-bar { height: 100%; background: var(--blue); border-radius: 3px; transition: width 0.4s ease; }
  .action-prob { width: 28px; text-align: right; }
  .action-row.selected .action-label { color: var(--accent); font-weight: bold; }
  .action-row.selected .action-bar { background: var(--accent); }

  /* Somniloquy text */
  .somniloquy-text {
    font-size: 11px; line-height: 1.6; color: #aaa; white-space: pre-wrap;
    border-left: 2px solid var(--accent); padding-left: 8px;
    max-height: 280px; overflow-y: auto;
  }
  .somniloquy-text::-webkit-scrollbar { width: 3px; }
  .somniloquy-text::-webkit-scrollbar-thumb { background: var(--border); }

  .placeholder { color: var(--dim); font-size: 11px; text-align: center; padding: 20px 0; }

  .wm-off-notice {
    color: var(--dim); font-size: 11px; text-align: center; padding: 30px 10px;
    border: 1px dashed var(--border); border-radius: 4px;
  }
</style>
</head>
<body>

<header>
  <h1>STELLA ◈ WORLD MODEL DEMO</h1>
  <div style="display:flex;gap:16px;align-items:center;font-size:11px;color:var(--dim)">
    <span><span class="status-dot"></span>WM puro — sin LLM</span>
    <span id="step-count">steps: 0</span>
  </div>
</header>

<div class="main">

  <!-- ── Chat ── -->
  <div class="chat-panel">
    <div class="messages" id="messages">
      <div class="msg system">World Model activo — escribe algo para ver el primer step.</div>
    </div>
    <div class="input-area">
      <input type="text" id="input" placeholder="Escribe un mensaje a Stella..." autocomplete="off">
      <button id="send-btn" onclick="send()">ENVIAR</button>
    </div>
  </div>

  <!-- ── WM Panel ── -->
  <div class="wm-panel">
    <div class="wm-header">
      <span class="wm-title">WORLD MODEL</span>
      <div class="toggle-wrap">
        <span>WM</span>
        <label class="toggle">
          <input type="checkbox" id="wm-toggle" checked onchange="toggleWM(this.checked)">
          <span class="slider"></span>
        </label>
        <span id="wm-state-label">ON</span>
      </div>
    </div>

    <div id="wm-off-notice" class="wm-off-notice" style="display:none">
      World Model desactivado.<br>El LLM recibe solo el contexto base.
    </div>

    <div id="wm-content">
      <!-- Rewards -->
      <div class="section">
        <div class="section-title">VECTOR DE RECOMPENSAS</div>
        <div id="reward-bars">
          <div class="placeholder">esperando primer step…</div>
        </div>
      </div>

      <!-- Action -->
      <div class="section">
        <div class="section-title">ACCIONES (ACTOR)</div>
        <div id="action-bars">
          <div class="placeholder">esperando primer step…</div>
        </div>
      </div>

      <!-- Estado interno -->
      <div class="section">
        <div class="section-title">ESTADO INTERNO (SOMNILOQUY)</div>
        <div id="somniloquy-text" class="placeholder">Aqui aparecera el estado interno del WM traducido a texto.</div>
      </div>

      <!-- Retrieval metadata -->
      <div class="section">
        <div class="section-title">RECUPERACION SEMANTICA</div>
        <div id="retrieval-meta" class="placeholder">esperando primer step...</div>
      </div>
    </div>
  </div>
</div>

<script>
const REWARD_COLORS = {
  curiosidad: '#f39c12', satisfaccion: '#27ae60', conexion: '#3498db',
  logro: '#9b59b6', identidad: '#1abc9c', malestar: '#e74c3c'
};
const ACTIONS = [
  'responder_chat','buscar_web','guardar_episodio','avanzar_quest',
  'ejecutar_experimento','guardar_nota','idle'
];

let wmEnabled = true;
let stepCount = 0;
let generating = false;

function toggleWM(on) {
  wmEnabled = on;
  document.getElementById('wm-state-label').textContent = on ? 'ON' : 'OFF';
  document.getElementById('wm-off-notice').style.display = on ? 'none' : 'block';
  document.getElementById('wm-content').style.opacity = on ? '1' : '0.3';
  fetch('/wm/toggle', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({enabled: on})});
}

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (role === 'stella') {
    div.innerHTML = '<div class="speaker">STELLA</div>' + escapeHtml(text);
  } else if (role === 'system') {
    div.textContent = text;
  } else {
    div.textContent = text;
  }
  document.getElementById('messages').appendChild(div);
  div.scrollIntoView({behavior: 'smooth'});
  return div;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
}

function updateWMPanel(data) {
  if (!data || !data.rewards) return;

  // Reward bars
  const rewards = data.rewards;
  let rHtml = '';
  for (const [name, val] of Object.entries(rewards)) {
    const absVal = Math.abs(val);
    const pct = Math.round(absVal * 100);
    const color = REWARD_COLORS[name] || '#888';
    const displayVal = val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2);
    rHtml += `<div class="reward-row">
      <span class="reward-label">${name}</span>
      <div class="reward-bar-wrap">
        <div class="reward-bar" style="width:${pct}%;background:${color}"></div>
      </div>
      <span class="reward-val" style="color:${color}">${displayVal}</span>
    </div>`;
  }
  document.getElementById('reward-bars').innerHTML = rHtml;

  // Action bars
  const probs = data.action_probs || {};
  const selectedAction = data.action || '';
  let aHtml = '';
  for (const actionName of ACTIONS) {
    const p = probs[actionName] || 0;
    const pct = Math.round(p * 100);
    const isSelected = actionName === selectedAction;
    aHtml += `<div class="action-row${isSelected ? ' selected' : ''}">
      <span class="action-label">${actionName}</span>
      <div class="action-bar-wrap">
        <div class="action-bar" style="width:${pct}%"></div>
      </div>
      <span class="action-prob">${(p*100).toFixed(0)}%</span>
    </div>`;
  }
  document.getElementById('action-bars').innerHTML = aHtml;

  // Estado interno (Somniloquy)
  const somEl = document.getElementById('somniloquy-text');
  somEl.className = 'somniloquy-text';
  somEl.textContent = data.somniloquy || '—';

  // Retrieval metadata
  if (data.retrieval) {
    const r = data.retrieval;
    const retEl = document.getElementById('retrieval-meta');
    retEl.className = 'somniloquy-text';
    retEl.innerHTML =
      `<b>Query matched:</b> "${r.matched_query || '—'}"\n` +
      `<b>Similitud:</b> ${r.similarity || '—'}\n` +
      `<b>Score final:</b> ${r.final_score || '—'}\n` +
      `<b>Candidatos evaluados:</b> ${r.candidates_considered || '—'}\n` +
      `<b>Longitud respuesta:</b> ${r.response_len || '—'} chars`;
  }

  // Step counter
  stepCount++;
  document.getElementById('step-count').textContent = `steps: ${stepCount}`;
}

async function send() {
  if (generating) return;
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

  generating = true;
  document.getElementById('send-btn').disabled = true;

  addMsg('user', msg);

  // Typing indicator
  const typingEl = document.createElement('div');
  typingEl.className = 'typing';
  typingEl.textContent = 'Stella está pensando…';
  document.getElementById('messages').appendChild(typingEl);
  typingEl.scrollIntoView({behavior: 'smooth'});

  let stellaDiv = null;
  let fullText = '';

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg, wm_enabled: wmEnabled})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, {stream: true});
      const lines = chunk.split('\\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === '[DONE]') continue;

        try {
          const evt = JSON.parse(raw);

          if (evt.type === 'wm_state') {
            // WM step completado — actualizar panel
            updateWMPanel(evt.data);
            typingEl.textContent = 'Stella generando respuesta…';

          } else if (evt.type === 'token') {
            if (!stellaDiv) {
              typingEl.remove();
              stellaDiv = addMsg('stella', '');
              stellaDiv.querySelector ? null : null;
            }
            fullText += evt.content;
            stellaDiv.innerHTML = '<div class="speaker">STELLA</div>' + escapeHtml(fullText);
            stellaDiv.scrollIntoView({behavior: 'smooth'});

          } else if (evt.type === 'done') {
            // nada extra
          } else if (evt.type === 'error') {
            typingEl.remove();
            addMsg('system', 'Error: ' + evt.message);
          }
        } catch(e) {}
      }
    }
  } catch(err) {
    typingEl.remove();
    addMsg('system', 'Error de conexión: ' + err.message);
  } finally {
    generating = false;
    document.getElementById('send-btn').disabled = false;
    input.focus();
  }
}

document.getElementById('input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
</script>
</body>
</html>"""


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/wm/toggle", methods=["POST"])
def wm_toggle():
    global _wm_enabled
    data = request.get_json(force=True)
    _wm_enabled = bool(data.get("enabled", True))
    return jsonify({"wm_enabled": _wm_enabled})


@app.route("/wm/state")
def wm_state():
    return jsonify(_wm_last_display)


@app.route("/chat/clear", methods=["POST"])
def chat_clear():
    global _session_history, _wm_h, _wm_last_rewards, _wm_last_action_idx, _wm_last_display
    with _session_lock:
        _session_history = []
    with _wm_lock:
        _wm_h = None
        _wm_last_rewards = None
        _wm_last_action_idx = 0
        _wm_last_display = {}
    return jsonify({"ok": True})


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    body = request.get_json(force=True)
    user_msg = body.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "empty"}), 400

    def generate():
        global _session_history

        # ── 1. World Model step ──────────────────────────────────────────────
        try:
            wm_display, internals = _wm_step(user_msg)
            yield f"data: {json.dumps({'type': 'wm_state', 'data': wm_display})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'WM step error: {e}'})}\n\n"
            return

        # ── 2. Generar respuesta con el WM (sin LLM) ─────────────────────────
        try:
            with _session_lock:
                history_snapshot = list(_session_history)

            response, meta = get_generator().generate(
                user_message=user_msg,
                rewards=internals["rewards"],
                z_vector=internals["z"],
                session_history=history_snapshot,
                h_state=internals.get("h"),
            )

            # Adjuntar metadata de recuperación al display
            wm_display["retrieval"] = meta
            yield f"data: {json.dumps({'type': 'wm_state', 'data': wm_display})}\n\n"

            # Simular stream palabra a palabra para que se vea natural
            words = response.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Generator error: {e}'})}\n\n"
            return

        # ── 3. Guardar en historial de sesión ───────────────────────────────
        with _session_lock:
            _session_history.append({"role": "user", "content": user_msg})
            _session_history.append({"role": "assistant", "content": response})
            if len(_session_history) > 40:
                _session_history = _session_history[-40:]

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Crear carpeta de pesos si no existe
    Path("worldmodel/weights").mkdir(parents=True, exist_ok=True)
    print("=" * 50)
    print("  STELLA -- WORLD MODEL DEMO (sin LLM)")
    print("  http://localhost:5001")
    print("  WM:  RSSM + retrieval semantico sobre historial de Stella")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, threaded=True, debug=False)
