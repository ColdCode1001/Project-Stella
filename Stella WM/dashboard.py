"""
Stella Dashboard -- HUD control panel.
Flask + SSE, puerto 5000.
"""

import json
import queue
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, "D:/stella")

from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory, stream_with_context

from scripts.health_monitor import HealthMonitor
from core.logger import get_logger

log = get_logger("dashboard")
app = Flask(__name__)
monitor = HealthMonitor()

LOGS_DIR   = Path("D:/stella/logs")
LOG_ALL    = LOGS_DIR / "stella.log"
LOG_ERRORS = LOGS_DIR / "stella-errors.log"

_cfg_path = Path("D:/stella/config/stella.yaml")
with open(_cfg_path, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

LLM_ENDPOINT = _cfg["llm"]["endpoint"]
LLM_MODEL    = _cfg["llm"]["model"]
LLM_API_KEY  = _cfg["llm"]["api_key"]

SOUL_FILE     = Path("D:/stella/memory/store/stella.soul")
CHATS_FILE    = Path("D:/stella/memory/store/stella.chats.jsonl")
THOUGHTS_FILE = Path("D:/stella/memory/store/stella.thoughts.jsonl")
SESSION_FILE  = Path("D:/stella/memory/store/stella.session.json")

AUTO_CONSOLIDATE_PCT     = 70    # % de contexto usado que dispara auto-consolidación
_auto_consolidate_enabled = True  # toggle desde el dashboard

_chats_lock    = threading.Lock()
_thoughts_lock = threading.Lock()


def _persist_chat(role: str, content: str, speaker: str = "arca"):
    """Escribe un mensaje al log permanente de chats."""
    try:
        entry = {
            "ts":      __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "role":    role,
            "speaker": speaker,
            "content": content,
        }
        with _chats_lock:
            with CHATS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("persist_chat: %s", e)


def _persist_thought(content: str, marked: bool):
    """Escribe un pensamiento idle al log permanente."""
    try:
        entry = {
            "ts":      __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "content": content,
            "marked":  marked,
        }
        with _thoughts_lock:
            with THOUGHTS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("persist_thought: %s", e)


WEB_LOG_FILE = Path("D:/stella/memory/store/stella.web.jsonl")
_web_log_lock = threading.Lock()


def _persist_web_query(trigger: str, plan: list[dict], citations: list[str]):
    """Registra una sesión de búsqueda web (qué disparó, qué buscó, URLs vistas)."""
    try:
        entry = {
            "ts":         __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "trigger":    (trigger or "")[:200],
            "plan":       [{"action": p["action"], "arg": p["arg"][:200]} for p in plan],
            "citations":  citations[:10],
        }
        with _web_log_lock:
            with WEB_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("persist_web_query: %s", e)


_session_lock = threading.Lock()

def _save_session():
    """Guarda _session_history en disco tras cada mensaje."""
    try:
        with _session_lock:
            data = {
                "ts":      __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "history": list(_session_history),
                "marked":  _marked_count,
            }
        SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("save_session: %s", e)


def _load_saved_session() -> list[dict]:
    """Lee la sesión guardada en disco si existe."""
    try:
        if SESSION_FILE.exists():
            data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            return data.get("history", [])
    except Exception as e:
        log.warning("load_saved_session: %s", e)
    return []


def _clear_session_file():
    try:
        SESSION_FILE.write_text(json.dumps({"ts": "", "history": [], "marked": 0}), encoding="utf-8")
    except Exception:
        pass


def _load_soul() -> str:
    try:
        return SOUL_FILE.read_text(encoding="utf-8").strip()
    except Exception as e:
        log.warning("No se pudo cargar stella.soul: %s", e)
        return ""

_soul_text        = _load_soul()
_thinking_enabled        = False
_memory_enabled          = True           # inyectar episodios en contexto
_internet_enabled        = bool(_cfg.get("internet", {}).get("enabled", False))
_agent_enabled           = False          # idle agéntico (deliberate→act→reflect) en coordinator
_auto_approve_experiments = False         # ejecutar [🧪] sin pedir confirmación
_session_history: list[dict] = []  # [{role, content}, ...] — se limpia al consolidar
_marked_count     = 0              # [✦] detectados esta sesion
_fine_tune_marks: list[int] = []   # indices de intercambios marcados para fine-tuning
_image_queue:     list[dict] = []  # prompts [🎨] acumulados por Stella
_image_lock       = threading.Lock()
_cmd_queue:       list[dict] = []  # comandos [⚙️] pendientes de aprobación
_cmd_lock         = threading.Lock()
_code_queue:      list[dict] = []  # bloques [🧪 python] pendientes de aprobación
_code_lock        = threading.Lock()

# -- coordinator experiment result callback -----------------------------------

def _notify_coordinator_result(code: str, result: dict):
    """Notifica al coordinator el resultado de un [🧪] auto-ejecutado."""
    stdout = result.get("stdout", "").strip()
    stderr = result.get("stderr", "").strip()
    rc     = result.get("returncode", -1)
    summary = stdout[:500] if stdout else (f"[error] {stderr[:300]}" if stderr else f"[rc={rc}]")
    try:
        httpx.post(
            "http://localhost:5002/experiment_result",
            json={"code": code[:300], "result": summary, "rc": rc},
            timeout=3,
        )
    except Exception:
        pass


# -- SSE fan-out --------------------------------------------------------------

_status_subs: list[queue.Queue] = []
_log_subs:    list[queue.Queue] = []
_err_subs:    list[queue.Queue] = []
_vrchat_subs: list[queue.Queue] = []
_agent_subs:  list[queue.Queue] = []
_thought_subs: list[queue.Queue] = []


def _vrchat_broadcast(event_type: str, **kwargs):
    _broadcast(_vrchat_subs, {"type": event_type, **kwargs})


def _agent_broadcast(event_type: str, label: str, **kwargs):
    """
    Emite un evento de actividad del agente. Tipos:
      planning, searching, reading, saving, thinking, done, idle_start,
      idle_done, web_used, error.
    label es el texto humano corto para mostrar en la barra.
    """
    _broadcast(_agent_subs, {
        "type":  event_type,
        "label": label,
        "ts":    time.time(),
        **kwargs,
    })


def _thought_broadcast(content: str, *, marked: bool = False, mode: str = "idle"):
    """Emite el contenido del último idle thought al panel live."""
    _broadcast(_thought_subs, {
        "content": content[:4000],
        "marked":  marked,
        "mode":    mode,
        "ts":      time.time(),
    })


def _broadcast(clients, data):
    for q in list(clients):
        try:
            q.put_nowait(data)
        except queue.Full:
            pass


def _fetch_coordinator() -> dict:
    try:
        r = httpx.get("http://localhost:5002/status", timeout=2)
        if r.status_code == 200:
            d = r.json()
            return {
                "running":        True,
                "mode":           d.get("mode", "idle"),
                "thoughts_today": d.get("thoughts_today", 0),
                "idle_seconds":   d.get("idle_seconds", 0),
                "osc_enabled":    d.get("osc_enabled", False),
                "vts_connected":  d.get("vts_connected", False),
            }
    except Exception:
        pass
    return {"running": False, "mode": "off", "thoughts_today": 0, "idle_seconds": 0}


def _poll_status():
    while True:
        try:
            _broadcast(_status_subs, {
                "services":    monitor.get_all(),
                "master":      monitor.master_status(),
                "thinking":    _thinking_enabled,
                "internet":    _internet_enabled,
                "agent":       _agent_enabled,
                "auto_experiments": _auto_approve_experiments,
                "coordinator": _fetch_coordinator(),
            })
        except Exception as e:
            log.error("poll_status: %s", e)
        time.sleep(4)


def _tail(path: Path, clients, label: str):
    pos = path.stat().st_size if path.exists() else 0
    while True:
        try:
            if path.exists():
                with open(path, encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read(16384)
                    if chunk:
                        for line in chunk.splitlines():
                            if line.strip():
                                _broadcast(clients, {"line": line, "src": label})
                    pos = f.tell()
        except Exception as e:
            log.warning("tail %s: %s", path, e)
        time.sleep(1)


threading.Thread(target=_poll_status, daemon=True).start()
threading.Thread(target=_tail, args=(LOG_ALL,    _log_subs, "stella"),  daemon=True).start()
threading.Thread(target=_tail, args=(LOG_ERRORS, _err_subs, "errors"),  daemon=True).start()

log.info("Dashboard iniciado en http://localhost:5000")

# -- HTML ---------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stella // Control</title>
<style>
:root {
  --bg:    #000;
  --card:  #050505;
  --b0:    #1a1a1a;
  --b1:    #2a2a2a;
  --b2:    #555;
  --white: #fff;
  --dim:   #aaa;
  --dim2:  #666;
  --dim3:  #444;
  --red:   #d01818;
  --red2:  #ff2a2a;
  --amber: #999;
  --cut:   12px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--dim);
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 12px;
  min-height: 100vh;
  /* subtle scanline */
  background-image: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(255,255,255,.008) 2px, rgba(255,255,255,.008) 4px
  );
}

/* =========================================================
   HEADER
   ========================================================= */
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 22px;
  border-bottom: 1px solid var(--b1);
  background: #000;
  position: sticky; top: 0; z-index: 20;
  gap: 12px;
}

.brand { display: flex; align-items: center; gap: 16px; flex-shrink: 0; }
.brand-name {
  font-size: 16px; letter-spacing: 8px; color: var(--white);
  text-transform: uppercase; font-weight: normal;
}
.brand-line { width: 1px; height: 22px; background: var(--b1); }
.brand-sub  { font-size: 9px; letter-spacing: 4px; color: var(--dim2); text-transform: uppercase; }

/* nav links */
.nav-links { display: flex; align-items: center; gap: 6px; margin-left: 4px; }
.nav-link {
  font-size: 9px; letter-spacing: 2px; color: var(--dim3); text-decoration: none;
  border: 1px solid var(--b0); padding: 4px 10px; text-transform: uppercase;
  transition: .15s; position: relative;
}
.nav-link::before { content:''; position:absolute; top:-1px; left:-1px; width:5px; height:5px; border-top:1px solid var(--b2); border-left:1px solid var(--b2); }
.nav-link:hover { color: var(--white); border-color: var(--b2); }

.hdr-right { display: flex; align-items: center; gap: 18px; }

/* thinking toggle */
.think-row { display: flex; align-items: center; gap: 8px; }
.think-tag {
  font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--dim2);
  transition: color .2s;
}
.think-tag.on { color: var(--white); }

/* osc switch */
.osc-wrap { display: flex; align-items: center; gap: 8px; }
.osc-tag  {
  font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--dim3);
  transition: color .2s;
}
.osc-tag.on { color: var(--red2); }

/* coordinator mode indicator */
.mode-wrap { display: flex; align-items: center; gap: 7px; }
.mode-badge {
  font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
  border: 1px solid var(--b0); padding: 3px 10px; color: var(--dim3);
  transition: color .3s, border-color .3s; position: relative;
}
.mode-badge::before {
  content: ''; position: absolute; top: -1px; left: -1px;
  width: 5px; height: 5px;
  border-top: 1px solid; border-left: 1px solid; border-color: inherit;
}
.mode-badge.idle     { color: var(--amber); border-color: var(--amber); animation: flicker 2.4s ease-in-out infinite; }
.mode-badge.reactive { color: var(--white); border-color: var(--b2); }
.mode-badge.off      { color: var(--dim3); border-color: var(--b0); }
.mode-thoughts { font-size: 9px; color: var(--dim3); letter-spacing: 1px; }

/* stop all */
.btn-kill {
  background: none; font-family: inherit;
  border: 1px solid var(--red); color: var(--red);
  font-size: 9px; letter-spacing: 3px; text-transform: uppercase;
  padding: 5px 14px; cursor: pointer; transition: .15s;
  position: relative;
}
.btn-kill::before, .btn-kill::after {
  content: '';
  position: absolute;
  width: 5px; height: 5px;
}
.btn-kill::before { top: -1px; left: -1px; border-top: 1px solid var(--red2); border-left: 1px solid var(--red2); }
.btn-kill::after  { bottom: -1px; right: -1px; border-bottom: 1px solid var(--red2); border-right: 1px solid var(--red2); }
.btn-kill:hover { background: rgba(208,24,24,.1); color: var(--red2); }

/* master */
.master { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.mled {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--b1); border: 1px solid var(--b2);
  transition: background .3s, box-shadow .3s;
}
.mled.green  { background: var(--white); border-color: var(--white); box-shadow: 0 0 8px var(--white), 0 0 20px rgba(255,255,255,.25); animation: breathe 2.4s ease-in-out infinite; }
.mled.yellow { background: var(--amber); border-color: var(--amber); animation: flicker 1.6s ease-in-out infinite; }
.mled.red    { background: var(--red2);  border-color: var(--red2);  box-shadow: 0 0 8px var(--red2); animation: blink .7s step-end infinite; }
.mlabel { font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--dim2); }
.mlabel.green  { color: var(--white); }
.mlabel.yellow { color: var(--amber); }
.mlabel.red    { color: var(--red2); }

/* =========================================================
   SECTION SEPARATOR
   ========================================================= */
.sep {
  display: flex; align-items: center; gap: 10px;
  padding: 18px 22px 10px;
  font-size: 9px; letter-spacing: 5px; text-transform: uppercase; color: var(--dim3);
}
.sep::after { content: ''; flex: 1; height: 1px; background: var(--b0); }

/* =========================================================
   CARDS GRID
   ========================================================= */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
  padding: 0 22px;
}

/* HUD card with corner brackets */
.card {
  background: var(--card);
  border: 1px solid var(--b0);
  padding: 16px 16px 13px;
  display: flex; flex-direction: column; gap: 10px;
  position: relative;
  transition: border-color .3s;
}

/* corner brackets via ::before (top-left) and ::after (bottom-right) */
.card::before, .card::after {
  content: '';
  position: absolute;
  width: 10px; height: 10px;
  pointer-events: none;
  transition: border-color .3s;
}
.card::before { top: -1px; left: -1px;   border-top: 1px solid var(--b2);  border-left: 1px solid var(--b2); }
.card::after  { bottom: -1px; right: -1px; border-bottom: 1px solid var(--b2); border-right: 1px solid var(--b2); }

/* top accent line -- the "angled" feel */
.c-accent {
  position: absolute; top: -1px;
  left: 10px; right: 40px;
  height: 1px;
  background: var(--b1);
  transition: background .35s, box-shadow .35s;
}
/* diagonal tick at top-right (mimics angled cut) */
.c-tick {
  position: absolute; top: -1px; right: -1px;
  width: 40px; height: 1px;
  background: var(--b0);
  transform-origin: right center;
  transform: rotate(-45deg) scaleX(.6);
}

/* status-driven styles */
.card.green { border-color: var(--b1); }
.card.green::before, .card.green::after { border-color: var(--white); }
.card.green .c-accent { background: var(--white); box-shadow: 0 0 10px rgba(255,255,255,.4); }

.card.yellow { border-color: var(--b0); }
.card.yellow::before, .card.yellow::after { border-color: var(--amber); }
.card.yellow .c-accent { background: var(--amber); }

.card.red { border-color: rgba(208,24,24,.3); }
.card.red::before, .card.red::after { border-color: var(--red2); }
.card.red .c-accent { background: var(--red); box-shadow: 0 0 8px var(--red); animation: blink .7s step-end infinite; }

/* card internals */
.c-top { display: flex; align-items: center; justify-content: space-between; }
.c-led-row { display: flex; align-items: center; gap: 7px; }

.led {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--b1); border: 1px solid var(--b2);
  flex-shrink: 0; transition: background .3s, box-shadow .3s;
}
.led.green  { background: var(--white); border-color: var(--white); box-shadow: 0 0 6px var(--white), 0 0 14px rgba(255,255,255,.2); animation: breathe 2.4s ease-in-out infinite; }
.led.yellow { background: var(--amber); border-color: var(--amber); animation: flicker 1.6s ease-in-out infinite; }
.led.red    { background: var(--red2);  border-color: var(--red2);  box-shadow: 0 0 6px var(--red2); animation: blink .7s step-end infinite; }

.led-tag { font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--dim3); transition: color .3s; }
.led-tag.green  { color: var(--white); }
.led-tag.yellow { color: var(--amber); }
.led-tag.red    { color: var(--red2); }

.lat { font-size: 9px; color: var(--dim3); min-width: 50px; text-align: right; font-variant-numeric: tabular-nums; }
.lat.on { color: var(--dim2); }

.c-name { font-size: 13px; letter-spacing: 4px; text-transform: uppercase; color: var(--white); }
.c-name.crit::after { content: " //!"; font-size: 9px; color: var(--red); letter-spacing: 2px; }
.c-desc  { font-size: 10px; color: var(--dim2); letter-spacing: .5px; }
.c-err   { font-size: 10px; color: var(--red2); min-height: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.c-err:not(:empty)::before { content: "> "; color: var(--red); }

/* card footer with switch */
.c-foot {
  display: flex; align-items: center; justify-content: space-between;
  padding-top: 8px; border-top: 1px solid var(--b0);
}
.sw-label { font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--dim3); }
.sw-label.on { color: var(--dim2); }

/* TOGGLE SWITCH -- angular HUD style */
.sw { position: relative; width: 42px; height: 20px; cursor: pointer; }
.sw input { opacity: 0; width: 0; height: 0; position: absolute; }
.sw-bg {
  position: absolute; inset: 0;
  background: #000; border: 1px solid var(--b1);
  transition: border-color .2s;
}
.sw-knob {
  position: absolute;
  width: 12px; height: 12px;
  top: 3px; left: 3px;
  background: var(--b2);
  transition: left .2s, background .2s, box-shadow .2s;
}
/* corner accent on switch */
.sw-bg::before {
  content: '';
  position: absolute; top: -1px; left: -1px;
  width: 5px; height: 5px;
  border-top: 1px solid var(--b2); border-left: 1px solid var(--b2);
}
.sw input:checked ~ .sw-bg { border-color: var(--white); }
.sw input:checked ~ .sw-bg::before { border-color: var(--white); }
.sw input:checked ~ .sw-knob { left: 27px; background: var(--white); box-shadow: 0 0 5px rgba(255,255,255,.5); }
.sw:hover .sw-bg { border-color: var(--dim); }

/* header-size small switch */
.sw-sm { width: 32px; height: 16px; }
.sw-sm .sw-knob { width: 8px; height: 8px; top: 3px; left: 3px; }
.sw-sm input:checked ~ .sw-knob { left: 21px; }

/* =========================================================
   LOG PANELS
   ========================================================= */
.log-row {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 10px; padding: 0 22px;
}

.log-panel {
  background: var(--card);
  border: 1px solid var(--b0);
  display: flex; flex-direction: column;
  height: 190px;
  position: relative;
}
.log-panel::before {
  content: '';
  position: absolute; top: -1px; left: -1px;
  width: 8px; height: 8px;
  border-top: 1px solid var(--b2); border-left: 1px solid var(--b2);
  pointer-events: none;
}
.log-panel.is-err::before { border-color: var(--red); }

.log-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 12px; border-bottom: 1px solid var(--b0);
  font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--dim2);
  flex-shrink: 0;
}
.log-dot { display: inline-block; width: 4px; height: 4px; background: var(--b2); margin-right: 8px; vertical-align: middle; }
.log-panel.is-err .log-dot { background: var(--red); box-shadow: 0 0 4px var(--red); }

.clr-btn {
  background: none; border: 1px solid var(--b0); color: var(--dim2);
  font-family: inherit; font-size: 9px; letter-spacing: 1px; padding: 1px 6px;
  cursor: pointer; transition: .15s;
}
.clr-btn:hover { border-color: var(--b2); color: var(--dim); }

.log-body {
  flex: 1; overflow-y: auto; padding: 6px 12px;
  display: flex; flex-direction: column-reverse;
  scrollbar-width: thin; scrollbar-color: var(--b1) transparent;
}
.log-line { line-height: 1.7; color: var(--dim3); font-size: 11px; white-space: pre-wrap; word-break: break-all; }
.log-line.err  { color: var(--dim); }
.log-line.warn { color: var(--dim2); }

/* =========================================================
   CHAT / TERMINAL
   ========================================================= */
.chat-panel {
  margin: 0 22px 24px;
  border: 1px solid var(--b0);
  background: var(--card);
  display: flex; flex-direction: column;
  position: relative;
}
.chat-panel::before, .chat-panel::after {
  content: ''; position: absolute; width: 8px; height: 8px; pointer-events: none;
}
.chat-panel::before { top: -1px; left: -1px; border-top: 1px solid var(--b2); border-left: 1px solid var(--b2); }
.chat-panel::after  { bottom: -1px; right: -1px; border-bottom: 1px solid var(--b2); border-right: 1px solid var(--b2); }

.chat-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 12px; border-bottom: 1px solid var(--b0);
  font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--dim2);
  flex-shrink: 0;
}
.think-badge {
  font-size: 9px; letter-spacing: 2px; color: var(--b2);
  border: 1px solid var(--b0); padding: 1px 8px;
}
.think-badge.on { color: var(--dim); border-color: var(--b2); }

.chat-body {
  height: 200px; overflow-y: auto; padding: 10px 14px;
  display: flex; flex-direction: column; gap: 10px;
  scrollbar-width: thin; scrollbar-color: var(--b1) transparent;
}
.chat-msg { display: flex; flex-direction: column; gap: 3px; }
.chat-who { font-size: 9px; letter-spacing: 3px; text-transform: uppercase; }
.chat-who.u { color: var(--dim); }
.chat-who.s { color: var(--dim2); }
.chat-text { font-size: 12px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }
.chat-text.u { color: var(--white); }
.chat-text.s { color: var(--dim); }
.memory-mark { color:#555; font-style:italic; display:block; margin-top:6px; padding-left:8px; border-left:1px solid #2a2a2a; font-size:11px; }
.chat-think {
  font-size: 11px; color: var(--b2); line-height: 1.6; font-style: italic;
  border-left: 2px solid var(--red); padding-left: 8px;
  margin-top: 2px; white-space: pre-wrap;
}

.chat-foot { display: flex; border-top: 1px solid var(--b0); flex-shrink: 0; }
#chat-in {
  flex: 1; background: transparent; border: none; outline: none;
  color: var(--white); font-family: inherit; font-size: 12px;
  padding: 10px 14px; resize: none; height: 40px;
  caret-color: var(--white);
}
#chat-in::placeholder { color: var(--b1); }
.btn-send {
  background: none; border: none; border-left: 1px solid var(--b0);
  color: var(--dim2); font-family: inherit; font-size: 9px;
  letter-spacing: 3px; text-transform: uppercase;
  padding: 0 18px; cursor: pointer; transition: .15s; flex-shrink: 0;
}
.btn-send:hover   { color: var(--white); background: rgba(255,255,255,.03); }
.btn-send:disabled{ color: var(--b1); cursor: default; }

/* =========================================================
   ANIMATIONS
   ========================================================= */
@keyframes breathe { 0%,100%{opacity:1} 50%{opacity:.4} }
@keyframes flicker { 0%,100%{opacity:1} 40%{opacity:.5} 60%{opacity:.8} }
@keyframes blink   { 0%,49%{opacity:1} 50%,100%{opacity:0} }
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="brand">
    <span class="brand-name">Stella</span>
    <div class="brand-line"></div>
    <span class="brand-sub">control center</span>
    <nav class="nav-links">
      <a href="/chat"     class="nav-link">CHAT</a>
      <a href="/research" class="nav-link">RESEARCH</a>
      <a href="/library"  class="nav-link">LIBRARY</a>
      <a href="/history"  class="nav-link">HISTORIAL</a>
      <a href="/vrchat"   class="nav-link">VRCHAT</a>
    </nav>
  </div>
  <div class="hdr-right">
    <div class="think-row">
      <label class="sw sw-sm">
        <input type="checkbox" id="sw-thinking" onchange="toggleThinking(this)">
        <div class="sw-bg"></div>
        <div class="sw-knob"></div>
      </label>
      <span class="think-tag" id="think-tag">THINKING OFF</span>
    </div>
    <div class="osc-wrap">
      <label class="sw sw-sm">
        <input type="checkbox" id="sw-osc" onchange="toggleOSC(this)">
        <div class="sw-bg"></div>
        <div class="sw-knob"></div>
      </label>
      <span class="osc-tag" id="osc-tag">OSC OFF</span>
    </div>
    <div class="osc-wrap">
      <label class="sw sw-sm">
        <input type="checkbox" id="sw-internet" onchange="toggleInternet(this)">
        <div class="sw-bg"></div>
        <div class="sw-knob"></div>
      </label>
      <span class="osc-tag" id="internet-tag">INTERNET OFF</span>
    </div>
    <div class="osc-wrap">
      <label class="sw sw-sm">
        <input type="checkbox" id="sw-agent" onchange="toggleAgent(this)">
        <div class="sw-bg"></div>
        <div class="sw-knob"></div>
      </label>
      <span class="osc-tag" id="agent-tag">AGENT OFF</span>
    </div>
    <div class="osc-wrap">
      <label class="sw sw-sm">
        <input type="checkbox" id="sw-autoexp" onchange="toggleAutoExp(this)">
        <div class="sw-bg"></div>
        <div class="sw-knob"></div>
      </label>
      <span class="osc-tag" id="autoexp-tag">🧪 MANUAL</span>
    </div>
    <div class="osc-wrap">
      <span class="osc-tag" id="vts-tag">VTS --</span>
      <button id="btn-vts" onclick="reconnectVTS()" style="background:none;border:1px solid var(--b1);color:var(--dim3);font-family:inherit;font-size:8px;letter-spacing:2px;padding:2px 6px;cursor:pointer;text-transform:uppercase;" title="Reconectar VTube Studio">↺</button>
    </div>
    <div class="mode-wrap" id="mode-wrap" style="display:none">
      <span class="mode-badge off" id="mode-badge">--</span>
      <span class="mode-thoughts" id="mode-thoughts"></span>
    </div>
    <button class="btn-kill" onclick="stopAll()">STOP ALL</button>
    <div class="master">
      <div class="mled off" id="mled"></div>
      <span class="mlabel off" id="mlabel">CONNECTING</span>
    </div>
  </div>
</header>

<!-- AGENT ACTIVITY TRACE -->
<div class="sep" style="display:flex;justify-content:space-between;align-items:center;">
  <span>ACTIVIDAD DE STELLA</span>
  <span id="agent-empty" style="font-size:9px;color:var(--dim3);letter-spacing:2px;">— en reposo —</span>
</div>
<div id="agent-trace" style="display:flex;gap:6px;padding:8px 14px;min-height:38px;overflow-x:auto;border-bottom:1px solid var(--b1);background:#0a0a0a;align-items:center;flex-wrap:nowrap;"></div>

<!-- NUDGE (Arca sugiere foco a Stella sin interrumpir idle) -->
<div style="display:flex;gap:8px;padding:8px 14px;border-bottom:1px solid var(--b1);background:#0a0a0a;align-items:center;">
  <span style="font-size:9px;color:var(--dim3);letter-spacing:2px;white-space:nowrap;">▸ SUGERIR A STELLA</span>
  <input type="text" id="nudge-input" placeholder="una sugerencia que vea en el próximo idle…"
         maxlength="500"
         style="flex:1;background:transparent;border:1px solid var(--b1);color:var(--white);font-family:inherit;font-size:11px;padding:5px 10px;letter-spacing:.5px;"
         onkeydown="if(event.key==='Enter')sendNudge()">
  <button onclick="sendNudge()"
          style="background:transparent;border:1px solid var(--b1);color:var(--dim);font-family:inherit;font-size:9px;letter-spacing:3px;padding:5px 14px;cursor:pointer;text-transform:uppercase;">
    ENVIAR
  </button>
  <span id="nudge-pending" style="font-size:8px;color:var(--dim3);letter-spacing:2px;min-width:80px;text-align:right;"></span>
</div>

<!-- LIVE THOUGHT (debug) -->
<div class="sep" style="display:flex;justify-content:space-between;align-items:center;">
  <span>ULTIMO PENSAMIENTO IDLE</span>
  <span id="thought-mode" style="font-size:9px;color:var(--dim3);letter-spacing:2px;">— sin actividad —</span>
</div>
<div id="thought-live" style="padding:10px 14px;font-size:11px;color:var(--dim1);line-height:1.5;background:#0a0a0a;border-bottom:1px solid var(--b1);min-height:38px;max-height:140px;overflow-y:auto;white-space:pre-wrap;font-style:italic;"></div>

<!-- SECURITY COMMAND QUEUE -->
<div class="sep" style="display:flex;justify-content:space-between;align-items:center;">
  <span>COMANDOS ⚙️</span>
  <span style="display:flex;gap:8px;align-items:center;">
    <span id="cmd-badge" style="font-size:9px;letter-spacing:2px;display:none;color:#e8a000;">● PENDIENTE</span>
    <button onclick="securityClear()" style="background:none;border:1px solid var(--b1);color:var(--dim3);font-family:inherit;font-size:8px;letter-spacing:2px;padding:2px 8px;cursor:pointer;text-transform:uppercase;">CLR</button>
  </span>
</div>
<div id="cmd-queue-wrap" style="background:#0a0a0a;border-bottom:1px solid var(--b1);min-height:28px;">
  <div id="cmd-queue-list" style="padding:6px 14px;">
    <span style="font-size:9px;color:var(--dim3);letter-spacing:2px;font-style:italic;">sin comandos pendientes</span>
  </div>
</div>

<!-- EXPERIMENTS CODE QUEUE -->
<div class="sep" style="display:flex;justify-content:space-between;align-items:center;">
  <span>EXPERIMENTOS 🧪</span>
  <span style="display:flex;gap:8px;align-items:center;">
    <span id="code-badge" style="font-size:9px;letter-spacing:2px;display:none;color:#3a8a5a;">● PENDIENTE</span>
    <button onclick="experimentsClear()" style="background:none;border:1px solid var(--b1);color:var(--dim3);font-family:inherit;font-size:8px;letter-spacing:2px;padding:2px 8px;cursor:pointer;text-transform:uppercase;">CLR</button>
  </span>
</div>
<div id="code-queue-wrap" style="background:#0a0a0a;border-bottom:1px solid var(--b1);min-height:28px;">
  <div id="code-queue-list" style="padding:6px 14px;">
    <span style="font-size:9px;color:var(--dim3);letter-spacing:2px;font-style:italic;">sin experimentos pendientes</span>
  </div>
</div>

<!-- MODULES -->
<div class="sep">MODULOS</div>
<div class="grid" id="svc-grid"></div>

<!-- TELEMETRY -->
<div class="sep">TELEMETRIA</div>
<div class="log-row">
  <div class="log-panel">
    <div class="log-head">
      <span><span class="log-dot"></span>SISTEMA / stella.log</span>
      <button class="clr-btn" onclick="clr('log-b')">CLR</button>
    </div>
    <div class="log-body" id="log-b"></div>
  </div>
  <div class="log-panel is-err">
    <div class="log-head">
      <span><span class="log-dot"></span>ERRORES / stella-errors.log</span>
      <span style="display:flex;gap:4px">
        <button class="clr-btn" onclick="copyErrors()" title="copiar al portapapeles">CPY</button>
        <button class="clr-btn" onclick="clr('err-b')">CLR</button>
      </span>
    </div>
    <div class="log-body" id="err-b"></div>
  </div>
</div>

<!-- TERMINAL -->
<div class="sep">TERMINAL</div>
<div class="chat-panel">
  <div class="chat-head">
    <span>CHAT DIRECTO // llama-server :8080</span>
    <span class="think-badge" id="think-badge">THINKING OFF</span>
  </div>
  <div class="chat-body" id="chat-b"></div>
  <div class="chat-foot">
    <textarea id="chat-in" placeholder="> input..."></textarea>
    <button class="btn-send" id="btn-send" onclick="sendChat()">ENVIAR</button>
  </div>
</div>

<script>
const MLABELS   = {green:'ONLINE', yellow:'DEGRADADO', red:'CRITICO', off:'OFFLINE'};
const LED_TEXT  = {green:'ONLINE', yellow:'INICIANDO', red:'ERROR', off:'APAGADO'};
let rendered    = {};
let generating  = false;

// ---- service grid ----------------------------------------------------------
function buildCard(name, info) {
  const d = document.createElement('div');
  d.id = 'card-' + name;
  d.className = 'card off';
  d.innerHTML = `
    <div class="c-accent"></div>
    <div class="c-tick"></div>
    <div class="c-top">
      <div class="c-led-row">
        <div class="led off" id="led-${name}"></div>
        <span class="led-tag off" id="ltag-${name}">----</span>
      </div>
      <span class="lat" id="lat-${name}">-- ms</span>
    </div>
    <div class="c-name${info.critical?' crit':''}">${info.label}</div>
    <div class="c-desc">${info.description}</div>
    <div class="c-err" id="cerr-${name}"></div>
    <div class="c-foot">
      <span class="sw-label" id="swl-${name}">APAGADO</span>
      <label class="sw">
        <input type="checkbox" id="sw-${name}" onchange="toggle('${name}',this)">
        <div class="sw-bg"></div>
        <div class="sw-knob"></div>
      </label>
    </div>`;
  document.getElementById('svc-grid').appendChild(d);
}

function updateCard(name, info) {
  const s = info.status;
  const card = document.getElementById('card-' + name);
  if (!card) return;
  card.className = 'card ' + s;
  document.getElementById('led-' + name).className  = 'led ' + s;
  const ltag = document.getElementById('ltag-' + name);
  ltag.className   = 'led-tag ' + s;
  ltag.textContent = LED_TEXT[s] || s;
  const lat = document.getElementById('lat-' + name);
  lat.textContent  = info.latency_ms ? info.latency_ms + ' ms' : '-- ms';
  lat.className    = 'lat' + (info.latency_ms ? ' on' : '');
  document.getElementById('cerr-' + name).textContent = info.last_error || '';
  const on = s === 'green' || s === 'yellow';
  document.getElementById('sw-' + name).checked = on;
  const swl = document.getElementById('swl-' + name);
  swl.textContent = on ? 'ACTIVO' : 'APAGADO';
  swl.className   = 'sw-label' + (on ? ' on' : '');
}

function updateMaster(status) {
  document.getElementById('mled').className   = 'mled ' + status;
  const lbl = document.getElementById('mlabel');
  lbl.className   = 'mlabel ' + status;
  lbl.textContent = MLABELS[status] || status;
}

// ---- controls --------------------------------------------------------------
async function toggle(name, cb) {
  const act = cb.checked ? 'start' : 'stop';
  cb.disabled = true;
  try {
    const r = await fetch('/service/' + name + '/' + act, {method:'POST'});
    if (!(await r.json()).ok) cb.checked = !cb.checked;
  } catch { cb.checked = !cb.checked; }
  finally { cb.disabled = false; }
}

async function stopAll() {
  if (!confirm('Detener todos los servicios activos?')) return;
  await fetch('/service/all/stop', {method:'POST'});
}

async function toggleOSC(cb) {
  try {
    const r = await fetch('/coordinator/osc', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({enabled: cb.checked})
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    applyOSCUI(d.enabled);
  } catch(e) {
    cb.checked = !cb.checked;
    console.warn('coordinator no disponible para toggle OSC:', e.message);
  }
}

function applyOSCUI(on) {
  const sw  = document.getElementById('sw-osc');
  const tag = document.getElementById('osc-tag');
  if (sw) sw.checked = on;
  if (tag) {
    tag.textContent = on ? 'OSC ON' : 'OSC OFF';
    tag.className   = 'osc-tag' + (on ? ' on' : '');
  }
}

function applyVTSUI(connected) {
  const tag = document.getElementById('vts-tag');
  const btn = document.getElementById('btn-vts');
  if (!tag) return;
  tag.textContent = connected ? 'VTS ON' : 'VTS OFF';
  tag.className   = 'osc-tag' + (connected ? ' on' : '');
  if (btn) btn.style.display = connected ? 'none' : '';
}

async function reconnectVTS() {
  const btn = document.getElementById('btn-vts');
  if (btn) { btn.textContent = '...'; btn.disabled = true; }
  try {
    const r = await fetch('/coordinator/vts', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reconnect: true})
    });
    const d = await r.json();
    if (d.reconnecting) {
      setTimeout(() => {
        fetch('/coordinator/vts').then(r=>r.json()).then(d=>applyVTSUI(d.connected));
        if (btn) { btn.textContent = '↺'; btn.disabled = false; }
      }, 5000);
    }
  } catch(e) {
    if (btn) { btn.textContent = '↺'; btn.disabled = false; }
  }
}

async function toggleThinking(cb) {
  const r = await fetch('/config/thinking', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({enabled: cb.checked})
  });
  applyThinkingUI((await r.json()).enabled);
}

async function toggleInternet(cb) {
  try {
    const r = await fetch('/config/internet', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({enabled: cb.checked})
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    applyInternetUI(d.enabled);
  } catch(e) {
    cb.checked = !cb.checked;
    console.warn('toggle INTERNET failed:', e.message);
  }
}

function applyInternetUI(on) {
  const sw  = document.getElementById('sw-internet');
  const tag = document.getElementById('internet-tag');
  if (sw)  sw.checked = on;
  if (tag) {
    tag.textContent = on ? 'INTERNET ON' : 'INTERNET OFF';
    tag.className   = 'osc-tag' + (on ? ' on' : '');
  }
}

async function toggleAgent(cb) {
  try {
    const r = await fetch('/config/agent', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({enabled: cb.checked})
    });
    const d = await r.json();
    applyAgentUI(d.enabled);
  } catch(e) {
    cb.checked = !cb.checked;
    console.warn('toggle AGENT failed:', e.message);
  }
}

function applyAgentUI(on) {
  const sw  = document.getElementById('sw-agent');
  const tag = document.getElementById('agent-tag');
  if (sw)  sw.checked = on;
  if (tag) {
    tag.textContent = on ? 'AGENT ON' : 'AGENT OFF';
    tag.className   = 'osc-tag' + (on ? ' on' : '');
  }
}

async function toggleAutoExp(cb) {
  try {
    const r = await fetch('/config/auto_experiments', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({enabled: cb.checked})
    });
    const d = await r.json();
    applyAutoExpUI(d.enabled);
  } catch(e) {
    cb.checked = !cb.checked;
    console.warn('toggle AUTO_EXP failed:', e.message);
  }
}

function applyAutoExpUI(on) {
  const sw   = document.getElementById('sw-autoexp');
  const sw2  = document.getElementById('sw-autoexp2');
  const tag  = document.getElementById('autoexp-tag');
  if (sw)  sw.checked  = on;
  if (sw2) sw2.checked = on;
  if (tag) {
    tag.textContent = on ? '🧪 AUTO' : '🧪 MANUAL';
    tag.className   = 'osc-tag' + (on ? ' on' : '');
  }
}

function applyThinkingUI(on) {
  document.getElementById('sw-thinking').checked = on;
  const tag   = document.getElementById('think-tag');
  const badge = document.getElementById('think-badge');
  tag.textContent   = on ? 'THINKING ON' : 'THINKING OFF';
  tag.className     = 'think-tag' + (on ? ' on' : '');
  badge.textContent = on ? 'THINKING ON' : 'THINKING OFF';
  badge.className   = 'think-badge' + (on ? ' on' : '');
}

// ---- SSE -------------------------------------------------------------------
new EventSource('/events/status').onmessage = e => {
  const d = JSON.parse(e.data);
  for (const [name, info] of Object.entries(d.services)) {
    if (!rendered[name]) { rendered[name] = true; buildCard(name, info); }
    updateCard(name, info);
  }
  updateMaster(d.master);
  applyThinkingUI(d.thinking);
  if (typeof d.internet         !== 'undefined') applyInternetUI(d.internet);
  if (typeof d.agent            !== 'undefined') applyAgentUI(d.agent);
  if (typeof d.auto_experiments !== 'undefined') applyAutoExpUI(d.auto_experiments);
  if (d.coordinator) updateCoordinatorUI(d.coordinator);
};

function updateCoordinatorUI(c) {
  const wrap     = document.getElementById('mode-wrap');
  const badge    = document.getElementById('mode-badge');
  const thoughts = document.getElementById('mode-thoughts');
  if (!c.running) { wrap.style.display = 'none'; return; }
  wrap.style.display = '';
  const modeLabel = c.mode === 'idle' ? 'IDLE' : c.mode === 'reactive' ? 'REACTIVO' : '--';
  badge.textContent = modeLabel;
  badge.className   = 'mode-badge ' + (c.running ? c.mode : 'off');
  thoughts.textContent = c.thoughts_today > 0 ? c.thoughts_today + ' ideas' : '';
  if (typeof c.osc_enabled   !== 'undefined') applyOSCUI(c.osc_enabled);
  if (typeof c.vts_connected !== 'undefined') applyVTSUI(c.vts_connected);
}

function addLogLine(id, text, cls) {
  const body = document.getElementById(id);
  if (!body) return;
  const el = document.createElement('div');
  el.className = 'log-line ' + (cls||'');
  el.textContent = text;
  body.prepend(el);
  if (body.children.length > 300) body.lastChild.remove();
}

new EventSource('/events/log').onmessage = e => {
  const d = JSON.parse(e.data);
  const cls = d.line.includes(' ERROR ') ? 'err' : d.line.includes(' WARNING ') ? 'warn' : '';
  addLogLine('log-b', d.line, cls);
};

// nudge — Arca sugiere foco al próximo idle ---------------------------------
let _nudgeSending = false;
async function sendNudge() {
  if (_nudgeSending) return;  // anti-double-click
  const inp = document.getElementById('nudge-input');
  const msg = (inp.value || '').trim();
  if (!msg) return;
  _nudgeSending = true;
  inp.disabled = true;
  try {
    const r = await fetch('/coordinator/nudge', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: msg, reset_idle: false})
    });
    const d = await r.json();
    if (d.ok) {
      inp.value = '';
      updateNudgePending(d.queue_size);
    } else {
      console.warn('nudge:', d);
    }
  } catch(e) {
    console.warn('nudge failed:', e.message);
  } finally {
    _nudgeSending = false;
    inp.disabled = false;
    inp.focus();
  }
}
function updateNudgePending(n) {
  const el = document.getElementById('nudge-pending');
  if (!el) return;
  if (n > 0) {
    el.textContent = n + ' EN COLA';
    el.style.color = 'var(--white)';
  } else {
    el.textContent = '';
  }
}
async function pollNudgeQueue() {
  try {
    const r = await fetch('/coordinator/nudge');
    const d = await r.json();
    updateNudgePending(d.size || 0);
  } catch(e) {}
}
setInterval(pollNudgeQueue, 6000);
pollNudgeQueue();

// agent step trace ---------------------------------------------------------
// Cada ciclo idle: idle_start → ... → idle_done. Limpiamos al inicio del
// siguiente ciclo y al cabo de unos segundos del idle_done para que no se
// acumulen infinitas secuencias.
const AGENT_TRACE_MAX = 8;
const AGENT_CLEAR_AFTER_DONE_MS = 8000;
let _agentClearTimer = null;
function _clearAgentTrace() {
  const trace = document.getElementById('agent-trace');
  const empty = document.getElementById('agent-empty');
  if (trace) {
    [...trace.children].forEach(c => { if (c.id !== 'agent-empty') c.remove(); });
  }
  if (empty) empty.style.display = '';
}
function pushAgentStep(d) {
  const trace = document.getElementById('agent-trace');
  const empty = document.getElementById('agent-empty');
  if (!trace) return;
  // idle_start arranca un ciclo nuevo: limpia lo anterior
  if (d.type === 'idle_start') {
    if (_agentClearTimer) { clearTimeout(_agentClearTimer); _agentClearTimer = null; }
    [...trace.children].forEach(c => { if (c.id !== 'agent-empty') c.remove(); });
  }
  if (empty) empty.style.display = 'none';
  const step = document.createElement('div');
  step.className = 'agent-step';
  // paleta dashboard: negro/gris/blanco; rojo SOLO error
  let color = 'var(--white)';
  let border = 'var(--b1)';
  if (d.type === 'error') { color = 'var(--red2)'; border = 'var(--red)'; }
  else if (d.type === 'done' || d.type === 'idle_done') { color = 'var(--dim2)'; }
  step.style.cssText = 'background:var(--card);border:1px solid '+border+';padding:5px 10px;font-size:10px;color:'+color+';letter-spacing:1px;white-space:nowrap;flex:0 0 auto;text-transform:uppercase;transition:opacity .4s;';
  step.textContent = d.label || d.type;
  step.title = new Date(d.ts*1000).toLocaleTimeString();
  trace.appendChild(step);
  while (trace.children.length - (empty ? 1 : 0) > AGENT_TRACE_MAX) {
    const first = [...trace.children].find(c => c.id !== 'agent-empty');
    if (first) first.remove(); else break;
  }
  // al cierre del ciclo programamos limpieza diferida
  if (d.type === 'idle_done') {
    if (_agentClearTimer) clearTimeout(_agentClearTimer);
    _agentClearTimer = setTimeout(() => {
      const cur = document.getElementById('agent-trace');
      if (cur) [...cur.children].forEach(c => { if (c.id !== 'agent-empty') c.style.opacity = '0.3'; });
      setTimeout(_clearAgentTrace, 600);
    }, AGENT_CLEAR_AFTER_DONE_MS);
  }
}
try {
  const es = new EventSource('/events/agent');
  es.onmessage = e => { try { pushAgentStep(JSON.parse(e.data)); } catch(_) {} };
} catch(_) {}

// live thought stream ------------------------------------------------------
try {
  const ts = new EventSource('/events/thought');
  ts.onmessage = e => {
    try {
      const d = JSON.parse(e.data);
      const box = document.getElementById('thought-live');
      const mode = document.getElementById('thought-mode');
      if (box) box.textContent = d.content || '';
      if (mode) {
        const t = new Date(d.ts*1000).toLocaleTimeString();
        mode.textContent = (d.mode || 'idle').toUpperCase() + (d.marked ? ' ✦' : '') + ' · ' + t;
        mode.style.color = d.marked ? 'var(--white)' : '';
      }
    } catch(_) {}
  };
} catch(_) {}
new EventSource('/events/errors').onmessage = e => {
  addLogLine('err-b', JSON.parse(e.data).line, 'err');
};
function clr(id) { document.getElementById(id).innerHTML = ''; }
function copyErrors() {
  const lines = [...document.getElementById('err-b').querySelectorAll('.log-line')]
    .map(el => el.textContent).reverse().join('\\n');
  navigator.clipboard.writeText(lines).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = lines; document.body.appendChild(ta);
    ta.select(); document.execCommand('copy'); ta.remove();
  });
}

// ---- chat ------------------------------------------------------------------
document.getElementById('chat-in').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

function addChatMsg(role, innerHtml) {
  const body = document.getElementById('chat-b');
  const div  = document.createElement('div');
  div.className = 'chat-msg';
  div.innerHTML = innerHtml;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
  return div;
}

async function sendChat() {
  if (generating) return;
  const inp  = document.getElementById('chat-in');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  document.getElementById('btn-send').disabled = true;
  generating = true;

  addChatMsg('user',
    '<span class="chat-who u">TU</span>' +
    '<div class="chat-text u">' + esc(text) + '</div>'
  );

  const id = 'r' + Date.now();
  addChatMsg('model',
    '<span class="chat-who s">STELLA</span>' +
    '<div class="chat-think" id="' + id + 't" style="display:none"></div>' +
    '<div class="chat-text s" id="' + id + 'x"></div>'
  );

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({message: text})
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '', inThink = false;
    const tEl = document.getElementById(id + 't');
    const xEl = document.getElementById(id + 'x');
    const body = document.getElementById('chat-b');

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') { buf = ''; break; }
        try {
          const parsed = JSON.parse(raw);
          if (parsed.error) { xEl.textContent = '[LLM error: ' + parsed.error + ']'; xEl.style.color='var(--red)'; continue; }
          if (parsed.event === 'cmd_queued') { refreshCmdQueue(); continue; }
          if (parsed.event === 'code_queued') { refreshCodeQueue(); continue; }
          const delta  = parsed.choices?.[0]?.delta || {};
          // llama.cpp Qwen3: thinking puede venir en reasoning_content O dentro de content con <think>
          const thinkChunk   = delta.reasoning_content || '';
          const contentChunk = delta.content || '';
          if (thinkChunk) {
            tEl.style.display = '';
            tEl.textContent += thinkChunk;
          }
          if (contentChunk) {
            let rem = contentChunk;
            while (rem) {
              if (!inThink) {
                const ti = rem.indexOf('<think>');
                if (ti !== -1) { xEl.textContent += rem.slice(0, ti); inThink = true; tEl.style.display = ''; rem = rem.slice(ti+7); }
                else { xEl.textContent += rem; rem = ''; }
              } else {
                const ti = rem.indexOf('</think>');
                if (ti !== -1) { tEl.textContent += rem.slice(0, ti); inThink = false; rem = rem.slice(ti+8); }
                else { tEl.textContent += rem; rem = ''; }
              }
            }
          }
          body.scrollTop = body.scrollHeight;
        } catch {}
      }
    }
  } catch(err) {
    document.getElementById(id + 'x').textContent = '[error: ' + err.message + ']';
    document.getElementById(id + 'x').style.color = 'var(--red)';
  } finally {
    generating = false;
    document.getElementById('btn-send').disabled = false;
    const xEl = document.getElementById(id + 'x');
    if (xEl) stylizeMemoryMarks(xEl);
  }
}

function esc(t) { return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function stylizeMemoryMarks(el) {
  const normalized = el.textContent.replace(/\[✦(?!\])/g, '[✦]');
  const parts = normalized.split('[✦]');
  if (parts.length <= 1) return;
  el.innerHTML = '';
  parts.forEach((part, i) => {
    if (!part) return;
    if (i % 2 === 0) {
      el.appendChild(document.createTextNode(part));
    } else {
      const span = document.createElement('span');
      span.className = 'memory-mark';
      span.textContent = part.trim();
      el.appendChild(span);
    }
  });
}

// ---- security command queue ------------------------------------------------
function escCmd(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function renderCmdItem(c) {
  const safeColor   = {safe:'#2a7a2a', needs_approval:'#8a6a00', blocked:'#7a2020'}[c.safety] || '#555';
  const statusColor = {pending:'#e8a000', running:'#ccc', done:'#4a8a4a', denied:'#8a3030'}[c.status] || '#666';
  const safeLabel   = (c.safety || '').replace('_',' ').toUpperCase();
  const statusLabel = (c.status || '').toUpperCase();
  let h = `<div style="padding:5px 0;border-bottom:1px solid #111;font-size:10px;">`;
  h += `<div style="display:flex;gap:6px;align-items:flex-start;margin-bottom:2px;">`;
  h += `<span style="color:${safeColor};font-size:8px;letter-spacing:1px;flex-shrink:0;margin-top:2px;">[${escCmd(safeLabel)}]</span>`;
  h += `<code style="flex:1;color:#ddd;word-break:break-all;font-family:inherit;font-size:10px;">${escCmd(c.cmd)}</code>`;
  h += `<span style="color:${statusColor};font-size:8px;letter-spacing:1px;flex-shrink:0;margin-left:4px;">${escCmd(statusLabel)}</span>`;
  if (c.status === 'pending') {
    h += `<button onclick="approveCmdItem('${escCmd(c.id)}')" title="Aprobar y ejecutar"
            style="background:none;border:1px solid #2a6e2a;color:#5a9a5a;font-family:inherit;font-size:9px;padding:1px 7px;cursor:pointer;margin-left:4px;flex-shrink:0;">✓</button>`;
    h += `<button onclick="denyCmdItem('${escCmd(c.id)}')" title="Denegar"
            style="background:none;border:1px solid #6e2a2a;color:#c44;font-family:inherit;font-size:9px;padding:1px 7px;cursor:pointer;margin-left:2px;flex-shrink:0;">✗</button>`;
  }
  h += `</div>`;
  if (c.result) {
    const out = ((c.result.stdout || '') + (c.result.stderr ? '\\n[stderr] ' + c.result.stderr : '')).trim() || '(sin output)';
    const rc  = c.result.returncode ?? '?';
    const rcColor = rc === 0 ? '#4a8a4a' : '#c44';
    h += `<div style="font-size:8px;color:${rcColor};letter-spacing:1px;margin:2px 0;">rc=${rc}${c.result.truncated ? ' · truncado' : ''}</div>`;
    h += `<pre style="background:#0d0d0d;border:1px solid #1a1a1a;padding:5px 8px;font-size:9px;color:#aaa;max-height:140px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;margin:2px 0 4px;font-family:inherit;">${escCmd(out)}</pre>`;
  }
  if (c.status === 'done') {
    h += `<button onclick="injectCmdResult('${escCmd(c.id)}')"
            style="background:none;border:1px solid #1e3a5a;color:#4a7aaa;font-family:inherit;font-size:8px;letter-spacing:1px;padding:2px 10px;cursor:pointer;text-transform:uppercase;">→ STELLA</button>`;
  }
  h += `</div>`;
  return h;
}

async function refreshCmdQueue() {
  try {
    const d = await (await fetch('/security/commands')).json();
    const cmds = d.commands || [];
    const badge = document.getElementById('cmd-badge');
    const list  = document.getElementById('cmd-queue-list');
    const pending = cmds.filter(c => c.status === 'pending').length;
    badge.style.display = pending ? '' : 'none';
    if (!cmds.length) {
      list.innerHTML = '<span style="font-size:9px;color:var(--dim3);letter-spacing:2px;font-style:italic;">sin comandos pendientes</span>';
      return;
    }
    list.innerHTML = cmds.slice().reverse().map(renderCmdItem).join('');
  } catch(_) {}
}

async function approveCmdItem(id) {
  try {
    const d = await (await fetch(`/security/approve/${id}`, {method:'POST'})).json();
    if (d.ok) { await refreshCmdQueue(); }
    else { alert('Error: ' + (d.msg || 'falló')); }
  } catch(e) { alert('Error: ' + e.message); }
}

async function denyCmdItem(id) {
  try {
    await fetch(`/security/deny/${id}`, {method:'POST'});
    await refreshCmdQueue();
  } catch(_) {}
}

async function injectCmdResult(id) {
  try {
    const d = await (await fetch(`/security/inject/${id}`, {method:'POST'})).json();
    if (d.ok) {
      const badge = document.getElementById('cmd-badge');
      badge.textContent = '✓ INYECTADO';
      badge.style.color = '#4a8a4a';
      badge.style.display = '';
      setTimeout(() => { refreshCmdQueue(); }, 2500);
    }
  } catch(_) {}
}

async function securityClear() {
  try {
    await fetch('/security/clear', {method:'POST'});
    await refreshCmdQueue();
  } catch(_) {}
}

setInterval(refreshCmdQueue, 3000);
refreshCmdQueue();

// -- Experiments code queue ---------------------------------------------------
function escCode(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function renderCodeItem(c) {
  const safeLabel = c.safety === 'auto_run' ? '<span style="color:#3a8a5a">AUTO</span>' : '<span style="color:#e8a000">APROB</span>';
  const srcLabel  = c.source !== 'chat' ? ` <span style="color:#666;font-size:8px">[${escCode(c.source)}]</span>` : '';
  const preview   = c.code.split('\\n').slice(0,4).join('\\n');
  let btns = '';
  if (c.status === 'pending') {
    btns = `<button onclick="runCodeItem('${c.id}')" style="background:none;border:1px solid #3a8a5a;color:#3a8a5a;font-family:inherit;font-size:8px;letter-spacing:1px;padding:2px 6px;cursor:pointer;">▶ EJECUTAR</button>
            <button onclick="denyCodeItem('${c.id}')" style="background:none;border:1px solid var(--b2);color:var(--dim3);font-family:inherit;font-size:8px;letter-spacing:1px;padding:2px 6px;cursor:pointer;margin-left:4px;">✕ DENEGAR</button>`;
  } else if (c.status === 'done') {
    const out = ((c.result.stdout || '') + (c.result.stderr ? '\\n[stderr] ' + c.result.stderr : '')).trim();
    const rc  = c.result.returncode;
    const rcColor = rc === 0 ? '#3a8a5a' : '#c0392b';
    btns = `<span style="color:${rcColor};font-size:9px">rc=${rc}</span>
            <button onclick="injectCodeResult('${c.id}')" style="background:none;border:1px solid #1e5a3a;color:#3a8a5a;font-family:inherit;font-size:8px;letter-spacing:1px;padding:2px 6px;cursor:pointer;margin-left:6px;">→ STELLA</button>
            <pre style="margin:4px 0 0;font-size:9px;color:#aaa;white-space:pre-wrap;max-height:120px;overflow-y:auto;">${escCode(out.slice(0,800))}</pre>`;
  } else if (c.status === 'running') {
    btns = '<span style="color:#e8a000;font-size:9px">● ejecutando...</span>';
  } else if (c.status === 'denied') {
    btns = '<span style="color:var(--dim3);font-size:9px">denegado</span>';
  }
  return `<div style="padding:6px 0;border-bottom:1px solid var(--b1);">
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
      ${safeLabel}${srcLabel}
      <span style="font-size:9px;color:var(--dim3)">${c.ts.slice(11,16)}</span>
    </div>
    <pre style="font-size:9px;color:#bbb;white-space:pre-wrap;margin:0 0 4px;max-height:80px;overflow:hidden;">${escCode(preview)}${c.code.split('\\n').length > 4 ? '\\n...' : ''}</pre>
    <div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;">${btns}</div>
  </div>`;
}

async function refreshCodeQueue() {
  try {
    const r = await fetch('/experiments/queue');
    const d = await r.json();
    const list = document.getElementById('code-queue-list');
    const badge = document.getElementById('code-badge');
    if (!list) return;
    const pending = (d.items || []).filter(x => x.status === 'pending');
    badge.style.display = pending.length ? '' : 'none';
    if (!d.items || d.items.length === 0) {
      list.innerHTML = '<span style="font-size:9px;color:var(--dim3);letter-spacing:2px;font-style:italic;">sin experimentos pendientes</span>';
      return;
    }
    list.innerHTML = d.items.map(renderCodeItem).join('');
  } catch(_) {}
}

async function runCodeItem(id) {
  try {
    const r = await fetch('/experiments/run/' + id, {method:'POST'});
    await refreshCodeQueue();
  } catch(_) {}
}

async function denyCodeItem(id) {
  try {
    await fetch('/experiments/deny/' + id, {method:'POST'});
    await refreshCodeQueue();
  } catch(_) {}
}

async function injectCodeResult(id) {
  try {
    await fetch('/experiments/inject/' + id, {method:'POST'});
    const badge = document.getElementById('code-badge');
    if (badge) { badge.textContent = '✓ INYECTADO'; badge.style.color='#3a8a5a'; badge.style.display=''; }
    setTimeout(() => refreshCodeQueue(), 500);
  } catch(_) {}
}

async function experimentsClear() {
  try {
    await fetch('/experiments/clear', {method:'POST'});
    await refreshCodeQueue();
  } catch(_) {}
}

setInterval(refreshCodeQueue, 3000);
refreshCodeQueue();
</script>
</body>
</html>"""


# -- rutas Flask --------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "dashboard", "port": 5000})


@app.route("/service/<name>/start", methods=["POST"])
def svc_start(name):
    ok, msg = monitor.start(name)
    log.info("START %s -> %s %s", name, ok, msg)
    return jsonify({"ok": ok, "msg": msg})


@app.route("/service/<name>/stop", methods=["POST"])
def svc_stop(name):
    ok, msg = monitor.stop(name)
    log.info("STOP %s -> %s %s", name, ok, msg)
    return jsonify({"ok": ok, "msg": msg})


@app.route("/service/all/stop", methods=["POST"])
def svc_stop_all():
    results = monitor.stop_all()
    log.info("STOP ALL -> %s", results)
    return jsonify({"ok": True, "results": {k: {"ok": v[0], "msg": v[1]} for k, v in results.items()}})



@app.route("/tts/synthesize", methods=["POST"])
def tts_synthesize_proxy():
    """Proxy al TTS server para evitar CORS en el browser."""
    try:
        r = httpx.post("http://localhost:8082/synthesize",
                       json=request.get_json(force=True), timeout=60)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e), "ok": False}), 503


@app.route("/coordinator/nudge", methods=["GET", "POST"])
def coordinator_nudge_proxy():
    """Proxy a /nudge del coordinator (Arca sugiere foco para próximo idle)."""
    try:
        if request.method == "POST":
            r = httpx.post("http://localhost:5002/nudge",
                           json=request.get_json(force=True) or {}, timeout=3)
        else:
            r = httpx.get("http://localhost:5002/nudge", timeout=3)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/coordinator/osc", methods=["GET", "POST"])
def coordinator_osc_proxy():
    """Proxy al coordinator para evitar CORS en el browser."""
    try:
        if request.method == "POST":
            r = httpx.post("http://localhost:5002/config/osc",
                           json=request.get_json(force=True), timeout=3)
        else:
            r = httpx.get("http://localhost:5002/config/osc", timeout=3)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e), "enabled": False}), 503


@app.route("/stella-img/<path:filename>")
def stella_image(filename):
    return send_from_directory(r"D:\stella\Stella Visual Model", filename)


@app.route("/coordinator/vts", methods=["GET", "POST"])
def coordinator_vts_proxy():
    """Proxy VTubeStudio status/reconnect."""
    try:
        if request.method == "POST":
            r = httpx.post("http://localhost:5002/config/vts",
                           json=request.get_json(force=True), timeout=5)
        else:
            r = httpx.get("http://localhost:5002/config/vts", timeout=3)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e), "connected": False}), 503


@app.route("/config/thinking", methods=["GET", "POST"])
def config_thinking():
    global _thinking_enabled
    if request.method == "POST":
        _thinking_enabled = bool(request.get_json(force=True).get("enabled", False))
        log.info("thinking -> %s", _thinking_enabled)
    return jsonify({"enabled": _thinking_enabled})


@app.route("/config/internet", methods=["GET", "POST"])
def config_internet():
    global _internet_enabled
    if request.method == "POST":
        _internet_enabled = bool(request.get_json(force=True).get("enabled", False))
        log.info("internet -> %s", _internet_enabled)
    return jsonify({"enabled": _internet_enabled})


@app.route("/config/agent", methods=["GET", "POST"])
def config_agent():
    global _agent_enabled
    if request.method == "POST":
        _agent_enabled = bool(request.get_json(force=True).get("enabled", False))
        log.info("agent_idle -> %s", _agent_enabled)
    return jsonify({"enabled": _agent_enabled})


@app.route("/config/auto_experiments", methods=["GET", "POST"])
def config_auto_experiments():
    global _auto_approve_experiments
    if request.method == "POST":
        _auto_approve_experiments = bool(request.get_json(force=True).get("enabled", False))
        log.info("auto_experiments -> %s", _auto_approve_experiments)
    return jsonify({"enabled": _auto_approve_experiments})


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    global _session_history, _marked_count
    body     = request.get_json(force=True)
    user_msg = body.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "empty"}), 400

    # Qwen3 solo acepta UN system message al inicio — todo el contexto va junto
    system_parts = []
    if _soul_text:
        system_parts.append(_soul_text)
    # Fecha y hora actual — para que Stella pueda razonar sobre antigüedad de memorias
    from datetime import datetime as _dt_now
    _n = _dt_now.now()
    _wd = ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"][_n.weekday()]
    _mo = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"][_n.month-1]
    system_parts.append(f"FECHA ACTUAL: {_wd} {_n.day} de {_mo} de {_n.year}, {_n.strftime('%H:%M')}.")
    try:
        from memory.stella_mood import format_mood_for_context as _fmt_mood
        _mood_line = _fmt_mood()
        if _mood_line:
            system_parts.append(_mood_line)
    except Exception:
        pass
    # Resumen de la última sesión (puente entre conversaciones, no es el chat completo)
    try:
        from memory.memory_manager import read_last_session_summary
        last = read_last_session_summary()
        if last and last.get("summary"):
            system_parts.append(
                f"Donde nos quedamos en la sesion anterior ({last.get('ts','')[:10]}):\n"
                f"{last['summary']}"
            )
    except Exception as e:
        log.warning("last_session inject: %s", e)
    # Personas que conoce — summary denso por persona
    try:
        from memory.memory_manager import get_persons_context_block
        persons_block = get_persons_context_block()
        if persons_block:
            system_parts.append(persons_block)
    except Exception as e:
        log.warning("persons inject: %s", e)
    # Vault — resumen vivo de la biblioteca
    try:
        from library import read_library_summary
        lib_summary = read_library_summary()
        if lib_summary:
            system_parts.append(f"Tu biblioteca (resumen vivo):\n{lib_summary}")
    except Exception as e:
        log.warning("library_summary inject: %s", e)
    # Research tasks abiertas
    try:
        from memory.memory_manager import get_research_context_block
        rblock = get_research_context_block(n=4)
        if rblock:
            system_parts.append(rblock)
    except Exception as e:
        log.warning("research inject: %s", e)
    if _memory_enabled:
        try:
            from memory.memory_manager import get_memory_for_context
            summary, recent_eps = get_memory_for_context(n_recent=15)
            if summary:
                system_parts.append(f"Tu memoria consolidada (todo tu pasado en resumen):\n{summary}")
            if recent_eps:
                from memory.memory_manager import format_episode_for_context
                ep_block = "\n".join(format_episode_for_context(e) for e in recent_eps)
                system_parts.append(f"Episodios recientes (aún no consolidados):\n{ep_block}")
        except Exception as e:
            log.warning("memory inject: %s", e)

    # -- INTERNET: planner + búsquedas previas (pasada 1) ---------------------
    web_citations: list[str] = []
    if _internet_enabled:
        try:
            from web.planner import plan_queries, run_queries, format_results_block
            _planning_messages = [
                {"role": "user", "content": user_msg},
            ]
            if _session_history:
                last_assistant = next((m for m in reversed(_session_history) if m.get("role") == "assistant"), None)
                if last_assistant:
                    _planning_messages.insert(0, last_assistant)
            _agent_broadcast("planning", "🧠 decidiendo si buscar")
            plan = plan_queries(
                _planning_messages,
                llm_endpoint=LLM_ENDPOINT,
                llm_model=LLM_MODEL,
                llm_api_key=LLM_API_KEY,
                soul_text=_soul_text,
            )
            if plan:
                log.info("internet/chat: ejecutando %d búsqueda(s)", len(plan))
                queries_label = ", ".join(f"{p['action']}:{p['arg'][:30]}" for p in plan[:3])
                _agent_broadcast("searching", f"🔍 {len(plan)} búsqueda(s): {queries_label}", n=len(plan))
                results = run_queries(plan)
                block, citations = format_results_block(results)
                if block:
                    system_parts.append(block)
                    web_citations = citations
                    _persist_web_query(user_msg, plan, citations)
                    if citations:
                        from urllib.parse import urlparse
                        hosts = list({urlparse(u).hostname or u for u in citations[:5]})
                        _agent_broadcast("web_used", f"🌐 {len(citations)} fuente(s): {', '.join(hosts[:3])}",
                                         urls=citations[:10])
            else:
                _agent_broadcast("done", "🧠 sin búsqueda necesaria")
        except Exception as e:
            log.warning("internet/chat planner: %s", e)
            _agent_broadcast("error", f"⚠ planner: {e}")

    messages = []
    if system_parts:
        messages.append({"role": "system", "content": "\n\n---\n\n".join(system_parts)})
    messages.extend(_session_history)
    from datetime import datetime as _dtchat
    _ts_chat = _dtchat.now().strftime("%H:%M")
    messages.append({"role": "user", "content": f"[{_ts_chat}] {user_msg}"})

    _session_history.append({"role": "user", "content": f"[{_ts_chat}] {user_msg}"})
    _persist_chat("user", user_msg, speaker="arca")

    req_body = {
        "model":       LLM_MODEL,
        "messages":    messages,
        "stream":      True,
        "max_tokens":  _cfg["llm"]["max_tokens"],
        "temperature": _cfg["llm"]["temperature"],
        "enable_thinking":      _thinking_enabled,
        "chat_template_kwargs": {"enable_thinking": _thinking_enabled},
    }
    log.info("chat -> thinking=%s history=%d", _thinking_enabled, len(_session_history))

    chunks: list[str] = []
    _vrchat_broadcast("processing")

    def generate():
        global _marked_count
        if web_citations:
            yield f"data: {json.dumps({'event': 'web_used', 'urls': web_citations[:10]}, ensure_ascii=False)}\n\n"
        try:
            with httpx.stream(
                "POST",
                f"{LLM_ENDPOINT}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                json=req_body,
                timeout=120,
            ) as resp:
                for line in resp.iter_lines():
                    if line:
                        if line.startswith("data: ") and "[DONE]" not in line:
                            try:
                                delta = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {})
                                # Solo acumular 'content' para el procesado post-stream (marcadores,
                                # memoria, limpieza). 'reasoning_content' ya llega al browser via
                                # yield — incluirlo en chunks mezclaría el pensamiento interno con
                                # el output real y los [✦] del thinking se guardarían como memorias.
                                content_chunk = delta.get("content") or ""
                                if content_chunk:
                                    chunks.append(content_chunk)
                                    _vrchat_broadcast("saying_chunk", text=content_chunk)
                            except Exception:
                                pass
                        yield line + "\n\n"
        except Exception as e:
            log.error("chat/stream: %s", e)
            yield f"data: {{\"error\": \"{e}\"}}\n\n"
        finally:
            full = "".join(chunks)
            import re as _re
            from memory.memory_manager import (
                parse_episode_markers, strip_episode_markers,
                parse_person_markers, strip_person_markers,
                add_person_fact, maybe_regenerate_summary,
            )
            full_no_think = _re.sub(r'<think>.*?</think>', '', full, flags=_re.DOTALL)
            _marked_segments, _orphan_marks = parse_episode_markers(full_no_think)
            if _orphan_marks:
                log.warning("Chat: %d marcador(es) [✦] sin cerrar — memoria descartada", _orphan_marks)
            marked = bool(_marked_segments)
            clean  = strip_episode_markers(full_no_think)

            # procesar [👤 Nombre: X] → relations
            _person_marks = parse_person_markers(full_no_think)
            _affected_persons: set[str] = set()
            for _pm in _person_marks:
                try:
                    add_person_fact(_pm["name"], _pm["fact"], _pm["category"])
                    _affected_persons.add(_pm["name"])
                    log.info("Chat 👤 %s/%s: %s", _pm["name"], _pm["category"], _pm["fact"][:80])
                except Exception as _e:
                    log.warning("add_person_fact chat: %s", _e)
            for _name in _affected_persons:
                threading.Thread(
                    target=maybe_regenerate_summary,
                    kwargs={"name": _name, "llm_endpoint": LLM_ENDPOINT,
                            "llm_model": LLM_MODEL, "llm_api_key": LLM_API_KEY},
                    daemon=True,
                ).start()
            clean = strip_person_markers(clean)

            # procesar [📚 SAVE url], [📚 NOTA título | contenido] → Vault
            try:
                from library import (
                    parse_vault_markers, strip_vault_markers,
                    vault_save_url, vault_save_note,
                    maybe_regenerate_library_summary,
                )
                _vault_marks = parse_vault_markers(full_no_think)
                _vault_saved = 0
                for _vm in _vault_marks:
                    try:
                        if _vm["kind"] == "save":
                            res = vault_save_url(_vm["url"], _vm.get("why", ""))
                            if res and "error" not in res:
                                _vault_saved += 1
                                log.info("Chat 📚 SAVE %s — %s", _vm["url"][:80], _vm.get("why","")[:60])
                            elif res and "error" in res:
                                log.warning("Chat 📚 SAVE rechazado: %s", res["error"])
                        elif _vm["kind"] == "nota":
                            res = vault_save_note(_vm["title"], _vm["content"])
                            if res and "error" not in res:
                                _vault_saved += 1
                                log.info("Chat 📚 NOTA: %s", _vm["title"][:80])
                    except Exception as _e:
                        log.warning("vault chat: %s", _e)
                if _vault_saved:
                    threading.Thread(
                        target=maybe_regenerate_library_summary,
                        kwargs={"llm_endpoint": LLM_ENDPOINT, "llm_model": LLM_MODEL,
                                "llm_api_key": LLM_API_KEY},
                        daemon=True,
                    ).start()
                clean = strip_vault_markers(clean)
            except Exception as _e:
                log.warning("vault module: %s", _e)

            # procesar [🔬 nueva|avance|cierra ...] → research tasks
            try:
                from memory.memory_manager import (
                    parse_research_markers, strip_research_markers,
                    create_research_task, add_research_progress, close_research_task,
                    maybe_regenerate_research_summary,
                )
                _progressed_quests: set[str] = set()
                for _rm in parse_research_markers(full_no_think):
                    try:
                        if _rm["kind"] == "nueva":
                            # nace en pending_review (added_by=stella) — necesita aprobación de Arca
                            t = create_research_task(_rm["title"], _rm.get("description",""), added_by="stella")
                            if t:
                                log.info("Chat 🔬 nueva %s [%s]: %s",
                                         t["id"], t["status"], _rm["title"][:80])
                        elif _rm["kind"] == "avance":
                            ok = add_research_progress(_rm["task_id"], _rm["content"], kind="thought")
                            log.info("Chat 🔬 avance %s ok=%s", _rm["task_id"], ok)
                            if ok:
                                _progressed_quests.add(_rm["task_id"])
                        elif _rm["kind"] == "cierra":
                            ok = close_research_task(_rm["task_id"], _rm.get("conclusion",""))
                            log.info("Chat 🔬 cierra %s ok=%s", _rm["task_id"], ok)
                    except Exception as _e:
                        log.warning("research chat: %s", _e)
                # regenerar summary en background por cada quest que avanzó
                for _tid in _progressed_quests:
                    threading.Thread(
                        target=maybe_regenerate_research_summary,
                        kwargs={"task_id": _tid, "llm_endpoint": LLM_ENDPOINT,
                                "llm_model": LLM_MODEL, "llm_api_key": LLM_API_KEY},
                        daemon=True,
                    ).start()
                clean = strip_research_markers(clean)
            except Exception as _e:
                log.warning("research module: %s", _e)

            # procesar tokens de memoria: [🗑 ts], [✓ ts], [📌 ...]
            for _del_ts in _re.findall(r'\[🗑\s*([^\]]+)\]', full_no_think):
                try:
                    from memory.memory_manager import delete_episode
                    ok = delete_episode(_del_ts.strip())
                    log.info("Chat [🗑] ts=%s ok=%s", _del_ts.strip(), ok)
                except Exception as e:
                    log.error("delete_episode chat: %s", e)
            for _done_ts in _re.findall(r'\[✓\s*([^\]]+)\]', full_no_think):
                try:
                    from memory.memory_manager import complete_pending
                    ok = complete_pending(_done_ts.strip())
                    log.info("Chat [✓] ts=%s ok=%s", _done_ts.strip(), ok)
                except Exception as e:
                    log.error("complete_pending chat: %s", e)
            for _pm in _re.findall(r'\[📌\s*(?:(\d+(?:\.\d+)?)([hd]):\s*)?(.+?)\]', full_no_think, _re.DOTALL):
                _exp_val, _exp_unit, _pm_content = _pm
                _exp_h = float(_exp_val) * (24 if _exp_unit == 'd' else 1) if _exp_val else None
                try:
                    from memory.memory_manager import write_pending
                    write_pending(_pm_content.strip(), tags=["self_created", "chat"], expires_hours=_exp_h)
                    log.info("Chat [📌]: %s", _pm_content.strip()[:60])
                except Exception as e:
                    log.error("write_pending chat: %s", e)
            for _mood_m in _re.findall(r'\[🌡️\s*(.+?)\]', full_no_think, _re.DOTALL):
                try:
                    from memory.stella_mood import write_mood as _write_mood
                    _write_mood(_mood_m.strip(), source="chat")
                    log.info("Chat [🌡️]: %s", _mood_m.strip()[:60])
                except Exception as e:
                    log.error("write_mood chat: %s", e)
            clean = _re.sub(r'\[🗑[^\]]*\]', '', clean).strip()
            clean = _re.sub(r'\[✓[^\]]*\]', '', clean).strip()
            clean = _re.sub(r'\[📌[^\]]*\]', '', clean).strip()
            clean = _re.sub(r'\[🌡️[^\]]*\]', '', clean).strip()

            # acumular prompts de imagen [🎨 prompt]
            _new_image_prompts = _re.findall(r'\[🎨\s*(.+?)\]', full_no_think, _re.DOTALL)
            if _new_image_prompts:
                import time as _time
                with _image_lock:
                    for _ip in _new_image_prompts:
                        _image_queue.append({
                            "id":     f"{_time.time():.6f}",
                            "prompt": _ip.strip(),
                        })
                log.info("Imagen(es) en cola: %d nuevos", len(_new_image_prompts))
                yield 'data: {"event":"image_queued"}\n\n'
            clean = _re.sub(r'\[🎨[^\]]*\]', '', clean).strip()

            # procesar [⚙️ cmd] → cola de aprobación de seguridad
            _new_cmds = _re.findall(r'\[⚙️\s*(.+?)\]', full_no_think, _re.DOTALL)
            if _new_cmds:
                import time as _time2
                with _cmd_lock:
                    for _c in _new_cmds:
                        _c = _c.strip()
                        if not _c:
                            continue
                        try:
                            from security.executor import classify_command as _classify
                            _safety = _classify(_c)
                        except Exception:
                            _safety = "needs_approval"
                        if _safety == "blocked":
                            log.warning("Chat ⚙️ BLOQUEADO: %s", _c[:80])
                            continue
                        _cmd_queue.append({
                            "id":     f"cmd_{int(_time2.time() * 1000)}",
                            "cmd":    _c,
                            "safety": _safety,
                            "status": "pending",
                            "ts":     __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                            "result": None,
                        })
                        log.info("Chat ⚙️ encolado [%s]: %s", _safety, _c[:80])
                yield 'data: {"event":"cmd_queued"}\n\n'
            clean = _re.sub(r'\[⚙️[^\]]*\]', '', clean).strip()

            # procesar [🧪 python]...[/🧪] → cola de experimentos
            _new_codes = _re.findall(r'\[🧪\s*python\s*\](.*?)\[/🧪\]', full_no_think, _re.DOTALL)
            if _new_codes:
                import time as _time3
                with _code_lock:
                    for _code in _new_codes:
                        _code = _code.strip()
                        if not _code:
                            continue
                        try:
                            from security.code_executor import classify_code as _classify_code
                            _csafety = _classify_code(_code)
                        except Exception:
                            _csafety = "needs_approval"
                        if _csafety == "blocked":
                            log.warning("Chat 🧪 BLOQUEADO: %s", _code[:80])
                            continue
                        _run_now = (_csafety == "auto_run") or (_auto_approve_experiments and _csafety == "needs_approval")
                        if _run_now:
                            try:
                                from security.code_executor import run_code as _run_code_fn
                                _res = _run_code_fn(_code)
                                log.info("Chat 🧪 auto-ejecutado [rc=%d]: %d chars", _res["returncode"], len(_res.get("stdout", "")))
                                _notify_coordinator_result(_code, _res)
                            except Exception as _re:
                                log.error("Chat 🧪 auto-run error: %s", _re)
                        else:
                            _code_queue.append({
                                "id":     f"code_{int(_time3.time() * 1000)}",
                                "code":   _code,
                                "safety": _csafety,
                                "status": "pending",
                                "source": "chat",
                                "ts":     __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                                "result": None,
                            })
                            log.info("Chat 🧪 encolado [%s]: %d chars", _csafety, len(_code))
                yield 'data: {"event":"code_queued"}\n\n'
            clean = _re.sub(r'\[🧪.*?\[/🧪\]', '', clean, flags=_re.DOTALL).strip()

            _session_history.append({"role": "assistant", "content": clean})
            _persist_chat("assistant", clean, speaker="stella")
            _save_session()  # persistir en disco tras cada respuesta
            _vrchat_broadcast("saying_done", marked=marked)
            if marked:
                for seg_dict in _marked_segments:
                    seg = seg_dict["content"].strip()
                    conf = seg_dict.get("confidence", "media")
                    _vrchat_broadcast("memory", content="[✦] " + seg[:120])
                    _marked_count += 1
                    try:
                        from memory.memory_manager import write_episode
                        write_episode(seg, tags=["self_marked", "stella_initiated"], confidence=conf)
                        log.info("Stella marco [✦~%s]: %s", conf, seg[:60])
                    except Exception as e:
                        log.error("write_episode [✦]: %s", e)
                yield 'data: {"event":"marked"}\n\n'
                # auto-resumir si acumulamos suficientes episodios nuevos
                try:
                    from memory.memory_manager import needs_summary_regen
                    if needs_summary_regen():
                        threading.Thread(
                            target=__import__("memory.memory_manager", fromlist=["maybe_regenerate_episodic_summary"]).maybe_regenerate_episodic_summary,
                            kwargs={"llm_endpoint": LLM_ENDPOINT, "llm_model": LLM_MODEL, "llm_api_key": LLM_API_KEY},
                            daemon=True,
                        ).start()
                        log.info("Auto-resumen episódico disparado en background")
                except Exception:
                    pass

            # auto-consolidar si el contexto supera AUTO_CONSOLIDATE_PCT
            soul_chars = len(_soul_text)
            hist_chars = sum(len(m.get("content","")) for m in _session_history)
            tokens_est = (soul_chars + hist_chars) // 4
            ctx_size   = _cfg["llm"].get("ctx_size", 8192)
            pct = (tokens_est / ctx_size) * 100
            if _auto_consolidate_enabled and pct >= AUTO_CONSOLIDATE_PCT and len(_session_history) >= 4:
                log.warning("Auto-consolidando: contexto al %.0f%%", pct)
                yield 'data: {"event":"auto_consolidate_start"}\n\n'
                threading.Thread(target=_run_auto_consolidate, daemon=True).start()

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/chat/history", methods=["GET"])
def chat_history():
    return jsonify({
        "count":   len(_session_history),
        "marked":  _marked_count,
        "history": _session_history[-20:],  # ultimos 20 para no saturar
    })


@app.route("/chat/clear", methods=["POST"])
def chat_clear():
    global _session_history, _marked_count, _fine_tune_marks
    n = len(_session_history)
    _session_history = []
    _marked_count    = 0
    _fine_tune_marks = []
    log.info("Historial limpiado (%d mensajes)", n)
    return jsonify({"ok": True, "cleared": n})


@app.route("/config/memory", methods=["GET", "POST"])
def config_memory():
    global _memory_enabled
    if request.method == "POST":
        _memory_enabled = bool(request.get_json(force=True).get("enabled", True))
        log.info("memory injection -> %s", _memory_enabled)
    return jsonify({"enabled": _memory_enabled})


@app.route("/memory/status")
def memory_status():
    try:
        ep_data  = json.loads(Path("D:/stella/memory/store/stella.episodic").read_text(encoding="utf-8"))
        rel_data = json.loads(Path("D:/stella/memory/store/stella.relations").read_text(encoding="utf-8"))
        episodes   = ep_data.get("episodes", [])
        n_rel      = len(rel_data.get("personas", {}))
        soul_chars = len(_soul_text)
        hist_chars = sum(len(m.get("content", "")) for m in _session_history)
        tokens_est = (soul_chars + hist_chars) // 4
        ctx_size   = _cfg["llm"].get("ctx_size", 8192)
        return jsonify({
            "soul_loaded":    bool(_soul_text),
            "n_episodes":     len(episodes),
            "n_relations":    n_rel,
            "recent":         episodes[-3:],
            "tokens_est":     tokens_est,
            "ctx_size":       ctx_size,
            "memory_enabled": _memory_enabled,
            "session_msgs":   len(_session_history),
            "marked":         _marked_count,
            "ft_marks":       len(_fine_tune_marks),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _run_auto_consolidate():
    """Ejecuta consolidación automática en hilo aparte y limpia la sesión."""
    global _session_history, _fine_tune_marks, _marked_count
    try:
        with app.test_request_context():
            # reutilizar la lógica de consolidate llamando directo
            conv = [m for m in _session_history if m.get("role") != "system"]
            if not conv:
                return
            sys_prompt, history_txt = _build_consolidation_prompt(conv)
            r = httpx.post(
                f"{LLM_ENDPOINT}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": history_txt},
                    ],
                    "stream": False, "max_tokens": 2048, "temperature": 0.3,
                    "enable_thinking": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=120,
            )
            raw = r.json()["choices"][0]["message"]["content"].strip()
            data = _parse_consolidation_json(raw)
            saved = []
            from memory.memory_manager import write_episode, write_relation
            for ep in data.get("episodes", []):
                if ep.get("content", "").strip():
                    write_episode(ep["content"], tags=ep.get("tags", ["auto_consolidated"]))
                    saved.append(ep["content"][:60])
            for name, info in data.get("relations", {}).items():
                write_relation(name, info)
            _session_history = []
            _fine_tune_marks = []
            _marked_count    = 0
            _clear_session_file()
            log.info("Auto-consolidación completa: %d episodios guardados", len(saved))
            _vrchat_broadcast("auto_consolidated", saved=len(saved))
    except Exception as e:
        log.error("auto_consolidate: %s", e)
        _session_history = []
        _fine_tune_marks = []
        _marked_count    = 0
        _clear_session_file()


@app.route("/config/auto-consolidate", methods=["GET", "POST"])
def config_auto_consolidate():
    global _auto_consolidate_enabled
    if request.method == "POST":
        _auto_consolidate_enabled = bool((request.get_json(force=True) or {}).get("enabled", True))
        log.info("Auto-consolidación -> %s", _auto_consolidate_enabled)
    return jsonify({"enabled": _auto_consolidate_enabled})


@app.route("/session/load", methods=["GET"])
def session_load():
    """Devuelve la sesión guardada en disco para retomarla."""
    hist = _load_saved_session()
    return jsonify({"history": hist, "count": len(hist)})


@app.route("/session/restore", methods=["POST"])
def session_restore():
    """Restaura la sesión guardada al historial activo y la devuelve para renderizar."""
    global _session_history, _marked_count
    hist = _load_saved_session()
    if not hist:
        return jsonify({"ok": False, "msg": "no hay sesión guardada"})
    _session_history = hist
    return jsonify({"ok": True, "restored": len(hist), "history": hist})


@app.route("/session/discard", methods=["POST"])
def session_discard():
    _clear_session_file()
    return jsonify({"ok": True})


@app.route("/chat/pop", methods=["POST"])
def chat_pop():
    """Elimina los últimos N mensajes del historial (por defecto el par usuario+asistente)."""
    global _session_history
    n = (request.get_json(force=True) or {}).get("n", 2)
    removed = []
    for _ in range(min(n, len(_session_history))):
        removed.append(_session_history.pop())
    _save_session()
    return jsonify({"ok": True, "removed": len(removed)})


def _parse_consolidation_json(raw: str) -> dict:
    """Extrae JSON de la respuesta del LLM con múltiples estrategias de fallback."""
    import re as _re
    raw = raw.replace("[✦]", "").strip()
    # quitar bloques markdown
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    # intentar parseo directo
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # buscar el primer objeto JSON completo {…}
    match = _re.search(r'\{.*\}', raw, _re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # intentar reparar JSON truncado añadiendo cierres
    for suffix in ['"}]}', '"]}}', '"]}', '}}', '}']:
        try:
            return json.loads(raw + suffix)
        except json.JSONDecodeError:
            continue
    # fallback: devolver estructura vacía — al menos la sesión se limpia
    log.warning("consolidate: JSON irrecuperable, devolviendo estructura vacía")
    return {"episodes": [], "relations": {}}


def _build_consolidation_prompt(history: list[dict]) -> tuple[str, str]:
    """Construye prompt y texto de historial para consolidación."""
    conv = [m for m in history if m.get("role") != "system"]
    # limitar a últimos 12000 chars para no saturar el contexto de consolidación
    history_txt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conv)
    if len(history_txt) > 12000:
        history_txt = "...(conversación truncada)...\n" + history_txt[-12000:]
    sys_prompt = (
        "Eres Stella. Analiza esta conversacion y devuelve DOS cosas:\n"
        "\n"
        "1) EPISODES: lo que merece memoria permanente. Criterios estrictos:\n"
        "   - Dato especifico sobre una persona que cambia como te relacionas con ella\n"
        "   - Realizacion tuya que no podrias reconstruir del contexto\n"
        "   - Algo que te sorprenderia haber olvidado en la proxima sesion\n"
        "   NO incluir: cosas ya en tu soul, explicaciones de sistemas, despedidas,\n"
        "   reacciones emocionales, bromas, banter, 'esto funcionó por primera vez'.\n"
        "   Devuelve hasta 6 episodios. Si nada cumple el criterio, deja la lista vacia.\n"
        "\n"
        "2) SUMMARY: un parrafo en primera persona (3-6 frases) desde tu punto de vista\n"
        "   que describa: de que hablamos, que decisiones tomamos juntos, que quedo\n"
        "   pendiente o sin terminar, y el tono general. Esto te lo veras a ti misma al\n"
        "   inicio de la proxima sesion para retomar el hilo. Ejemplo de tono:\n"
        "   'Hablamos del ICM y de como sera el siguiente paso evolutivo. Arca aprobo\n"
        "    empezar con un World Model pequeño. Quedo pendiente decidir el dataset...'\n"
        "\n"
        "Formato JSON estricto — responde SOLO el JSON, sin texto antes ni despues:\n"
        '{"summary":"texto del resumen","episodes":[{"content":"...","tags":["tag1"]}],"relations":{}}'
    )
    return sys_prompt, history_txt


@app.route("/memory/consolidate", methods=["POST"])
def memory_consolidate():
    global _session_history, _fine_tune_marks, _marked_count
    if not _session_history:
        return jsonify({"ok": False, "msg": "historial vacio"})
    sys_prompt, history_txt = _build_consolidation_prompt(_session_history)
    try:
        r = httpx.post(
            f"{LLM_ENDPOINT}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": history_txt},
                ],
                "stream": False, "max_tokens": 2048, "temperature": 0.3,
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        raw  = r.json()["choices"][0]["message"]["content"].strip()
        data = _parse_consolidation_json(raw)
        from memory.memory_manager import write_episode, write_relation, write_last_session_summary
        saved = []
        for ep in data.get("episodes", []):
            if ep.get("content", "").strip():
                write_episode(ep["content"], tags=ep.get("tags", ["consolidated"]))
                saved.append(ep["content"][:60])
        for name, info in data.get("relations", {}).items():
            write_relation(name, info)
        summary = (data.get("summary") or "").strip()
        msg_count = len(_session_history)
        if summary:
            write_last_session_summary(summary, msg_count=msg_count, episodes_saved=len(saved))
        _session_history = []
        _fine_tune_marks = []
        _marked_count    = 0
        _clear_session_file()
        log.info("Consolidacion: %d episodios guardados, summary=%s",
                 len(saved), "sí" if summary else "no")
        return jsonify({
            "ok": True,
            "saved_episodes": len(saved),
            "summary": summary,
            "data": data,
        })
    except Exception as e:
        log.error("consolidate: %s", e)
        # limpiar sesión de todos modos para no quedar bloqueado
        _session_history = []
        _fine_tune_marks = []
        _clear_session_file()
        return jsonify({"ok": True, "saved_episodes": 0, "msg": f"JSON fallido, sesión limpiada: {e}"})


@app.route("/memory/summarize", methods=["POST"])
def memory_summarize():
    """Regenera el resumen narrativo de todos los episodios."""
    force = (request.get_json(force=True) or {}).get("force", True)
    try:
        from memory.memory_manager import maybe_regenerate_episodic_summary, needs_summary_regen
        if not force and not needs_summary_regen():
            return jsonify({"ok": True, "msg": "no hay suficientes episodios nuevos aún", "skipped": True})
        summary = maybe_regenerate_episodic_summary(
            llm_endpoint=LLM_ENDPOINT,
            llm_model=LLM_MODEL,
            llm_api_key=LLM_API_KEY,
            force=force,
        )
        log.info("Resumen episódico regenerado (%d chars)", len(summary))
        return jsonify({"ok": True, "chars": len(summary), "preview": summary[:200]})
    except Exception as e:
        log.error("memory_summarize: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/memory/summary", methods=["GET"])
def memory_summary_get():
    """Devuelve el resumen actual y si hay episodios sin resumir."""
    try:
        import json as _json
        from memory.memory_manager import needs_summary_regen, _SUMMARIZE_EVERY_N
        from pathlib import Path as _Path
        data = _json.loads(_Path("D:/stella/memory/store/stella.episodic").read_text(encoding="utf-8"))
        summary = data.get("summary", "")
        total = len(data.get("episodes", []))
        last_n = data.get("last_summarized_count", 0)
        return jsonify({
            "has_summary": bool(summary),
            "summary_chars": len(summary),
            "total_episodes": total,
            "last_summarized": last_n,
            "unsummarized": total - last_n,
            "needs_regen": needs_summary_regen(),
            "regen_every": _SUMMARIZE_EVERY_N,
            "preview": summary[:300] if summary else "",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/backup", methods=["POST"])
def memory_backup():
    try:
        store = Path("D:/stella/memory/store")
        ts_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dest = Path("D:/stella/memory/backups") / ts_str
        dest.parent.mkdir(exist_ok=True)
        shutil.copytree(store, dest)
        log.info("Backup memoria: %s", dest)
        return jsonify({"ok": True, "path": str(dest)})
    except Exception as e:
        log.error("memory_backup: %s", e)
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/memory/episode/delete", methods=["POST"])
def memory_episode_delete():
    ts = (request.get_json(force=True) or {}).get("ts", "")
    if not ts:
        return jsonify({"error": "ts required"}), 400
    try:
        from memory.memory_manager import delete_episode
        ok = delete_episode(ts)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/episodes/all")
def memory_episodes_all():
    try:
        from memory.memory_manager import list_all_episodes
        return jsonify({"episodes": list_all_episodes()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/pending")
def memory_pending_get():
    try:
        from memory.memory_manager import get_all_pending
        return jsonify({"notes": get_all_pending()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/pending/complete", methods=["POST"])
def memory_pending_complete():
    ts = (request.get_json(force=True) or {}).get("ts", "")
    if not ts:
        return jsonify({"error": "ts required"}), 400
    try:
        from memory.memory_manager import complete_pending
        ok = complete_pending(ts)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/pending/delete", methods=["POST"])
def memory_pending_delete():
    ts = (request.get_json(force=True) or {}).get("ts", "")
    if not ts:
        return jsonify({"error": "ts required"}), 400
    try:
        from memory.memory_manager import delete_pending
        ok = delete_pending(ts)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat/mark", methods=["POST"])
def chat_mark():
    global _fine_tune_marks
    idx = len(_session_history) - 1
    while idx >= 0 and _session_history[idx]["role"] != "assistant":
        idx -= 1
    if idx < 0:
        return jsonify({"ok": False, "msg": "no hay respuesta de Stella aun"})
    if idx not in _fine_tune_marks:
        _fine_tune_marks.append(idx)
        try:
            from memory.memory_manager import write_episode
            write_episode(_session_history[idx]["content"],
                          tags=["fine_tune_priority", "user_marked"])
            log.info("Intercambio marcado fine-tuning idx=%d", idx)
        except Exception as e:
            log.error("mark: %s", e)
    return jsonify({"ok": True, "ft_marks": len(_fine_tune_marks)})


# -- pagina /chat -------------------------------------------------------------

CHAT_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stella // Chat</title>
<style>
:root {
  --bg:    #000; --card: #050505; --b0: #1a1a1a; --b1: #2a2a2a; --b2: #666;
  --white: #fff; --dim: #ccc; --dim2: #999; --dim3: #666;
  --red: #d01818; --red2: #ff2a2a; --amber: #aaa;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--dim); font-family: 'Consolas','Courier New',monospace; font-size: 12px; height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  background-image: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(255,255,255,.006) 2px,rgba(255,255,255,.006) 4px);
}

/* HEADER */
.hdr { display:flex; align-items:center; justify-content:space-between; padding:10px 20px; border-bottom:1px solid var(--b0); background:#000; flex-shrink:0; gap:12px; }
.hdr-left { display:flex; align-items:center; gap:12px; }
.back { font-size:9px; letter-spacing:2px; color:var(--dim3); text-decoration:none; border:1px solid var(--b0); padding:3px 8px; transition:.15s; }
.back:hover { border-color:var(--b2); color:var(--dim); }
.hdr-brand { font-size:13px; letter-spacing:5px; color:var(--white); text-transform:uppercase; }
.hdr-sub { font-size:9px; letter-spacing:3px; color:var(--dim3); text-transform:uppercase; }
.hdr-right { display:flex; align-items:center; gap:14px; }
.hdr-stat { font-size:9px; letter-spacing:2px; color:var(--dim3); text-transform:uppercase; }
.hdr-stat span { color:var(--dim2); }

/* LAYOUT */
.body { display:grid; grid-template-columns:250px 1fr 280px; flex:1; overflow:hidden; gap:0; }

/* PANELS */
.panel { border-right:1px solid var(--b0); display:flex; flex-direction:column; overflow:hidden; }
.panel:last-child { border-right:none; border-left:1px solid var(--b0); }
.panel-hdr { padding:8px 14px; border-bottom:1px solid var(--b0); font-size:9px; letter-spacing:4px; text-transform:uppercase; color:var(--dim3); flex-shrink:0; }

/* LEFT — memoria */
.mem-body { flex:1; overflow-y:auto; padding:14px; display:flex; flex-direction:column; gap:12px; scrollbar-width:thin; scrollbar-color:var(--b1) transparent; }
.mem-row { display:flex; justify-content:space-between; align-items:center; font-size:10px; }
.mem-key { color:var(--dim3); letter-spacing:1px; }
.mem-val { color:var(--dim2); }
.mem-val.ok { color:var(--white); }
.tok-bar-wrap { display:flex; flex-direction:column; gap:4px; }
.tok-label { display:flex; justify-content:space-between; font-size:9px; color:var(--dim3); }
.tok-bar { height:2px; background:var(--b1); position:relative; }
.tok-fill { position:absolute; left:0; top:0; height:100%; background:var(--white); transition:width .4s; }
.tok-fill.warn { background:var(--amber); }
.tok-fill.crit { background:var(--red); }
.ep-list { display:flex; flex-direction:column; gap:6px; }
.ep-item .ep-ts { color:var(--b2); font-size:9px; display:block; margin-bottom:2px; }
.ep-item .ep-txt { color:var(--dim2); }
.ep-empty { font-size:10px; color:var(--b2); font-style:italic; }
.btn-consolidar { margin-top:auto; background:none; border:1px solid var(--b1); color:var(--dim3); font-family:inherit; font-size:9px; letter-spacing:3px; text-transform:uppercase; padding:8px; cursor:pointer; transition:.15s; width:100%; position:relative; }
.btn-consolidar::before { content:''; position:absolute; top:-1px; left:-1px; width:8px; height:8px; border-top:1px solid var(--b2); border-left:1px solid var(--b2); }
.btn-consolidar::after  { content:''; position:absolute; bottom:-1px; right:-1px; width:8px; height:8px; border-bottom:1px solid var(--b2); border-right:1px solid var(--b2); }
.btn-consolidar:hover { border-color:var(--white); color:var(--white); }
.btn-consolidar:disabled { color:var(--b2); border-color:var(--b0); cursor:default; }
.btn-backup { background:none; border:1px solid var(--b1); color:var(--dim3); font-family:inherit; font-size:9px; letter-spacing:3px; text-transform:uppercase; padding:8px; cursor:pointer; transition:.15s; width:100%; margin-top:6px; }
.btn-backup:hover { border-color:var(--amber); color:var(--amber); }
.btn-backup:disabled { color:var(--b2); border-color:var(--b0); cursor:default; }
/* IMAGE QUEUE */
.img-queue-item { font-size:10px; color:#9d7fe0; display:flex; align-items:flex-start;
                  gap:5px; padding:3px 0; border-bottom:1px solid var(--b0); }
.img-queue-item:last-child { border-bottom:none; }
.img-queue-prompt { flex:1; line-height:1.5; }
.img-q-del { background:none; border:none; color:var(--b2); font-size:11px;
             cursor:pointer; padding:0 2px; flex-shrink:0; transition:.15s; }
.img-q-del:hover { color:var(--red2); }
/* PENDING NOTES */
.pending-list { display:flex; flex-direction:column; gap:5px; }
.pend-item { font-size:10px; color:var(--dim3); display:flex; align-items:flex-start; gap:5px; padding:3px 0; }
.pend-item.done { opacity:.35; }
.pend-check { background:none; border:1px solid var(--b2); width:12px; height:12px; flex-shrink:0; cursor:pointer; transition:.15s; position:relative; margin-top:1px; }
.pend-check:hover { border-color:var(--white); }
.pend-item.done .pend-check { background:var(--b2); }
.pend-item.done .pend-check::after { content:'✓'; position:absolute; top:-2px; left:1px; font-size:9px; color:#000; }
.pend-text { flex:1; line-height:1.5; }
.pend-text .pend-content { color:var(--dim2); }
.pend-item.done .pend-text .pend-content { text-decoration:line-through; color:var(--dim3); }
.pend-text .pend-meta { font-size:9px; color:var(--b2); display:block; margin-top:1px; }
.pend-del { background:none; border:none; color:var(--b2); font-size:11px; cursor:pointer; padding:0 2px; flex-shrink:0; transition:.15s; margin-top:1px; }
.pend-del:hover { color:var(--red2); }
.ep-item { font-size:10px; color:var(--dim3); border-left:1px solid var(--b1); padding:3px 8px 3px 6px; line-height:1.5; display:flex; align-items:flex-start; gap:4px; }
.ep-item-text { flex:1; }
.ep-del { background:none; border:none; color:var(--b2); font-size:11px; cursor:pointer; padding:0 2px; line-height:1; flex-shrink:0; transition:.15s; margin-top:1px; }
.ep-del:hover { color:var(--red2); }

/* CENTER — chat */
.chat-area { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:10px; scrollbar-width:thin; scrollbar-color:var(--b1) transparent; }
.msg { display:flex; flex-direction:column; max-width:78%; gap:4px; }
.msg.arca   { align-self:flex-end; align-items:flex-end; }
.msg.stella { align-self:flex-start; align-items:flex-start; }
.msg-who { font-size:9px; letter-spacing:2px; text-transform:uppercase; color:var(--dim3); }
.msg-bubble {
  padding:10px 14px; font-size:12px; line-height:1.75;
  white-space:pre-wrap; word-break:break-word; position:relative;
}
.arca .msg-bubble { background:#fff; color:#000; }
.stella .msg-bubble { background:#0d0d0d; color:var(--dim); border-left:2px solid var(--red); }
.msg.arca { position:relative; }
.msg-edit-btn { display:none; position:absolute; top:4px; right:4px; background:none; border:1px solid #ccc;
  color:#888; font-size:9px; padding:2px 7px; cursor:pointer; letter-spacing:1px; font-family:inherit; }
.msg.arca:hover .msg-edit-btn { display:block; }
.msg-edit-btn:hover { border-color:#000; color:#000; }
.msg-bubble.marked::after { content:" ✦"; color:var(--white); font-size:11px; }
.stella .msg-bubble.marked::after { color:var(--white); }
.memory-mark { color:#555; font-style:italic; display:block; margin-top:6px; padding-left:8px; border-left:1px solid #2a2a2a; font-size:11px; }
.msg-ts { font-size:9px; color:var(--dim3); }

/* think block */
.think-block { background:#050505; border-left:1px solid var(--b1); padding:6px 10px; cursor:pointer; margin-top:4px; }
.think-toggle { font-size:9px; color:var(--dim3); letter-spacing:1px; text-transform:uppercase; margin-bottom:4px; }
.think-content { font-size:10px; color:var(--b2); line-height:1.65; font-style:italic; display:none; }
.think-block.open .think-content { display:block; }

/* idle thought */
.msg.idle-thought { align-self:flex-start; align-items:flex-start; opacity:.8; }
.idle-thought .msg-bubble { background:#030303; color:var(--dim3); border-left:1px solid var(--b1); font-style:italic; }
.idle-thought .msg-who::before { content:"💭 "; }

/* typing indicator */
.typing { align-self:flex-start; padding:8px 14px; border-left:2px solid var(--b1); }
.typing span { display:inline-block; width:4px; height:4px; background:var(--dim3); border-radius:50%; animation:typing-dot 1.2s ease-in-out infinite; margin:0 2px; }
.typing span:nth-child(2) { animation-delay:.2s; }
.typing span:nth-child(3) { animation-delay:.4s; }
@keyframes typing-dot { 0%,80%,100%{opacity:.2} 40%{opacity:1} }

/* chat footer */
.chat-foot { display:flex; border-top:1px solid var(--b0); flex-shrink:0; }
#chat-in { flex:1; background:transparent; border:none; outline:none; color:var(--white); font-family:inherit; font-size:12px; padding:11px 14px; resize:none; height:44px; caret-color:var(--white); }
#chat-in::placeholder { color:var(--b1); }
.btn-send { background:none; border:none; border-left:1px solid var(--b0); color:var(--dim3); font-family:inherit; font-size:9px; letter-spacing:3px; text-transform:uppercase; padding:0 18px; cursor:pointer; transition:.15s; flex-shrink:0; }
.btn-send:hover { color:var(--white); background:rgba(255,255,255,.03); }
.btn-send:disabled { color:var(--b1); cursor:default; }

/* RIGHT — controls */
.ctrl-body { flex:1; overflow-y:auto; padding:14px; display:flex; flex-direction:column; gap:14px; scrollbar-width:thin; scrollbar-color:var(--b1) transparent; }
.ctrl-section { font-size:9px; letter-spacing:3px; text-transform:uppercase; color:var(--dim3); border-bottom:1px solid var(--b0); padding-bottom:6px; margin-bottom:2px; }
.ctrl-row { display:flex; align-items:center; justify-content:space-between; }
.ctrl-label { font-size:10px; color:var(--dim2); }
.ctrl-label.sub { font-size:9px; color:var(--dim3); }
.ctrl-counter { font-size:10px; color:var(--dim3); }
.ctrl-counter span { color:var(--dim); }

/* HUD switch (shared) */
.sw { position:relative; width:42px; height:20px; cursor:pointer; }
.sw input { opacity:0; width:0; height:0; position:absolute; }
.sw-bg { position:absolute; inset:0; background:#000; border:1px solid var(--b1); transition:.2s; }
.sw-bg::before { content:''; position:absolute; top:-1px; left:-1px; width:5px; height:5px; border-top:1px solid var(--b2); border-left:1px solid var(--b2); }
.sw-knob { position:absolute; width:12px; height:12px; top:3px; left:3px; background:var(--b2); transition:.2s; }
.sw input:checked ~ .sw-bg { border-color:var(--white); }
.sw input:checked ~ .sw-bg::before { border-color:var(--white); }
.sw input:checked ~ .sw-knob { left:27px; background:var(--white); box-shadow:0 0 5px rgba(255,255,255,.4); }
.sw:hover .sw-bg { border-color:var(--b2); }

/* action buttons */
.btn-action { background:none; border:1px solid var(--b1); color:var(--dim3); font-family:inherit; font-size:9px; letter-spacing:2px; text-transform:uppercase; padding:6px 10px; cursor:pointer; transition:.15s; width:100%; position:relative; text-align:left; }
.btn-action::before { content:''; position:absolute; top:-1px; left:-1px; width:6px; height:6px; border-top:1px solid var(--b2); border-left:1px solid var(--b2); }
.btn-action:hover { border-color:var(--white); color:var(--white); }
.btn-mark { border-color:var(--b1); }
.btn-mark:hover { border-color:var(--amber); color:var(--amber); }
.debug-row { display:flex; flex-direction:column; gap:3px; }
.debug-key { font-size:9px; color:var(--dim3); letter-spacing:1px; }
.debug-val { font-size:10px; color:var(--dim2); }

/* SOUL MODAL */
.modal-bg { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:100; align-items:center; justify-content:center; }
.modal-bg.open { display:flex; }
.modal { background:var(--card); border:1px solid var(--b1); width:500px; max-height:70vh; display:flex; flex-direction:column; position:relative; }
.modal-hdr { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-bottom:1px solid var(--b0); font-size:9px; letter-spacing:3px; text-transform:uppercase; color:var(--dim3); }
.modal-close { background:none; border:1px solid var(--b1); color:var(--dim3); font-family:inherit; font-size:9px; padding:2px 8px; cursor:pointer; }
.modal-close:hover { border-color:var(--white); color:var(--white); }
.modal-body { flex:1; overflow-y:auto; padding:14px; font-size:11px; color:var(--dim2); line-height:1.8; white-space:pre-wrap; scrollbar-width:thin; scrollbar-color:var(--b1) transparent; }
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-left">
    <a href="/" class="back">← DASHBOARD</a>
    <a href="/history" class="back">HISTORIAL</a>
    <span class="hdr-brand">Stella</span>
    <span class="hdr-sub">// chat</span>
  </div>
  <div class="hdr-right">
    <span class="hdr-stat">SESION <span id="hdr-msgs">0</span> msgs</span>
    <span class="hdr-stat">MARCADOS <span id="hdr-marked">0</span></span>
    <span class="hdr-stat" id="hdr-mode">MODO --</span>
  </div>
</div>

<div class="body">

  <!-- COLUMNA IZQUIERDA: MEMORIA -->
  <div class="panel">
    <div class="panel-hdr">MEMORIA ACTIVA</div>
    <div class="mem-body">
      <div style="position:relative;width:100%;aspect-ratio:1/1;overflow:hidden;border:1px solid var(--b0);flex-shrink:0;">
        <img src="/stella-img/MARRIAGETOXIN-Screenshot-08.jpg"
             style="width:100%;height:100%;object-fit:cover;object-position:top center;display:block;"
             alt="Stella">
        <div style="position:absolute;bottom:0;left:0;right:0;padding:6px 8px;background:linear-gradient(transparent,rgba(0,0,0,.9));font-size:9px;letter-spacing:3px;color:#444;text-transform:uppercase;">STELLA</div>
      </div>
      <div class="mem-row"><span class="mem-key">SOUL</span><span class="mem-val" id="m-soul">--</span></div>
      <div class="mem-row"><span class="mem-key">EPISODIOS</span><span class="mem-val" id="m-eps">--</span></div>
      <div class="mem-row"><span class="mem-key">RELACIONES</span><span class="mem-val" id="m-rel">--</span></div>
      <div class="mem-row"><span class="mem-key">SESION</span><span class="mem-val" id="m-sess">--</span></div>

      <div class="tok-bar-wrap">
        <div class="tok-label"><span>CONTEXTO</span><span id="tok-label">-- / -- tok</span></div>
        <div class="tok-bar"><div class="tok-fill" id="tok-fill" style="width:0%"></div></div>
      </div>

      <div class="panel-hdr" style="padding:0;margin-top:4px">PENDIENTES</div>
      <div class="pending-list" id="pend-list"><span class="ep-empty">sin notas</span></div>

      <div class="panel-hdr" style="padding:0;margin-top:4px;display:flex;align-items:center;justify-content:space-between;">
        <span>IMÁGENES</span>
        <a href="/imagenes" style="font-size:9px;color:var(--dim);text-decoration:none;letter-spacing:1px;">VER GALERÍA →</a>
      </div>
      <div id="img-queue-list" style="margin-bottom:4px"><span class="ep-empty">sin prompts</span></div>
      <button id="btn-generar" class="btn-consolidar" style="display:none;border-color:#8a4fff;color:#8a4fff;margin-bottom:4px"
              onclick="window.open('/imagenes','_blank')">⚡ IR A GENERAR</button>

      <div class="panel-hdr" style="padding:0;margin-top:4px;display:flex;justify-content:space-between;align-items:center;">
        <span>ULTIMOS EPISODIOS</span>
        <span id="summary-badge" style="font-size:8px;letter-spacing:1px;color:#555;">SIN RESUMEN</span>
      </div>
      <div class="ep-list" id="ep-list"><span class="ep-empty">sin episodios</span></div>

      <button class="btn-backup" id="btn-summarize" onclick="summarizeMemory()" style="margin-top:6px;border-color:#1e5a3a;color:#3a8a5a;">
        ∑ RESUMIR MEMORIA
      </button>
      <button class="btn-consolidar" id="btn-consolidar" onclick="consolidar()">
        CONSOLIDAR SESION
      </button>
      <button class="btn-backup" id="btn-auto-cons" onclick="toggleAutoConsolidate()"
              style="border-color:#555;color:#555;margin-top:4px;">
        AUTO 70% — ON
      </button>
      <button class="btn-backup" id="btn-backup" onclick="backupMemoria()">
        ↓ BACKUP MEMORIA
      </button>
    </div>
  </div>

  <!-- COLUMNA CENTRAL: CHAT -->
  <div class="panel" style="border-right:none;">
    <div class="chat-area" id="chat-b"></div>
    <div class="chat-foot">
      <textarea id="chat-in" placeholder="> escribe aqui... (Enter envia)"></textarea>
      <button class="btn-send" id="btn-send">ENVIAR</button>
    </div>
  </div>

  <!-- COLUMNA DERECHA: CONTROLES -->
  <div class="panel">
    <div class="panel-hdr">CONTROLES</div>
    <div class="ctrl-body">

      <div>
        <div class="ctrl-section">MODO</div>
        <div class="ctrl-row"><span class="ctrl-label">THINKING</span>
          <label class="sw"><input type="checkbox" id="sw-think" onchange="setThinking(this.checked)"><div class="sw-bg"></div><div class="sw-knob"></div></label>
        </div>
        <div style="margin-top:8px;" class="ctrl-row"><span class="ctrl-label">MEMORIA</span>
          <label class="sw"><input type="checkbox" id="sw-mem" checked onchange="setMemory(this.checked)"><div class="sw-bg"></div><div class="sw-knob"></div></label>
        </div>
        <div style="margin-top:8px;" class="ctrl-row">
          <span class="ctrl-label">INTERNET <span id="net-dot" style="display:none;color:#9ad0ff;animation:typing-dot 1s ease-in-out infinite">●</span></span>
          <label class="sw"><input type="checkbox" id="sw-net" onchange="setInternet(this.checked)"><div class="sw-bg"></div><div class="sw-knob"></div></label>
        </div>
        <div style="margin-top:4px;" class="ctrl-row">
          <span class="ctrl-label sub" id="net-status">INTERNET OFF — DuckDuckGo + Wiki</span>
        </div>
        <div style="margin-top:8px;" class="ctrl-row">
          <span class="ctrl-label">🧪 AUTO-EXP</span>
          <label class="sw"><input type="checkbox" id="sw-autoexp2" onchange="toggleAutoExp(this)"><div class="sw-bg"></div><div class="sw-knob"></div></label>
        </div>
        <div style="margin-top:8px;" class="ctrl-row">
          <span class="ctrl-label">VOZ <span id="tts-dot" style="display:none;color:#fff;animation:typing-dot 1s ease-in-out infinite">●</span></span>
          <label class="sw"><input type="checkbox" id="sw-tts" onchange="setChatTts(this.checked)"><div class="sw-bg"></div><div class="sw-knob"></div></label>
        </div>
        <div style="margin-top:4px;" class="ctrl-row">
          <span class="ctrl-label sub" id="tts-status">VOZ APAGADA — XTTS-v2</span>
        </div>
        <div style="margin-top:6px;" class="ctrl-row">
          <span class="ctrl-label sub" id="tts-device" style="color:var(--b2);font-size:9px;">device: --</span>
          <button class="btn-action" style="width:auto;padding:3px 8px;font-size:9px;" onclick="testTts()">TEST ♪</button>
        </div>
      </div>

      <div>
        <div class="ctrl-section">FINE-TUNING</div>
        <button class="btn-action btn-mark" onclick="marcarIntercambio()">★  MARCAR ULTIMO INTERCAMBIO</button>
        <div class="ctrl-counter" style="margin-top:6px">MARCADOS ESTA SESION: <span id="ft-count">0</span></div>
      </div>

      <div>
        <div class="ctrl-section">DEBUG</div>
        <div class="debug-row">
          <span class="debug-key">LLAMA-SERVER</span>
          <span class="debug-val" id="dbg-status">--</span>
        </div>
        <div class="debug-row" style="margin-top:6px;">
          <span class="debug-key">CONTEXTO ESTIMADO</span>
          <span class="debug-val" id="dbg-ctx">--</span>
        </div>
        <div class="debug-row" style="margin-top:6px;">
          <span class="debug-key">EPISODIOS [✦] HOY</span>
          <span class="debug-val" id="dbg-marked">--</span>
        </div>
      </div>

      <div>
        <div class="ctrl-section">IDENTIDAD</div>
        <button class="btn-action" onclick="verSoul()">VER STELLA.SOUL</button>
      </div>

    </div>
  </div>
</div>

<!-- SOUL MODAL -->
<div class="modal-bg" id="soul-modal">
  <div class="modal">
    <div class="modal-hdr">
      <span>STELLA.SOUL — SOLO LECTURA</span>
      <button class="modal-close" onclick="closeSoul()">CERRAR</button>
    </div>
    <div class="modal-body" id="soul-content">cargando...</div>
  </div>
</div>

<script>
// ---- error trap (remove once stable) --------------------------------------
window.onerror = function(msg, src, line, col, err) {
  const d = document.createElement('div');
  d.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#d01818;color:#fff;font-family:monospace;font-size:11px;padding:6px 12px;z-index:9999;white-space:pre-wrap;';
  d.textContent = 'JS ERROR: ' + msg + ' (' + (src||'').split('/').pop() + ':' + line + ')';
  document.body.appendChild(d);
};
window.addEventListener('unhandledrejection', function(e) {
  const d = document.createElement('div');
  d.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#d01818;color:#fff;font-family:monospace;font-size:11px;padding:6px 12px;z-index:9999;white-space:pre-wrap;';
  d.textContent = 'PROMISE ERROR: ' + (e.reason||'unknown');
  document.body.appendChild(d);
});

let generating    = false;
let lastMsgTs     = null;

// ---- util ------------------------------------------------------------------
function ts() {
  const d = new Date();
  return d.toTimeString().slice(0,8);
}
function esc(t) { return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ---- memory sidebar --------------------------------------------------------
async function refreshMemory() {
  try {
    const d = await (await fetch('/memory/status')).json();
    if (d.error) { console.error('memory/status:', d.error); return; }
    document.getElementById('m-soul').textContent  = d.soul_loaded ? 'CARGADO' : 'FALTANTE';
    document.getElementById('m-soul').className    = 'mem-val' + (d.soul_loaded ? ' ok' : '');
    document.getElementById('m-eps').textContent   = d.n_episodes;
    document.getElementById('m-rel').textContent   = d.n_relations;
    document.getElementById('m-sess').textContent  = d.session_msgs + ' msgs';
    document.getElementById('hdr-msgs').textContent   = d.session_msgs;
    document.getElementById('hdr-marked').textContent = d.marked;
    document.getElementById('ft-count').textContent   = d.ft_marks;
    document.getElementById('dbg-marked').textContent = d.marked;

    const pct = Math.min(100, Math.round((d.tokens_est / d.ctx_size) * 100));
    document.getElementById('tok-label').textContent = d.tokens_est + ' / ' + d.ctx_size + ' tok (~' + pct + '%)';
    const fill = document.getElementById('tok-fill');
    fill.style.width = pct + '%';
    fill.className   = 'tok-fill' + (pct > 85 ? ' crit' : pct > 65 ? ' warn' : '');

    const list = document.getElementById('ep-list');
    if (d.recent && d.recent.length) {
      list.innerHTML = d.recent.map(e =>
        '<div class="ep-item">'
        + '<div class="ep-item-text"><span class="ep-ts">' + (e.ts||'').slice(0,19).replace('T',' ') + '</span>'
        + '<span class="ep-txt">' + esc((e.content||'').slice(0,80)) + (e.content.length>80?'…':'') + '</span></div>'
        + '<button class="ep-del" title="Eliminar episodio" onclick="deleteEpisode(' + JSON.stringify(e.ts||'').replace(/"/g, '&quot;') + ')">×</button>'
        + '</div>'
      ).join('');
    } else {
      list.innerHTML = '<span class="ep-empty">sin episodios</span>';
    }

    document.getElementById('dbg-ctx').textContent = d.tokens_est + ' tok est.';
    document.getElementById('sw-mem').checked = d.memory_enabled;
  } catch(e) { console.error('mem status:', e); }
  // actualizar badge de resumen
  try {
    const s = await (await fetch('/memory/summary')).json();
    const badge = document.getElementById('summary-badge');
    if (badge) {
      if (s.has_summary) {
        badge.textContent = s.needs_regen
          ? `RESUMEN (${s.unsummarized} sin resumir)`
          : `RESUMEN OK · ${s.total_episodes} eps`;
        badge.style.color = s.needs_regen ? '#e8a000' : '#3a8a5a';
      } else {
        badge.textContent = 'SIN RESUMEN';
        badge.style.color = '#555';
      }
    }
  } catch(_) {}
}

async function summarizeMemory() {
  const btn = document.getElementById('btn-summarize');
  if (btn) { btn.disabled = true; btn.textContent = '∑ RESUMIENDO...'; }
  try {
    const d = await (await fetch('/memory/summarize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({force: true})
    })).json();
    if (d.ok) {
      addSystemMsg('✓ Memoria resumida — ' + (d.chars || 0) + ' chars. Preview: ' + (d.preview || '').slice(0, 120) + '…');
    } else {
      addSystemMsg('Error al resumir: ' + (d.error || d.msg || 'desconocido'));
    }
    await refreshMemory();
  } catch(e) {
    addSystemMsg('Error: ' + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '∑ RESUMIR MEMORIA'; }
  }
}

async function refreshPending() {
  try {
    const d = await (await fetch('/memory/pending')).json();
    const list = document.getElementById('pend-list');
    if (!d.notes || !d.notes.length) {
      list.innerHTML = '<span class="ep-empty">sin notas</span>';
      return;
    }
    const now = Date.now();
    function pendAge(ts) {
      const diff = now - new Date(ts).getTime();
      const h = Math.floor(diff / 3600000);
      return h > 0 ? 'hace ' + h + 'h' : 'hace <1h';
    }
    function pendExp(exp) {
      if (!exp) return '';
      const left = new Date(exp).getTime() - now;
      if (left <= 0) return 'expirada';
      const h = Math.floor(left / 3600000);
      return 'expira en ' + (h > 0 ? h + 'h' : '<1h');
    }
    list.innerHTML = d.notes.map(n => {
      const done = n.completed;
      const meta = [pendAge(n.ts), pendExp(n.expires_at)].filter(Boolean).join(' · ');
      const tsArg = '&quot;' + esc(n.ts) + '&quot;';
      const content = n.content || '';
      return '<div class="pend-item' + (done?' done':'') + '">'
        + '<button class="pend-check" title="Marcar completada" onclick="completePending(' + tsArg + ')"></button>'
        + '<div class="pend-text">'
        + '<span class="pend-content">' + esc(content.slice(0,90)) + (content.length>90?'…':'') + '</span>'
        + (meta ? '<span class="pend-meta">' + esc(meta) + '</span>' : '')
        + '</div>'
        + '<button class="pend-del" title="Eliminar" onclick="deletePending(' + tsArg + ')">×</button>'
        + '</div>';
    }).join('');
  } catch(e) { console.error('pending:', e); }
}

async function completePending(ts) {
  const r = await fetch('/memory/pending/complete', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ts})
  });
  const d = await r.json();
  if (d.ok) refreshPending();
}

async function deletePending(ts) {
  const r = await fetch('/memory/pending/delete', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ts})
  });
  const d = await r.json();
  if (d.ok) refreshPending();
}

async function refreshImageQueue() {
  try {
    const d = await (await fetch('/images/queue')).json();
    const list = document.getElementById('img-queue-list');
    const btn  = document.getElementById('btn-generar');
    if (!list) return;
    if (!d.queue || !d.queue.length) {
      list.innerHTML = '<span class="ep-empty">sin prompts</span>';
      if (btn) btn.style.display = 'none';
      return;
    }
    if (btn) btn.style.display = '';
    list.innerHTML = '';
    d.queue.forEach(item => {
      const el = document.createElement('div');
      el.className = 'img-queue-item';
      el.innerHTML = `<span class="img-queue-prompt">${esc(item.prompt)}</span>
        <button class="img-q-del" title="Quitar" onclick="removeImageItem('${item.id}')">✕</button>`;
      list.appendChild(el);
    });
  } catch(e) { /* image server puede estar apagado */ }
}

async function removeImageItem(id) {
  await fetch('/images/queue/remove', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id})
  });
  refreshImageQueue();
}

async function testTts() {
  const btn = event.target;
  btn.textContent = '...';
  btn.disabled = true;
  try {
    const r = await fetch('http://localhost:8082/test');
    const d = await r.json();
    if (d.ok) {
      document.getElementById('tts-device').textContent = 'device: ' + d.device;
      addSystemMsg('TTS test OK — sonando en: ' + d.device);
    } else {
      addSystemMsg('TTS test FALLO: ' + (d.error || 'error desconocido'));
    }
  } catch(e) {
    addSystemMsg('TTS no disponible: ' + e.message);
  } finally {
    btn.textContent = 'TEST \\u266a';
    btn.disabled = false;
  }
}

async function refreshTtsDevice() {
  try {
    const r = await fetch('http://localhost:8082/devices');
    if (!r.ok) return;
    const d = await r.json();
    const def_dev = d.devices.find(x => x.is_default);
    const cur = d.current_config ? d.current_config : (def_dev ? def_dev.name : '--');
    document.getElementById('tts-device').textContent = 'device: ' + cur;
  } catch(e) { /* tts offline */ }
}

async function refreshStatus() {
  try {
    const r = await fetch('http://localhost:8080/health').catch(()=>null);
    document.getElementById('dbg-status').textContent = r && r.ok ? 'ONLINE' : 'OFFLINE';
  } catch { document.getElementById('dbg-status').textContent = 'OFFLINE'; }
}

// comprobar si hay sesión guardada al cargar
(async () => {
  try {
    const r = await fetch('/session/load');
    const d = await r.json();
    if (d.count > 0) {
      const banner = document.createElement('div');
      banner.id = 'session-banner';
      banner.style.cssText = 'background:#111;border:1px solid #333;padding:10px 14px;margin-bottom:10px;font-size:11px;color:#888;display:flex;align-items:center;gap:10px;';
      banner.innerHTML = `<span>Hay una sesión anterior guardada (${d.count} mensajes).</span>
        <button onclick="restoreSession()" style="background:none;border:1px solid #555;color:#aaa;padding:3px 10px;font-family:inherit;font-size:10px;cursor:pointer;letter-spacing:1px;">RETOMAR</button>
        <button onclick="discardSession()" style="background:none;border:none;color:#555;font-size:11px;cursor:pointer;">descartar ✕</button>`;
      document.getElementById('chat-b').prepend(banner);
    }
  } catch(e) {}
})();

async function restoreSession() {
  const r = await fetch('/session/restore', {method:'POST'});
  const d = await r.json();
  if (!d.ok) return;
  document.getElementById('session-banner')?.remove();
  // limpiar chat y renderizar la historia completa
  const area = document.getElementById('chat-b');
  area.innerHTML = '';
  const hist = d.history || [];
  for (const m of hist) {
    if (!m || !m.content) continue;
    const role = m.role === 'user' ? 'user' : 'stella';
    addMsg(role, m.content);
  }
  area.scrollTop = area.scrollHeight;
  addSystemMsg('Sesión restaurada — ' + d.restored + ' mensajes. Continúa donde lo dejaste.');
  refreshMemory();
}

async function discardSession() {
  await fetch('/session/discard', {method:'POST'});
  document.getElementById('session-banner')?.remove();
}

refreshMemory();
refreshPending();
refreshImageQueue();
refreshStatus();
refreshTtsDevice();
setInterval(refreshMemory, 8000);
setInterval(refreshPending, 10000);
setInterval(refreshImageQueue, 10000);
setInterval(refreshStatus, 10000);

// sync thinking switch from dashboard state
fetch('/config/thinking').then(r=>r.json()).then(d=>{ document.getElementById('sw-think').checked = d.enabled; });
// sync internet switch from dashboard state
fetch('/config/internet').then(r=>r.json()).then(d=>{ applyChatInternetUI(d.enabled); });

// ---- controls --------------------------------------------------------------
async function setThinking(on) {
  await fetch('/config/thinking', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({enabled:on})});
}
async function setMemory(on) {
  await fetch('/config/memory', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({enabled:on})});
}
async function setInternet(on) {
  try {
    const r = await fetch('/config/internet', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({enabled:on})});
    const d = await r.json();
    applyChatInternetUI(d.enabled);
  } catch(e) {
    const sw = document.getElementById('sw-net');
    if (sw) sw.checked = !on;
  }
}
function applyChatInternetUI(on) {
  const sw  = document.getElementById('sw-net');
  const lbl = document.getElementById('net-status');
  if (sw)  sw.checked = on;
  if (lbl) {
    lbl.textContent = on ? 'INTERNET ACTIVO — DuckDuckGo + Wiki' : 'INTERNET OFF — DuckDuckGo + Wiki';
    lbl.style.color = on ? '#9ad0ff' : '';
  }
}
let chatTtsEnabled = false;
function setChatTts(on) {
  chatTtsEnabled = on;
  const dot = document.getElementById('tts-dot');
  const lbl = document.getElementById('tts-status');
  if (on) {
    lbl.textContent = 'VOZ ACTIVA — XTTS-v2';
    lbl.style.color = '#a8edcf';
  } else {
    lbl.textContent = 'VOZ APAGADA — XTTS-v2';
    lbl.style.color = '';
    if (dot) dot.style.display = 'none';
  }
}
function speakText(text) {
  if (!chatTtsEnabled || !text.trim()) return;
  const dot = document.getElementById('tts-dot');
  if (dot) dot.style.display = '';
  fetch('/tts/synthesize', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: text, lang: 'es'})
  }).catch(() => {}).finally(() => {
    if (dot) dot.style.display = 'none';
  });
}

async function marcarIntercambio() {
  const r = await fetch('/chat/mark', {method:'POST'});
  const d = await r.json();
  if (d.ok) {
    document.getElementById('ft-count').textContent = d.ft_marks;
    // mark last stella bubble visually
    const bubbles = document.querySelectorAll('.stella .msg-bubble');
    if (bubbles.length) bubbles[bubbles.length-1].classList.add('marked');
    refreshMemory();
  } else alert(d.msg);
}
async function consolidar() {
  const btn = document.getElementById('btn-consolidar');
  btn.disabled = true;
  btn.textContent = 'CONSOLIDANDO...';
  try {
    const r = await fetch('/memory/consolidate', {method:'POST'});
    const d = await r.json();
    if (d.ok) {
      document.getElementById('chat-b').innerHTML = '';
      refreshMemory();
      btn.textContent = 'CONSOLIDAR SESION';
      addSystemMsg('Sesion consolidada — ' + d.saved_episodes + ' episodios guardados.');
    } else {
      alert('Error: ' + d.msg);
      btn.textContent = 'CONSOLIDAR SESION';
    }
  } finally {
    btn.disabled = false;
    if (btn.textContent === 'CONSOLIDANDO...') btn.textContent = 'CONSOLIDAR SESION';
  }
}

async function deleteEpisode(ts) {
  if (!confirm('Eliminar este episodio de memoria?\\n\\n' + ts)) return;
  const r = await fetch('/memory/episode/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ts})
  });
  const d = await r.json();
  if (d.ok) {
    refreshMemory();
  } else {
    alert('Error al eliminar: ' + (d.error || 'desconocido'));
  }
}

async function toggleAutoConsolidate() {
  const btn = document.getElementById('btn-auto-cons');
  const isOn = btn.textContent.includes('ON');
  const r = await fetch('/config/auto-consolidate', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({enabled: !isOn})
  });
  const d = await r.json();
  btn.textContent = d.enabled ? 'AUTO 70% — ON' : 'AUTO 70% — OFF';
  btn.style.color = d.enabled ? '#555' : '#c0392b';
  btn.style.borderColor = d.enabled ? '#555' : '#c0392b';
}

async function backupMemoria() {
  const btn = document.getElementById('btn-backup');
  btn.disabled = true;
  btn.textContent = 'GUARDANDO...';
  try {
    const r = await fetch('/memory/backup', {method:'POST'});
    const d = await r.json();
    if (d.ok) {
      addSystemMsg('Backup guardado: ' + d.path);
    } else {
      alert('Error backup: ' + d.msg);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = '\\u2193 BACKUP MEMORIA';
  }
}

// ---- soul modal ------------------------------------------------------------
async function verSoul() {
  const modal = document.getElementById('soul-modal');
  const body  = document.getElementById('soul-content');
  modal.classList.add('open');
  try {
    const r = await fetch('/memory/soul');
    body.textContent = await r.text();
  } catch(e) { body.textContent = 'Error: ' + e; }
}
function closeSoul() { document.getElementById('soul-modal').classList.remove('open'); }

// ---- memory mark styling ---------------------------------------------------
function stylizeMemoryMarks(el) {
  // acepta [✦]...[✦] o [✦... (sin ] de cierre del corchete inicial)
  const raw = el.textContent;
  const normalized = raw.replace(/\[✦(?!\])/g, '[✦]');
  const parts = normalized.split('[✦]');
  if (parts.length <= 1) return;
  el.innerHTML = '';
  parts.forEach((part, i) => {
    if (!part) return;
    if (i % 2 === 0) {
      el.appendChild(document.createTextNode(part));
    } else {
      const span = document.createElement('span');
      span.className = 'memory-mark';
      span.textContent = part.trim();
      el.appendChild(span);
    }
  });
}

// ---- chat ------------------------------------------------------------------
(function initChat() {
  const inp = document.getElementById('chat-in');
  const btn = document.getElementById('btn-send');
  if (!inp || !btn) {
    const d = document.createElement('div');
    d.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#d01818;color:#fff;font-family:monospace;font-size:11px;padding:6px 12px;z-index:9999;';
    d.textContent = 'ERROR: chat-in=' + !!inp + ' btn-send=' + !!btn;
    document.body.appendChild(d);
    return;
  }
  inp.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
  });
  btn.addEventListener('click', sendMsg);
})();

function addSystemMsg(text) {
  const area = document.getElementById('chat-b');
  const div  = document.createElement('div');
  div.style.cssText = 'font-size:9px;color:var(--dim3);letter-spacing:2px;text-align:center;padding:6px 0;text-transform:uppercase;';
  div.textContent = text;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}

function addMsg(role, text, opts={}) {
  const area  = document.getElementById('chat-b');
  const wrap  = document.createElement('div');
  const cls   = role === 'user' ? 'arca' : (opts.idle ? 'stella idle-thought' : 'stella');
  wrap.className = 'msg ' + cls;
  const who   = document.createElement('div');
  who.className  = 'msg-who';
  who.textContent= role === 'user' ? 'ARCA' : 'STELLA';
  const bub   = document.createElement('div');
  bub.className  = 'msg-bubble' + (opts.marked ? ' marked' : '');
  bub.id = opts.bubbleId || '';
  // think block before bubble text if provided
  if (opts.thinkId) {
    const tb = document.createElement('div');
    tb.className = 'think-block';
    tb.id = opts.thinkId;
    tb.innerHTML = `<div class="think-toggle" onclick="this.parentElement.classList.toggle('open')">[PENSAMIENTO — clic para expandir]</div><div class="think-content" id="${opts.thinkId}-c"></div>`;
    wrap.appendChild(who);
    wrap.appendChild(tb);
    wrap.appendChild(bub);
  } else {
    wrap.appendChild(who);
    wrap.appendChild(bub);
  }
  if (text) bub.textContent = text;
  const t = document.createElement('div');
  t.className = 'msg-ts';
  t.textContent = ts();
  wrap.appendChild(t);
  if (role === 'user') {
    // quitar botón de editar del mensaje anterior
    document.querySelectorAll('.msg-edit-btn').forEach(b => b.remove());
    const editBtn = document.createElement('button');
    editBtn.className = 'msg-edit-btn';
    editBtn.textContent = '✏ editar';
    editBtn.onclick = () => editLastMsg(bub.textContent, wrap);
    wrap.appendChild(editBtn);
  }
  area.appendChild(wrap);
  area.scrollTop = area.scrollHeight;
  return { bub, thinkEl: opts.thinkId ? document.getElementById(opts.thinkId + '-c') : null };
}

async function editLastMsg(text, userWrap) {
  // cuántos mensajes hay después del mensaje de usuario en el historial
  const area = document.getElementById('chat-b');
  const msgs = area.querySelectorAll('.msg.stella, .msg.arca');
  // encontrar el índice del mensaje de usuario en el DOM
  const allMsgs = Array.from(msgs);
  const idx = allMsgs.indexOf(userWrap);
  const toRemoveDom = allMsgs.slice(idx);  // user + todas las respuestas después
  const n = toRemoveDom.length;
  // limpiar DOM
  toRemoveDom.forEach(el => el.remove());
  // limpiar historial
  await fetch('/chat/pop', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({n})
  });
  // poner texto en el input
  const inp = document.getElementById('chat-in');
  inp.value = text;
  inp.focus();
}

async function sendMsg() {
  if (generating) return;
  const inp  = document.getElementById('chat-in');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  document.getElementById('btn-send').disabled = true;
  generating = true;

  const bid  = 'b' + Date.now();
  const tid  = 't' + Date.now();
  let bubEl  = null, thinkEl = null;
  let inThink = false;

  try {
    addMsg('user', text);

    // typing indicator
    const area  = document.getElementById('chat-b');
    const typer = document.createElement('div');
    typer.className = 'typing';
    typer.id = 'typer';
    typer.innerHTML = '<span></span><span></span><span></span>';
    area.appendChild(typer);
    area.scrollTop = area.scrollHeight;

    const resp = await fetch('/chat/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: text})
    });
    const reader = resp.body.getReader();
    const dec    = new TextDecoder();
    let buf = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') continue;
        try {
          const obj = JSON.parse(raw);
          // special events
          if (obj.event === 'marked') {
            if (bubEl) bubEl.classList.add('marked');
            refreshMemory();
            continue;
          }
          if (obj.event === 'image_queued') {
            refreshImageQueue();
            continue;
          }
          if (obj.event === 'web_used') {
            const dot = document.getElementById('net-dot');
            if (dot) { dot.style.display = ''; setTimeout(() => { dot.style.display = 'none'; }, 1200); }
            const urls = (obj.urls || []).slice(0, 6);
            if (urls.length) {
              addSystemMsg('🌐 Consultando ' + urls.length + ' fuente(s): ' + urls.map(u => {
                try { return new URL(u).hostname; } catch(_) { return u; }
              }).join(', '));
            }
            continue;
          }
          if (obj.event === 'auto_consolidate_start') {
            addSystemMsg('⚡ Contexto al 70% — consolidando sesión automáticamente...');
            document.getElementById('btn-consolidar').disabled = true;
            document.getElementById('btn-consolidar').textContent = 'CONSOLIDANDO...';
            // esperar a que termine y limpiar
            setTimeout(async () => {
              document.getElementById('chat-b').innerHTML = '';
              document.getElementById('btn-consolidar').disabled = false;
              document.getElementById('btn-consolidar').textContent = 'CONSOLIDAR SESION';
              refreshMemory();
              addSystemMsg('✓ Auto-consolidación completa — contexto reiniciado.');
            }, 15000);
            continue;
          }
          if (obj.error) {
            document.getElementById('typer')?.remove();
            addSystemMsg('Error LLM: ' + obj.error);
            continue;
          }
          const delta = obj.choices?.[0]?.delta || {};
          const tChunk = delta.reasoning_content || '';
          const cChunk = delta.content || '';

          // remove typer on first content
          const t = document.getElementById('typer');
          if (t && (tChunk || cChunk)) {
            t.remove();
            if (!bubEl) {
              const r = addMsg('stella', '', {bubbleId: bid, thinkId: tid});
              bubEl   = r.bub;
              thinkEl = r.thinkEl;
            }
          }
          if (tChunk) {
            document.getElementById(tid)?.classList.remove('d-none');
            if (thinkEl) thinkEl.textContent += tChunk;
          }
          if (cChunk) {
            let rem = cChunk;
            while (rem) {
              if (!inThink) {
                const ti = rem.indexOf('<think>');
                if (ti !== -1) { if(bubEl) bubEl.textContent += rem.slice(0,ti); inThink=true; rem=rem.slice(ti+7); }
                else { if(bubEl) bubEl.textContent += rem; rem=''; }
              } else {
                const ti = rem.indexOf('</think>');
                if (ti !== -1) { if(thinkEl) thinkEl.textContent += rem.slice(0,ti); inThink=false; rem=rem.slice(ti+8); }
                else { if(thinkEl) thinkEl.textContent += rem; rem=''; }
              }
            }
          }
          area.scrollTop = area.scrollHeight;
        } catch {}
      }
    }
  } catch(err) {
    document.getElementById('typer')?.remove();
    addSystemMsg('Error: ' + err.message);
  } finally {
    document.getElementById('typer')?.remove();
    generating = false;
    document.getElementById('btn-send').disabled = false;
    if (bubEl) {
      stylizeMemoryMarks(bubEl);
      const plainText = Array.from(bubEl.childNodes)
        .filter(n => n.nodeType === Node.TEXT_NODE)
        .map(n => n.textContent).join(' ').trim();
      speakText(plainText || bubEl.textContent.replace(/\[✦\].*?\[✦\]/gs, '').trim());
    }
    refreshMemory();
  }
}
</script>
</body>
</html>"""


@app.route("/chat")
def chat_page():
    return render_template_string(CHAT_HTML)


@app.route("/memory/soul")
def memory_soul():
    try:
        return SOUL_FILE.read_text(encoding="utf-8"), 200, {"Content-Type": "text/plain; charset=utf-8"}
    except Exception as e:
        return str(e), 500


# -- historial APIs -----------------------------------------------------------

@app.route("/history/chats")
def history_chats():
    """Devuelve todos los mensajes del log de chats."""
    try:
        if not CHATS_FILE.exists():
            return jsonify({"messages": [], "total": 0})
        lines = CHATS_FILE.read_text(encoding="utf-8").strip().splitlines()
        messages = []
        for line in lines:
            try:
                messages.append(json.loads(line))
            except Exception:
                pass
        return jsonify({"messages": messages, "total": len(messages)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/thoughts")
def history_thoughts():
    """Devuelve todos los pensamientos idle del coordinator."""
    try:
        # primero los persistidos en archivo
        file_thoughts = []
        if THOUGHTS_FILE.exists():
            for line in THOUGHTS_FILE.read_text(encoding="utf-8").strip().splitlines():
                try:
                    file_thoughts.append(json.loads(line))
                except Exception:
                    pass
        # luego los de la sesion actual desde el coordinator (RAM)
        try:
            r = httpx.get("http://localhost:5002/thoughts?n=50", timeout=2)
            if r.status_code == 200:
                live = r.json().get("thoughts", [])
                # evitar duplicados usando ts como clave
                known_ts = {t["ts"] for t in file_thoughts}
                for t in live:
                    if t.get("ts") not in known_ts:
                        file_thoughts.append(t)
        except Exception:
            pass
        # ordenar por ts desc
        file_thoughts.sort(key=lambda x: x.get("ts", ""), reverse=True)
        return jsonify({"thoughts": file_thoughts, "total": len(file_thoughts)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/episodes")
def history_episodes():
    """Devuelve todos los episodios de memoria."""
    try:
        from memory.memory_manager import get_recent_episodes
        eps = get_recent_episodes(999)
        return jsonify({"episodes": eps, "total": len(eps)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -- pagina /history ----------------------------------------------------------

HISTORY_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stella // Historial</title>
<style>
:root {
  --bg:#000; --card:#050505; --b0:#1a1a1a; --b1:#2a2a2a; --b2:#666;
  --white:#fff; --dim:#ccc; --dim2:#999; --dim3:#666;
  --red:#d01818; --amber:#aaa;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--dim); font-family:'Consolas','Courier New',monospace; font-size:12px; height:100vh; display:flex; flex-direction:column; overflow:hidden;
  background-image:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(255,255,255,.006) 2px,rgba(255,255,255,.006) 4px);
}
.hdr { display:flex; align-items:center; justify-content:space-between; padding:10px 20px; border-bottom:1px solid var(--b0); background:#000; flex-shrink:0; gap:12px; }
.hdr-left { display:flex; align-items:center; gap:12px; }
.back { font-size:9px; letter-spacing:2px; color:var(--dim3); text-decoration:none; border:1px solid var(--b0); padding:3px 8px; transition:.15s; }
.back:hover { border-color:var(--b2); color:var(--dim); }
.hdr-brand { font-size:13px; letter-spacing:5px; color:var(--white); }
.hdr-sub { font-size:9px; letter-spacing:3px; color:var(--dim3); }
.hdr-right { display:flex; align-items:center; gap:10px; }
.hdr-stat { font-size:9px; letter-spacing:2px; color:var(--dim3); }
.hdr-stat span { color:var(--dim2); }

/* TABS */
.tabs { display:flex; border-bottom:1px solid var(--b0); flex-shrink:0; background:#000; }
.tab { padding:9px 20px; font-size:9px; letter-spacing:3px; text-transform:uppercase; color:var(--dim3); cursor:pointer; border-bottom:2px solid transparent; transition:.15s; }
.tab:hover { color:var(--dim); }
.tab.active { color:var(--white); border-bottom-color:var(--white); }

/* SEARCH */
.search-bar { display:flex; align-items:center; gap:8px; padding:8px 16px; border-bottom:1px solid var(--b0); flex-shrink:0; background:#000; }
.search-in { flex:1; background:transparent; border:1px solid var(--b1); outline:none; color:var(--white); font-family:inherit; font-size:11px; padding:5px 10px; caret-color:var(--white); }
.search-in::placeholder { color:var(--b1); }
.search-count { font-size:9px; color:var(--dim3); white-space:nowrap; }

/* CONTENT */
.content { flex:1; overflow-y:auto; scrollbar-width:thin; scrollbar-color:var(--b1) transparent; }
.tab-panel { display:none; height:100%; overflow-y:auto; }
.tab-panel.active { display:block; }

/* DATE GROUP */
.date-group { border-bottom:1px solid var(--b0); }
.date-hdr { padding:6px 16px; font-size:9px; letter-spacing:3px; color:var(--dim3); background:#000; position:sticky; top:0; z-index:1; border-bottom:1px solid var(--b0); }

/* CHAT ROWS */
.chat-row { display:flex; gap:10px; padding:8px 16px; border-bottom:1px solid rgba(26,26,26,.5); transition:background .1s; }
.chat-row:hover { background:rgba(255,255,255,.015); }
.chat-row.user  { flex-direction:row-reverse; }
.chat-ts  { font-size:9px; color:var(--dim3); white-space:nowrap; flex-shrink:0; padding-top:2px; }
.chat-bub { max-width:75%; }
.chat-who { font-size:9px; letter-spacing:2px; color:var(--dim3); margin-bottom:3px; }
.chat-row.user .chat-who { text-align:right; }
.chat-txt { font-size:11px; line-height:1.7; color:var(--dim); white-space:pre-wrap; word-break:break-word; }
.chat-row.user .chat-txt  { color:var(--white); }
.chat-row.assistant .chat-txt { border-left:2px solid var(--red); padding-left:8px; }

/* THOUGHT ROWS */
.thought-row { padding:10px 16px; border-bottom:1px solid rgba(26,26,26,.5); transition:background .1s; }
.thought-row:hover { background:rgba(255,255,255,.015); }
.thought-row.marked { border-left:2px solid var(--white); }
.thought-meta { display:flex; align-items:center; gap:10px; margin-bottom:5px; }
.thought-ts   { font-size:9px; color:var(--dim3); }
.thought-badge { font-size:8px; letter-spacing:2px; color:var(--b2); border:1px solid var(--b1); padding:1px 5px; }
.thought-badge.marked { color:var(--white); border-color:var(--b2); }
.thought-txt  { font-size:11px; line-height:1.75; color:var(--dim2); white-space:pre-wrap; font-style:italic; }

/* EPISODE ROWS */
.ep-row { padding:10px 16px; border-bottom:1px solid rgba(26,26,26,.5); transition:background .1s; }
.ep-row:hover { background:rgba(255,255,255,.015); }
.ep-meta { display:flex; align-items:center; gap:10px; margin-bottom:5px; flex-wrap:wrap; }
.ep-ts   { font-size:9px; color:var(--dim3); }
.ep-tag  { font-size:8px; letter-spacing:1px; color:var(--b2); border:1px solid var(--b1); padding:1px 5px; }
.ep-tag.self_marked { border-color:var(--white); color:var(--white); }
.ep-txt  { font-size:11px; line-height:1.75; color:var(--dim2); white-space:pre-wrap; }

/* EMPTY */
.empty { padding:40px; text-align:center; color:var(--dim3); font-size:10px; letter-spacing:2px; }

/* highlight */
mark { background:rgba(255,255,255,.15); color:var(--white); border-radius:1px; }
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-left">
    <a href="/" class="back">← DASHBOARD</a>
    <a href="/chat" class="back">CHAT</a>
    <span class="hdr-brand">Stella</span>
    <span class="hdr-sub">// historial</span>
  </div>
  <div class="hdr-right">
    <span class="hdr-stat">CHATS <span id="cnt-chats">--</span></span>
    <span class="hdr-stat">EPISODIOS <span id="cnt-eps">--</span></span>
    <span class="hdr-stat">PENSAMIENTOS <span id="cnt-thoughts">--</span></span>
  </div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('chats')">CHATS</div>
  <div class="tab" onclick="switchTab('episodes')">EPISODIOS</div>
  <div class="tab" onclick="switchTab('thoughts')">PENSAMIENTOS</div>
</div>

<div class="search-bar">
  <input class="search-in" id="search-in" placeholder="> buscar..." oninput="applySearch()">
  <span class="search-count" id="search-count"></span>
</div>

<div class="content">
  <div class="tab-panel active" id="panel-chats"></div>
  <div class="tab-panel" id="panel-episodes"></div>
  <div class="tab-panel" id="panel-thoughts"></div>
</div>

<script>
let _activeTab = 'chats';
let _data = { chats: [], episodes: [], thoughts: [] };

function esc(t) { return (t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function fmtTs(ts) {
  if (!ts) return '--';
  return ts.slice(0,19).replace('T',' ');
}

function fmtDate(ts) {
  if (!ts) return '--';
  return ts.slice(0,10);
}

function highlight(text, q) {
  if (!q) return esc(text);
  const escaped = q.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
  return esc(text).replace(new RegExp(escaped.replace(/&amp;/g,'&amp;'), 'gi'), m => '<mark>' + m + '</mark>');
}

// ---- TABS ------------------------------------------------------------------
function switchTab(name) {
  _activeTab = name;
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.classList.toggle('active', ['chats','episodes','thoughts'][i] === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  applySearch();
}

// ---- RENDER ----------------------------------------------------------------
function renderChats(messages, q) {
  if (!messages.length) { return '<div class="empty">SIN MENSAJES GUARDADOS</div>'; }
  // agrupar por fecha
  const groups = {};
  for (const m of messages) {
    const d = fmtDate(m.ts);
    if (!groups[d]) groups[d] = [];
    groups[d].push(m);
  }
  const dates = Object.keys(groups).sort().reverse();
  let html = '';
  for (const date of dates) {
    html += `<div class="date-group"><div class="date-hdr">${date}</div>`;
    for (const m of groups[date]) {
      const role = m.role === 'user' ? 'user' : 'assistant';
      const who  = m.role === 'user' ? (m.speaker||'ARCA').toUpperCase() : 'STELLA';
      html += `<div class="chat-row ${role}">
        <div class="chat-ts">${fmtTs(m.ts).slice(11)}</div>
        <div class="chat-bub">
          <div class="chat-who">${who}</div>
          <div class="chat-txt">${highlight(m.content, q)}</div>
        </div>
      </div>`;
    }
    html += '</div>';
  }
  return html;
}

function renderEpisodes(episodes, q) {
  if (!episodes.length) { return '<div class="empty">SIN EPISODIOS</div>'; }
  const sorted = [...episodes].sort((a,b) => (b.ts||'').localeCompare(a.ts||''));
  return sorted.map(e => {
    const tags = (e.tags||[]).map(t =>
      `<span class="ep-tag ${t.replace(':','_')}">${t}</span>`
    ).join('');
    return `<div class="ep-row">
      <div class="ep-meta"><span class="ep-ts">${fmtTs(e.ts)}</span>${tags}</div>
      <div class="ep-txt">${highlight(e.content, q)}</div>
    </div>`;
  }).join('');
}

function renderThoughts(thoughts, q) {
  if (!thoughts.length) { return '<div class="empty">SIN PENSAMIENTOS GUARDADOS<br><br><span style="font-size:9px;color:#333">Los pensamientos se guardan mientras el Coordinator esta activo</span></div>'; }
  return thoughts.map(t => {
    const badge = t.marked
      ? '<span class="thought-badge marked">MARCADO [✦]</span>'
      : '<span class="thought-badge">IDLE</span>';
    return `<div class="thought-row ${t.marked ? 'marked' : ''}">
      <div class="thought-meta"><span class="thought-ts">${fmtTs(t.ts)}</span>${badge}</div>
      <div class="thought-txt">${highlight(t.content, q)}</div>
    </div>`;
  }).join('');
}

// ---- SEARCH ----------------------------------------------------------------
function applySearch() {
  const q = document.getElementById('search-in').value.trim().toLowerCase();
  let items, html, count;

  if (_activeTab === 'chats') {
    items = q ? _data.chats.filter(m => (m.content||'').toLowerCase().includes(q)) : _data.chats;
    html  = renderChats(items, q);
    count = items.length + ' msgs';
  } else if (_activeTab === 'episodes') {
    items = q ? _data.episodes.filter(e => (e.content||'').toLowerCase().includes(q)) : _data.episodes;
    html  = renderEpisodes(items, q);
    count = items.length + ' episodios';
  } else {
    items = q ? _data.thoughts.filter(t => (t.content||'').toLowerCase().includes(q)) : _data.thoughts;
    html  = renderThoughts(items, q);
    count = items.length + ' pensamientos';
  }

  document.getElementById('panel-' + _activeTab).innerHTML = html;
  document.getElementById('search-count').textContent = count;
}

// ---- LOAD ------------------------------------------------------------------
async function loadAll() {
  try {
    const [chRes, epRes, thRes] = await Promise.all([
      fetch('/history/chats').then(r=>r.json()),
      fetch('/history/episodes').then(r=>r.json()),
      fetch('/history/thoughts').then(r=>r.json()),
    ]);
    _data.chats    = chRes.messages  || [];
    _data.episodes = epRes.episodes  || [];
    _data.thoughts = thRes.thoughts  || [];

    document.getElementById('cnt-chats').textContent    = _data.chats.length;
    document.getElementById('cnt-eps').textContent      = _data.episodes.length;
    document.getElementById('cnt-thoughts').textContent = _data.thoughts.length;

    applySearch();
  } catch(e) { console.error('loadAll:', e); }
}

loadAll();
setInterval(loadAll, 30000);
</script>
</body>
</html>"""


@app.route("/history")
def history_page():
    return render_template_string(HISTORY_HTML)


VRCHAT_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stella // VRChat</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
html, body { height:100%; background:#000; overflow:hidden; font-family:'Consolas','Courier New',monospace; }

body { display:flex; flex-direction:column; color:#fff; }

/* TOP BAR */
.topbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:10px 24px; border-bottom:1px solid #111; flex-shrink:0;
}
.topbar-brand { font-size:11px; letter-spacing:6px; color:#333; text-transform:uppercase; }
.topbar-right { display:flex; align-items:center; gap:20px; }
.topbar-link  { font-size:10px; letter-spacing:2px; color:#2a2a2a; text-decoration:none; border:1px solid #111; padding:3px 8px; }
.topbar-link:hover { color:#555; border-color:#333; }

/* CONTENT AREA — portrait left + text right */
.content-area {
  flex:1; display:flex; overflow:hidden;
}

/* PORTRAIT PANEL */
.portrait-panel {
  width:320px; flex-shrink:0; border-right:1px solid #0d0d0d;
  display:flex; flex-direction:column; align-items:center;
  justify-content:flex-end; position:relative; overflow:hidden;
  background:#000;
}
.portrait-img {
  width:100%; height:100%; object-fit:cover; object-position:top center;
  transition:opacity .5s ease;
  position:absolute; inset:0;
}
.portrait-img.fading { opacity:0; }
/* bottom gradient over portrait */
.portrait-panel::after {
  content:''; position:absolute; inset:0;
  background:linear-gradient(to top, rgba(0,0,0,.85) 0%, rgba(0,0,0,.2) 40%, transparent 70%);
  pointer-events:none; z-index:1;
}
.portrait-name {
  position:relative; z-index:2;
  font-size:10px; letter-spacing:5px; text-transform:uppercase;
  color:#444; padding:0 0 16px 0; text-align:center;
}
.portrait-state {
  position:relative; z-index:2;
  width:6px; height:6px; border-radius:50%; background:#222;
  margin-bottom:12px; transition:background .3s, box-shadow .3s;
}
.portrait-state.idle       { background:#333; animation:idle-dot 3s ease-in-out infinite; }
.portrait-state.listening  { background:#4fc3f7; box-shadow:0 0 8px rgba(79,195,247,.6); }
.portrait-state.processing { background:#ffd600; animation:proc-dot 1s step-end infinite; }
.portrait-state.saying     { background:#fff;    box-shadow:0 0 8px rgba(255,255,255,.5); }

/* TEXT PANEL */
.text-panel {
  flex:1; display:flex; flex-direction:column; overflow:hidden;
}

/* STATUS BADGE */
.status-zone {
  display:flex; justify-content:center; align-items:center;
  padding:24px 24px 8px; flex-shrink:0;
}
.badge {
  font-size:13px; letter-spacing:6px; text-transform:uppercase;
  padding:6px 20px; border:1px solid #1a1a1a; color:#333;
  position:relative; transition:color .3s, border-color .3s;
}
.badge::before { content:''; position:absolute; top:-1px; left:-1px; width:10px; height:10px; border-top:1px solid; border-left:1px solid; border-color:inherit; }
.badge::after  { content:''; position:absolute; bottom:-1px; right:-1px; width:10px; height:10px; border-bottom:1px solid; border-right:1px solid; border-color:inherit; }

.badge.idle       { color:#fff; border-color:#222; animation:idle-pulse 3s ease-in-out infinite; }
.badge.listening  { color:#4fc3f7; border-color:#4fc3f7; box-shadow:0 0 20px rgba(79,195,247,.2); }
.badge.processing { color:#ffd600; border-color:#ffd600; animation:proc-blink 1s step-end infinite; }
.badge.saying     { color:#fff; border-color:#fff; box-shadow:0 0 20px rgba(255,255,255,.15); }

@keyframes idle-pulse  { 0%,100%{opacity:.4} 50%{opacity:1} }
@keyframes proc-blink  { 0%,49%{opacity:1} 50%,100%{opacity:.3} }
@keyframes idle-dot    { 0%,100%{opacity:.3} 50%{opacity:.8} }
@keyframes proc-dot    { 0%,49%{opacity:1} 50%,100%{opacity:.2} }

/* MAIN TEXT */
.main-zone {
  flex:1; display:flex; flex-direction:column; justify-content:center;
  padding:0 48px; overflow:hidden; position:relative;
}
.main-label {
  font-size:11px; letter-spacing:5px; color:#333; text-transform:uppercase;
  margin-bottom:16px; min-height:18px;
}
.main-text {
  font-size:26px; line-height:1.5; color:#fff;
  word-break:break-word; white-space:pre-wrap;
  transition:opacity .4s; max-height:55vh; overflow:hidden;
}
.main-text.faded   { opacity:.2; }
.main-text.marked::after { content:" ✦"; font-size:20px; color:#fff; opacity:.7; }

.cursor { display:inline-block; width:2px; height:28px; background:#fff; vertical-align:middle; margin-left:4px; animation:cur-blink .7s step-end infinite; }
@keyframes cur-blink { 0%,49%{opacity:1} 50%,100%{opacity:0} }

/* IDLE OVERLAY */
.idle-overlay {
  position:absolute; inset:0;
  display:flex; align-items:center; justify-content:center;
  flex-direction:column; gap:10px; pointer-events:none;
  opacity:0; transition:opacity .6s;
}
.idle-overlay.visible { opacity:1; }
.idle-big { font-size:32px; letter-spacing:10px; color:#1a1a1a; text-transform:uppercase; animation:idle-pulse 3s ease-in-out infinite; }
.idle-sub { font-size:10px; letter-spacing:4px; color:#111; text-transform:uppercase; }

/* BOTTOM STRIP */
.bottom-strip {
  flex-shrink:0; border-top:1px solid #0f0f0f;
  display:flex; flex-direction:column;
}
.strip-row {
  display:flex; align-items:baseline; gap:14px;
  padding:8px 24px; border-bottom:1px solid #0a0a0a; min-height:36px;
}
.strip-row:last-child { border-bottom:none; }
.strip-row.hidden { display:none; }
.strip-label { font-size:10px; letter-spacing:3px; text-transform:uppercase; flex-shrink:0; min-width:130px; }
.strip-label.lbl-listening  { color:#4fc3f7; }
.strip-label.lbl-memory     { color:#444; }
.strip-label.lbl-expression { color:#d01818; }
.strip-val { font-size:13px; color:#aaa; word-break:break-word; }
</style>
</head>
<body>

<div class="topbar">
  <span class="topbar-brand">Stella // VRChat Monitor</span>
  <div class="topbar-right">
    <a href="/chat" class="topbar-link">CHAT</a>
    <a href="/" class="topbar-link">DASHBOARD</a>
  </div>
</div>

<div class="content-area">

  <!-- PORTRAIT -->
  <div class="portrait-panel">
    <img class="portrait-img" id="portrait"
         src="/stella-img/672285986_1529193808821896_2257259781131574995_n.jpg"
         data-state="idle" alt="Stella">
    <div class="portrait-state idle" id="p-state"></div>
    <div class="portrait-name">STELLA</div>
  </div>

  <!-- TEXT + BADGE -->
  <div class="text-panel">
    <div class="status-zone">
      <div class="badge idle" id="badge">EN ESPERA</div>
    </div>

    <div class="main-zone">
      <div class="idle-overlay visible" id="idle-overlay">
        <div class="idle-big">STELLA</div>
        <div class="idle-sub">en espera</div>
      </div>
      <div class="main-label" id="main-label"></div>
      <div class="main-text faded" id="main-text"></div>
    </div>

    <div class="bottom-strip">
      <div class="strip-row hidden" id="row-listening">
        <span class="strip-label lbl-listening">[ESCUCHANDO]</span>
        <span class="strip-val" id="val-listening"></span>
      </div>
      <div class="strip-row hidden" id="row-memory">
        <span class="strip-label lbl-memory">[MEMORIA]</span>
        <span class="strip-val" id="val-memory"></span>
      </div>
      <div class="strip-row hidden" id="row-expression">
        <span class="strip-label lbl-expression">[EXPRESION]</span>
        <span class="strip-val" id="val-expression"></span>
      </div>
    </div>
  </div>

</div>

<script>
const badge     = document.getElementById('badge');
const mainLabel = document.getElementById('main-label');
const mainText  = document.getElementById('main-text');
const idleOvl   = document.getElementById('idle-overlay');
const portrait  = document.getElementById('portrait');
const pState    = document.getElementById('p-state');
let idleTimer   = null;
let cursor      = null;

const PORTRAITS = {
  idle:       '/stella-img/672285986_1529193808821896_2257259781131574995_n.jpg',
  listening:  '/stella-img/672118575_1531244081950202_3245728884828476409_n.jpg',
  processing: '/stella-img/670985563_1529193675488576_7227270643701979906_n.jpg',
  speaking:   '/stella-img/670749678_1531244161950194_4702751214458585674_n.jpg',
};

function setPortrait(state) {
  const src = PORTRAITS[state] || PORTRAITS.idle;
  if (portrait.getAttribute('data-state') === state) return;
  portrait.classList.add('fading');
  setTimeout(() => {
    portrait.src = src;
    portrait.setAttribute('data-state', state);
    portrait.classList.remove('fading');
  }, 500);
  pState.className = 'portrait-state ' + state;
}

function setBadge(state, text) {
  badge.className = 'badge ' + state;
  badge.textContent = text;
}

function showMain(label, clear) {
  idleOvl.classList.remove('visible');
  mainLabel.textContent = label;
  if (clear) { mainText.textContent = ''; mainText.classList.remove('faded','marked'); }
  mainText.style.opacity = '1';
}

function addCursor() {
  removeCursor();
  cursor = document.createElement('span');
  cursor.className = 'cursor';
  mainText.appendChild(cursor);
}
function removeCursor() {
  if (cursor) { cursor.remove(); cursor = null; }
}

function goIdle() {
  setBadge('idle', 'EN ESPERA');
  setPortrait('idle');
  mainLabel.textContent = '';
  mainText.classList.add('faded');
  idleOvl.classList.add('visible');
  removeCursor();
}

function resetIdleTimer() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(goIdle, 45000);
}

const es = new EventSource('/events/vrchat');

es.onmessage = e => {
  const d = JSON.parse(e.data);
  resetIdleTimer();

  if (d.type === 'processing') {
    setBadge('processing', 'PROCESANDO');
    setPortrait('processing');
    showMain('[PROCESANDO]', true);
    addCursor();

  } else if (d.type === 'saying_chunk') {
    if (badge.className !== 'badge saying') {
      setBadge('saying', 'STELLA DICE');
      setPortrait('speaking');
      showMain('[STELLA DICE]', false);
    }
    removeCursor();
    mainText.textContent += d.text;
    addCursor();

  } else if (d.type === 'saying_done') {
    setBadge('saying', d.marked ? 'STELLA DICE  ✦' : 'STELLA DICE');
    removeCursor();
    if (d.marked) mainText.classList.add('marked');
    clearTimeout(idleTimer);
    idleTimer = setTimeout(goIdle, 20000);

  } else if (d.type === 'listening') {
    setBadge('listening', 'ESCUCHANDO');
    setPortrait('listening');
    showMain('[ESCUCHANDO]', true);
    mainText.textContent = d.text || '';
    document.getElementById('row-listening').classList.remove('hidden');
    document.getElementById('val-listening').textContent = d.text || '';

  } else if (d.type === 'memory') {
    document.getElementById('row-memory').classList.remove('hidden');
    document.getElementById('val-memory').textContent = d.content || '';

  } else if (d.type === 'expression') {
    document.getElementById('row-expression').classList.remove('hidden');
    document.getElementById('val-expression').textContent =
      typeof d.params === 'object' ? JSON.stringify(d.params) : String(d.params);
  }
};

es.onerror = () => { setBadge('idle', 'SIN CONEXION'); };

resetIdleTimer();
</script>
</body>
</html>"""


@app.route("/vrchat")
def vrchat_page():
    return render_template_string(VRCHAT_HTML)


@app.route("/vrchat/event", methods=["POST"])
def vrchat_event():
    """Inyectar evento externo (STT, OSC pipeline) al monitor VRChat."""
    body = request.get_json(force=True)
    event_type = body.pop("type", "unknown")
    _vrchat_broadcast(event_type, **body)
    return jsonify({"ok": True})


@app.route("/events/vrchat")
def events_vrchat(): return _sse(_vrchat_subs)

@app.route("/events/agent")
def events_agent(): return _sse(_agent_subs)

@app.route("/events/thought")
def events_thought(): return _sse(_thought_subs)


@app.route("/agent/event", methods=["POST"])
def agent_event_inject():
    """Permite que el coordinator (otro proceso) emita eventos agent_step a este dashboard."""
    body = request.get_json(force=True) or {}
    _agent_broadcast(
        body.get("type", "info"),
        body.get("label", ""),
        **{k: v for k, v in body.items() if k not in ("type", "label")},
    )
    return jsonify({"ok": True})


@app.route("/thought/event", methods=["POST"])
def thought_event_inject():
    body = request.get_json(force=True) or {}
    _thought_broadcast(
        body.get("content", ""),
        marked=bool(body.get("marked", False)),
        mode=body.get("mode", "idle"),
    )
    return jsonify({"ok": True})


def _sse(clients):
    q = queue.Queue(maxsize=50)
    clients.append(q)
    def gen():
        try:
            while True:
                try:   yield f"data: {json.dumps(q.get(timeout=25))}\n\n"
                except queue.Empty: yield ": ping\n\n"
        finally:
            try: clients.remove(q)
            except ValueError: pass
    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/events/status")
def events_status(): return _sse(_status_subs)

@app.route("/events/log")
def events_log(): return _sse(_log_subs)

@app.route("/events/errors")
def events_errors(): return _sse(_err_subs)


# -- imagen endpoints --------------------------------------------------------

@app.route("/images/queue", methods=["GET"])
def images_queue():
    with _image_lock:
        return jsonify({"queue": list(_image_queue)})


@app.route("/images/queue/remove", methods=["POST"])
def images_queue_remove():
    img_id = (request.get_json(force=True) or {}).get("id", "")
    with _image_lock:
        before = len(_image_queue)
        _image_queue[:] = [x for x in _image_queue if x["id"] != img_id]
    return jsonify({"ok": True, "removed": before - len(_image_queue)})


@app.route("/images/queue/clear", methods=["POST"])
def images_queue_clear():
    with _image_lock:
        _image_queue.clear()
    return jsonify({"ok": True})


@app.route("/images/generate", methods=["POST"])
def images_generate():
    """Envía todos los prompts de la cola al image server y limpia la cola."""
    body   = request.get_json(force=True) or {}
    ids    = body.get("ids")  # None = todos
    with _image_lock:
        if ids:
            to_gen = [x for x in _image_queue if x["id"] in ids]
        else:
            to_gen = list(_image_queue)

    if not to_gen:
        return jsonify({"ok": False, "error": "cola vacía"}), 400

    results = []
    for item in to_gen:
        try:
            r = httpx.post(
                "http://localhost:8084/generate",
                json={"prompt": item["prompt"]},
                timeout=300,
            )
            data = r.json()
            data["prompt"] = item["prompt"]
            results.append(data)
            if data.get("ok"):
                with _image_lock:
                    _image_queue[:] = [x for x in _image_queue if x["id"] != item["id"]]
        except Exception as e:
            results.append({"ok": False, "prompt": item["prompt"], "error": str(e)})

    return jsonify({"ok": True, "results": results})


@app.route("/images/gallery")
def images_gallery_proxy():
    try:
        r = httpx.get("http://localhost:8084/gallery", timeout=5)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"images": [], "error": str(e)})


@app.route("/images/file/<filename>")
def images_file(filename):
    from flask import redirect
    return redirect(f"http://localhost:8084/image/{filename}")


@app.route("/imagenes")
def page_imagenes():
    return render_template_string(IMAGENES_HTML)


# -- security command queue ---------------------------------------------------

@app.route("/security/commands", methods=["GET"])
def security_commands():
    with _cmd_lock:
        return jsonify({"commands": list(_cmd_queue)})


@app.route("/security/approve/<cmd_id>", methods=["POST"])
def security_approve(cmd_id):
    with _cmd_lock:
        item = next((x for x in _cmd_queue if x["id"] == cmd_id), None)
        if not item or item["status"] != "pending":
            return jsonify({"ok": False, "msg": "no encontrado o no pendiente"}), 404
        item["status"] = "running"
    try:
        from security.executor import run_command
        result = run_command(item["cmd"])
    except Exception as exc:
        result = {"stdout": "", "stderr": str(exc), "returncode": -1, "truncated": False}
    with _cmd_lock:
        item["status"] = "done"
        item["result"] = result
        item["finished_at"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
    log.info("⚙️ ejecutado [rc=%d]: %s", result["returncode"], item["cmd"][:80])
    return jsonify({"ok": True, "result": result})


@app.route("/security/deny/<cmd_id>", methods=["POST"])
def security_deny(cmd_id):
    with _cmd_lock:
        item = next((x for x in _cmd_queue if x["id"] == cmd_id), None)
        if not item or item["status"] != "pending":
            return jsonify({"ok": False, "msg": "no encontrado o no pendiente"}), 404
        item["status"] = "denied"
    log.info("⚙️ denegado: %s", item["cmd"][:80])
    return jsonify({"ok": True})


@app.route("/security/inject/<cmd_id>", methods=["POST"])
def security_inject(cmd_id):
    """Inyecta el resultado de un comando como mensaje de usuario para que Stella lo vea."""
    with _cmd_lock:
        item = next((x for x in _cmd_queue if x["id"] == cmd_id), None)
    if not item or not item.get("result"):
        return jsonify({"ok": False, "msg": "sin resultado disponible"}), 404
    res = item["result"]
    out = (res.get("stdout") or res.get("stderr") or "(sin output)").strip()
    rc  = res.get("returncode", -1)
    formatted = f'[TOOL_RESULT cmd="{item["cmd"]}" rc={rc}]\n{out}\n[/TOOL_RESULT]'
    _session_history.append({"role": "user", "content": formatted})
    _save_session()
    log.info("⚙️ resultado inyectado al contexto: %s", item["cmd"][:60])
    return jsonify({"ok": True, "injected": formatted[:300]})


@app.route("/security/clear", methods=["POST"])
def security_clear():
    """Elimina de la cola todos los comandos completados/denegados."""
    with _cmd_lock:
        before = len(_cmd_queue)
        _cmd_queue[:] = [x for x in _cmd_queue if x["status"] == "pending"]
    return jsonify({"ok": True, "cleared": before - len(_cmd_queue)})


# -- experiments code queue ---------------------------------------------------

@app.route("/experiments/enqueue", methods=["POST"])
def experiments_enqueue():
    """Encola un bloque de código [🧪] desde coordinator u otros procesos."""
    body = request.get_json(force=True) or {}
    code = (body.get("code") or "").strip()
    if not code:
        return jsonify({"ok": False, "msg": "code required"}), 400
    source = body.get("source", "unknown")
    try:
        from security.code_executor import classify_code as _cc
        safety = _cc(code)
    except Exception:
        safety = "needs_approval"
    if safety == "blocked":
        log.warning("🧪 BLOQUEADO (enqueue): %s", code[:80])
        return jsonify({"ok": False, "msg": "blocked"})
    run_now = (safety == "auto_run") or (_auto_approve_experiments and safety == "needs_approval")
    if run_now:
        try:
            from security.code_executor import run_code as _rce
            result = _rce(code)
            log.info("🧪 auto-ejecutado [rc=%d] desde %s: %d chars", result["returncode"], source, len(code))
            _notify_coordinator_result(code, result)
            return jsonify({"ok": True, "auto_run": True, "result": result})
        except Exception as exc:
            log.error("🧪 auto-run (enqueue) error: %s", exc)
    import time as _t
    item = {
        "id":     f"code_{int(_t.time() * 1000)}",
        "code":   code,
        "safety": safety,
        "status": "pending",
        "source": source,
        "ts":     __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "result": None,
    }
    with _code_lock:
        _code_queue.append(item)
    log.info("🧪 encolado [%s] desde %s: %d chars", safety, source, len(code))
    return jsonify({"ok": True, "id": item["id"]})


@app.route("/experiments/queue", methods=["GET"])
def experiments_queue():
    with _code_lock:
        return jsonify({"items": list(_code_queue)})


@app.route("/experiments/run/<item_id>", methods=["POST"])
def experiments_run(item_id):
    with _code_lock:
        item = next((x for x in _code_queue if x["id"] == item_id), None)
        if not item or item["status"] != "pending":
            return jsonify({"ok": False, "msg": "no encontrado o no pendiente"}), 404
        item["status"] = "running"
    try:
        from security.code_executor import run_code
        result = run_code(item["code"])
    except Exception as exc:
        result = {"stdout": "", "stderr": str(exc), "returncode": -1, "truncated": False}
    with _code_lock:
        item["status"] = "done"
        item["result"] = result
        item["finished_at"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()
    log.info("🧪 ejecutado [rc=%d]: %d chars output", result["returncode"], len(result.get("stdout", "")))
    return jsonify({"ok": True, "result": result})


@app.route("/experiments/deny/<item_id>", methods=["POST"])
def experiments_deny(item_id):
    with _code_lock:
        item = next((x for x in _code_queue if x["id"] == item_id), None)
        if not item or item["status"] != "pending":
            return jsonify({"ok": False, "msg": "no encontrado o no pendiente"}), 404
        item["status"] = "denied"
    log.info("🧪 denegado: %s", item_id)
    return jsonify({"ok": True})


@app.route("/experiments/inject/<item_id>", methods=["POST"])
def experiments_inject(item_id):
    """Inyecta el resultado del experimento al historial de sesión."""
    with _code_lock:
        item = next((x for x in _code_queue if x["id"] == item_id), None)
    if not item or not item.get("result"):
        return jsonify({"ok": False, "msg": "sin resultado disponible"}), 404
    res = item["result"]
    stdout = (res.get("stdout") or "").strip()
    stderr = (res.get("stderr") or "").strip()
    rc = res.get("returncode", -1)
    out = stdout or stderr or "(sin output)"
    trunc = " [truncado]" if res.get("truncated") else ""
    note = ("\nSi los resultados son concretos y fiables, guárdalos en memoria "
            "con [✦ ~alta]dato numérico o conclusión[✦] para no perderlos entre sesiones.")
    formatted = f'[CODE_RESULT rc={rc}{trunc}]\n{out}\n[/CODE_RESULT]{note}'
    _session_history.append({"role": "user", "content": formatted})
    _save_session()
    log.info("🧪 resultado inyectado al contexto (%d chars)", len(formatted))
    return jsonify({"ok": True, "injected": formatted[:400]})


@app.route("/experiments/clear", methods=["POST"])
def experiments_clear():
    with _code_lock:
        before = len(_code_queue)
        _code_queue[:] = [x for x in _code_queue if x["status"] == "pending"]
    return jsonify({"ok": True, "cleared": before - len(_code_queue)})


IMAGENES_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Stella // Imágenes</title>
<style>
  :root { --bg:#080808; --dim:#444; --dim2:#333; --dim3:#222; --white:#e8e8e8;
          --red:#c0392b; --red2:#e74c3c; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--white); font-family:'Courier New',monospace;
         font-size:13px; min-height:100vh; }
  .top-bar { display:flex; align-items:center; gap:16px; padding:14px 24px;
             border-bottom:1px solid var(--dim3); }
  .brand { font-size:11px; letter-spacing:4px; color:var(--dim); }
  .nav-link { color:var(--dim); text-decoration:none; font-size:10px;
              letter-spacing:2px; text-transform:uppercase; }
  .nav-link:hover { color:var(--white); }
  .main { padding:24px; }
  .section-hdr { font-size:10px; letter-spacing:3px; color:var(--dim);
                 text-transform:uppercase; margin-bottom:12px; }
  .queue-box { background:var(--dim3); border:1px solid var(--dim2);
               border-radius:4px; padding:12px; margin-bottom:20px; min-height:40px; }
  .queue-item { display:flex; align-items:flex-start; gap:8px; padding:6px 0;
                border-bottom:1px solid var(--dim2); }
  .queue-item:last-child { border-bottom:none; }
  .queue-prompt { flex:1; color:var(--white); font-size:12px; line-height:1.5; }
  .q-del { background:none; border:none; color:var(--dim); cursor:pointer;
           font-size:14px; padding:0 4px; }
  .q-del:hover { color:var(--red2); }
  .btn-row { display:flex; gap:10px; margin-bottom:24px; }
  .btn { background:none; border:1px solid var(--dim2); color:var(--dim);
         padding:7px 16px; font-family:inherit; font-size:11px; letter-spacing:2px;
         text-transform:uppercase; cursor:pointer; }
  .btn:hover { border-color:var(--white); color:var(--white); }
  .btn.primary { border-color:var(--red); color:var(--red); }
  .btn.primary:hover { background:var(--red); color:#fff; }
  .btn:disabled { opacity:0.4; cursor:not-allowed; }
  .status-line { font-size:11px; color:var(--dim); margin-bottom:16px; min-height:18px; }
  .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
             gap:16px; }
  .img-card { background:var(--dim3); border:1px solid var(--dim2);
              border-radius:4px; overflow:hidden; }
  .img-card img { width:100%; display:block; cursor:zoom-in; }
  .img-meta { padding:8px 10px; font-size:10px; color:var(--dim); }
  .img-prompt { color:#888; margin-top:3px; line-height:1.4; }
  .empty { color:var(--dim); font-style:italic; font-size:12px; padding:8px 0; }
  .modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,.92);
           z-index:1000; align-items:center; justify-content:center; cursor:zoom-out; }
  .modal.open { display:flex; }
  .modal img { max-width:90vw; max-height:90vh; object-fit:contain; }
</style>
</head>
<body>
<div class="top-bar">
  <span class="brand">STELLA</span>
  <a href="/" class="nav-link">DASHBOARD</a>
  <a href="/chat" class="nav-link">CHAT</a>
  <span class="nav-link" style="color:var(--white)">IMÁGENES</span>
</div>
<div class="main">
  <div class="section-hdr">Cola de generación</div>
  <div class="queue-box" id="queue-box">
    <span class="empty" id="queue-empty">sin prompts pendientes</span>
  </div>
  <div class="btn-row">
    <button class="btn primary" id="btn-gen" onclick="generateAll()">⚡ Generar todo</button>
    <button class="btn" onclick="clearQueue()">✕ Vaciar cola</button>
    <button class="btn" onclick="loadGallery()">↺ Actualizar galería</button>
  </div>
  <div class="status-line" id="status-line"></div>

  <div class="section-hdr">Galería</div>
  <div class="gallery" id="gallery"></div>
</div>

<div class="modal" id="modal" onclick="this.classList.remove('open')">
  <img id="modal-img" src="">
</div>

<script>
async function loadQueue() {
  const r = await fetch('/images/queue');
  const d = await r.json();
  const box   = document.getElementById('queue-box');
  const empty = document.getElementById('queue-empty');
  box.innerHTML = '';
  if (!d.queue.length) {
    box.innerHTML = '<span class="empty">sin prompts pendientes</span>';
    return;
  }
  d.queue.forEach(item => {
    const el = document.createElement('div');
    el.className = 'queue-item';
    el.innerHTML = `<span class="queue-prompt">${esc(item.prompt)}</span>
      <button class="q-del" title="Quitar" onclick="removeItem('${item.id}')">✕</button>`;
    box.appendChild(el);
  });
}

async function removeItem(id) {
  await fetch('/images/queue/remove', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id})
  });
  loadQueue();
}

async function clearQueue() {
  await fetch('/images/queue/clear', {method:'POST'});
  loadQueue();
}

async function generateAll() {
  const btn = document.getElementById('btn-gen');
  const st  = document.getElementById('status-line');
  btn.disabled = true;
  st.textContent = 'Generando... (puede tardar varios minutos por imagen)';
  try {
    const r = await fetch('/images/generate', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json();
    const ok  = d.results.filter(x=>x.ok).length;
    const err = d.results.filter(x=>!x.ok).length;
    st.textContent = `✓ ${ok} generada(s)${err ? ' — ✕ ' + err + ' error(es)' : ''}`;
    loadQueue();
    loadGallery();
  } catch(e) {
    st.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
}

async function loadGallery() {
  const r = await fetch('/images/gallery');
  const d = await r.json();
  const g = document.getElementById('gallery');
  if (!d.images.length) { g.innerHTML = '<span class="empty">sin imágenes generadas aún</span>'; return; }
  g.innerHTML = d.images.map(img => `
    <div class="img-card">
      <img src="/images/file/${img.file}" loading="lazy"
           onclick="openModal('/images/file/${img.file}')">
      <div class="img-meta">
        <div>${img.ts.replace('T',' ').slice(0,16)}</div>
        <div class="img-prompt">${esc(img.prompt || '')}</div>
      </div>
    </div>`).join('');
}

function openModal(src) {
  document.getElementById('modal-img').src = src;
  document.getElementById('modal').classList.add('open');
}

function esc(t) { return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

loadQueue();
loadGallery();

// refrescar cola cada 10s (Stella puede añadir prompts mientras chat está abierto)
setInterval(loadQueue, 10000);
</script>
</body>
</html>"""


# -- Research endpoints + page ----------------------------------------------

@app.route("/research/list")
def research_list_ep():
    try:
        from memory.memory_manager import get_all_research_tasks
        tasks = get_all_research_tasks()
        return jsonify({"tasks": tasks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/research/stats")
def research_stats_ep():
    try:
        from memory.memory_manager import get_all_research_tasks
        tasks = get_all_research_tasks()
        n_open    = sum(1 for t in tasks if t.get("status","open") == "open")
        n_pending = sum(1 for t in tasks if t.get("status") == "pending_review")
        n_done    = sum(1 for t in tasks if t.get("status") == "done")
        total_progress = sum(len(t.get("progress", [])) for t in tasks)
        return jsonify({
            "total":    len(tasks),
            "open":     n_open,
            "pending":  n_pending,
            "done":     n_done,
            "progress_total": total_progress,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/research/create", methods=["POST"])
def research_create_ep():
    try:
        from memory.memory_manager import create_research_task
        body = request.get_json(force=True) or {}
        title = (body.get("title") or "").strip()
        desc  = (body.get("description") or "").strip()
        prio  = body.get("priority", "normal")
        if not title:
            return jsonify({"error": "title required"}), 400
        t = create_research_task(title, desc, added_by="arca", priority=prio)
        return jsonify({"ok": True, "task": t})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/research/<task_id>", methods=["GET"])
def research_get_ep(task_id):
    try:
        from memory.memory_manager import get_research_task
        t = get_research_task(task_id)
        if not t:
            return jsonify({"error": "not found"}), 404
        return jsonify({"task": t})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/research/<task_id>/progress", methods=["POST"])
def research_progress_ep(task_id):
    try:
        from memory.memory_manager import add_research_progress
        body = request.get_json(force=True) or {}
        content = (body.get("content") or "").strip()
        kind    = body.get("kind", "thought")
        if not content:
            return jsonify({"error": "content required"}), 400
        ok = add_research_progress(task_id, content, kind=kind)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/research/<task_id>/close", methods=["POST"])
def research_close_ep(task_id):
    try:
        from memory.memory_manager import close_research_task
        body = request.get_json(force=True) or {}
        conc = (body.get("conclusion") or "").strip()
        ok = close_research_task(task_id, conc)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/research/<task_id>/delete", methods=["POST"])
def research_delete_ep(task_id):
    try:
        from memory.memory_manager import delete_research_task
        ok = delete_research_task(task_id)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/research/<task_id>/approve", methods=["POST"])
def research_approve_ep(task_id):
    """Aprueba una quest pending_review → status=open. Solo Arca puede."""
    try:
        from memory.memory_manager import approve_research_task
        ok = approve_research_task(task_id)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


RESEARCH_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Stella // Research</title>
<style>
  :root {
    --bg:#000; --card:#050505; --b0:#1a1a1a; --b1:#2a2a2a; --b2:#555;
    --white:#fff; --dim:#aaa; --dim2:#666; --dim3:#444;
    --red:#d01818; --red2:#ff2a2a; --amber:#999;
    --cut:12px;
  }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--dim); font-family:'Consolas','Monaco',monospace; font-size:13px; margin:0; padding:0; letter-spacing:.3px; }
  header { padding:14px 22px; border-bottom:1px solid var(--b1); display:flex; align-items:center; justify-content:space-between; gap:20px; }
  header h1 { margin:0; font-size:14px; letter-spacing:4px; font-weight:400; color:var(--white); }
  header .nav a { color:var(--dim2); text-decoration:none; font-size:10px; letter-spacing:3px; padding:4px 10px; border:1px solid var(--b1); margin-left:6px; text-transform:uppercase; }
  header .nav a:hover { color:var(--white); border-color:var(--b2); }
  .container { padding:22px; max-width:1400px; margin:0 auto; }
  .top-row { display:grid; grid-template-columns: 1fr 2fr; gap:14px; margin-bottom:18px; }
  .panel { background:var(--card); border:1px solid var(--b1); padding:14px; clip-path:polygon(var(--cut) 0, 100% 0, 100% calc(100% - var(--cut)), calc(100% - var(--cut)) 100%, 0 100%, 0 var(--cut)); }
  .panel-hdr { font-size:9px; color:var(--dim3); letter-spacing:4px; text-transform:uppercase; margin-bottom:10px; }
  .stat-grid { display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; }
  .stat-cell { text-align:center; }
  .stat-num { font-size:22px; color:var(--white); font-weight:400; }
  .stat-num.warn { color:var(--amber); }
  .stat-num.crit { color:var(--red2); }
  .stat-lbl { font-size:8px; color:var(--dim3); letter-spacing:3px; text-transform:uppercase; margin-top:4px; }
  .btn { background:transparent; border:1px solid var(--b1); color:var(--dim2); font-family:inherit; font-size:9px; letter-spacing:3px; padding:5px 12px; cursor:pointer; text-transform:uppercase; }
  .btn:hover { color:var(--white); border-color:var(--b2); }
  .btn.primary { color:var(--white); border-color:var(--white); }
  .btn.danger:hover { color:var(--red2); border-color:var(--red); }
  input, textarea, select { background:var(--bg); border:1px solid var(--b1); color:var(--white); font-family:inherit; font-size:12px; padding:7px 10px; width:100%; }
  input:focus, textarea:focus, select:focus { outline:none; border-color:var(--b2); }
  label { display:block; font-size:9px; color:var(--dim3); letter-spacing:3px; text-transform:uppercase; margin-bottom:6px; margin-top:10px; }
  .form-row { display:flex; gap:10px; align-items:flex-end; margin-top:12px; }
  textarea { resize:vertical; min-height:60px; }
  .tasks { display:flex; flex-direction:column; gap:14px; }
  .task { background:var(--card); border:1px solid var(--b1); padding:16px; clip-path:polygon(var(--cut) 0, 100% 0, 100% calc(100% - var(--cut)), calc(100% - var(--cut)) 100%, 0 100%, 0 var(--cut)); }
  .task.done { opacity:.55; }
  .task.done .task-title { text-decoration:line-through; color:var(--dim2); }
  .task-hdr { display:flex; justify-content:space-between; align-items:start; gap:12px; margin-bottom:8px; }
  .task-title { font-size:14px; color:var(--white); flex:1; word-break:break-word; }
  .task-meta { display:flex; gap:8px; align-items:center; flex-shrink:0; }
  .task-id { font-size:9px; color:var(--dim3); letter-spacing:2px; }
  .task-prio { font-size:9px; padding:2px 8px; border:1px solid var(--b1); color:var(--dim2); letter-spacing:2px; text-transform:uppercase; }
  .task-prio.high { color:var(--red2); border-color:var(--red); }
  .task-prio.low { color:var(--dim3); }
  .task-status { font-size:9px; padding:2px 8px; letter-spacing:2px; text-transform:uppercase; border:1px solid var(--b1); color:var(--dim2); }
  .task-status.open { color:var(--white); border-color:var(--white); }
  .task-status.pending_review { color:var(--amber); border-color:var(--amber); }
  .task-status.done { color:var(--dim3); }
  #pending-badge { display:none; margin-left:6px; padding:1px 6px; background:var(--amber); color:var(--bg); font-size:8px; }
  #pending-badge.active { display:inline-block; }
  .task-desc { font-size:11px; color:var(--dim); line-height:1.5; margin-bottom:10px; }
  .progress { border-left:1px solid var(--b1); padding-left:12px; margin-top:10px; }
  .progress-hdr { font-size:9px; color:var(--dim3); letter-spacing:3px; text-transform:uppercase; margin-bottom:6px; }
  .progress-item { font-size:11px; color:var(--dim); line-height:1.5; margin-bottom:8px; }
  .progress-item .pi-meta { font-size:9px; color:var(--dim3); letter-spacing:1px; margin-bottom:2px; }
  .progress-item.thought { border-left:2px solid var(--dim3); padding-left:8px; }
  .progress-item.finding { border-left:2px solid var(--white); padding-left:8px; }
  .progress-item.question { border-left:2px solid var(--amber); padding-left:8px; }
  .progress-item.reflection { border-left:2px solid var(--white); padding-left:8px; background:rgba(255,255,255,0.02); padding:6px 8px; }
  .task-actions { display:flex; gap:6px; margin-top:12px; flex-wrap:wrap; }
  .empty { text-align:center; color:var(--dim3); padding:60px 0; font-style:italic; letter-spacing:2px; }
  .added-by { font-size:8px; color:var(--dim3); letter-spacing:2px; text-transform:uppercase; }
  .added-by.stella { color:var(--white); }
  .filter-row { display:flex; gap:6px; margin-bottom:14px; align-items:center; }
  .filter-row .btn.active { color:var(--white); border-color:var(--white); }
  details summary { cursor:pointer; font-size:9px; color:var(--dim2); letter-spacing:2px; text-transform:uppercase; padding:4px 0; }
  details summary:hover { color:var(--white); }
</style>
</head>
<body>
<header>
  <h1>STELLA // RESEARCH</h1>
  <div class="nav">
    <a href="/">← DASHBOARD</a>
    <a href="/chat">CHAT</a>
    <a href="/history">HISTORIAL</a>
    <a href="/library">LIBRARY</a>
  </div>
</header>
<div class="container">
  <div class="top-row">
    <div class="panel">
      <div class="panel-hdr">QUESTS</div>
      <div class="stat-grid">
        <div class="stat-cell"><div class="stat-num" id="s-open">0</div><div class="stat-lbl">ABIERTAS</div></div>
        <div class="stat-cell"><div class="stat-num" id="s-pending">0</div><div class="stat-lbl">PROPUESTAS</div></div>
        <div class="stat-cell"><div class="stat-num" id="s-done">0</div><div class="stat-lbl">CERRADAS</div></div>
        <div class="stat-cell"><div class="stat-num" id="s-progress">0</div><div class="stat-lbl">AVANCES</div></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-hdr">NUEVA QUEST</div>
      <label>TÍTULO</label>
      <input id="new-title" type="text" placeholder="ej. Preguntas abiertas sobre alineamiento de IA" maxlength="200">
      <label>DESCRIPCIÓN (opcional)</label>
      <textarea id="new-desc" placeholder="qué te interesa explorar, por qué importa, hacia dónde quieres llegar..."></textarea>
      <div class="form-row">
        <div style="flex:0 0 auto;">
          <label style="margin-top:0;">PRIORIDAD</label>
          <select id="new-prio">
            <option value="normal">NORMAL</option>
            <option value="high">ALTA</option>
            <option value="low">BAJA</option>
          </select>
        </div>
        <button class="btn primary" onclick="createTask()" style="margin-left:auto;">CREAR QUEST</button>
      </div>
    </div>
  </div>

  <div class="filter-row">
    <button class="btn active" data-filter="open"           onclick="setFilter('open')">ABIERTAS</button>
    <button class="btn"        data-filter="pending_review" onclick="setFilter('pending_review')">PROPUESTAS DE STELLA <span id="pending-badge"></span></button>
    <button class="btn"        data-filter="done"           onclick="setFilter('done')">CERRADAS</button>
    <button class="btn"        data-filter="all"            onclick="setFilter('all')">TODAS</button>
  </div>

  <div class="tasks" id="tasks">
    <div class="empty">cargando...</div>
  </div>
</div>

<script>
let currentFilter = 'open';
let allTasks = [];

function esc(t) { return String(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function loadStats() {
  try {
    const s = await (await fetch('/research/stats')).json();
    document.getElementById('s-open').textContent     = s.open || 0;
    document.getElementById('s-pending').textContent  = s.pending || 0;
    document.getElementById('s-done').textContent     = s.done || 0;
    document.getElementById('s-progress').textContent = s.progress_total || 0;
    const badge = document.getElementById('pending-badge');
    if (badge) {
      if (s.pending > 0) {
        badge.textContent = s.pending;
        badge.classList.add('active');
      } else {
        badge.classList.remove('active');
      }
    }
    // resaltar el bloque PROPUESTAS si hay alguna
    const pendCell = document.getElementById('s-pending');
    if (pendCell) pendCell.className = 'stat-num' + (s.pending > 0 ? ' warn' : '');
  } catch(e) {}
}

async function loadTasks() {
  try {
    const d = await (await fetch('/research/list')).json();
    allTasks = d.tasks || [];
    renderTasks();
  } catch(e) {
    document.getElementById('tasks').innerHTML = '<div class="empty">error: ' + e.message + '</div>';
  }
}

function setFilter(f) {
  currentFilter = f;
  document.querySelectorAll('.filter-row .btn').forEach(b => {
    b.classList.toggle('active', b.dataset.filter === f);
  });
  renderTasks();
}

function renderTasks() {
  let list = allTasks;
  if (currentFilter === 'open')           list = allTasks.filter(t => (t.status || 'open') === 'open');
  if (currentFilter === 'pending_review') list = allTasks.filter(t => t.status === 'pending_review');
  if (currentFilter === 'done')           list = allTasks.filter(t => t.status === 'done');
  // ordenar: pending_review primero, luego open, luego done
  const statusRank = { pending_review:0, open:1, done:2 };
  const prioRank = { high:0, normal:1, low:2 };
  list = [...list].sort((a, b) => {
    const sa = statusRank[a.status||'open'] ?? 1;
    const sb = statusRank[b.status||'open'] ?? 1;
    if (sa !== sb) return sa - sb;
    if (a.status === 'open' || a.status === 'pending_review') {
      const pa = prioRank[a.priority||'normal'] ?? 1;
      const pb = prioRank[b.priority||'normal'] ?? 1;
      if (pa !== pb) return pa - pb;
      return (b.ts_updated||'').localeCompare(a.ts_updated||'');
    }
    return (b.ts_updated||'').localeCompare(a.ts_updated||'');
  });

  const box = document.getElementById('tasks');
  if (!list.length) {
    box.innerHTML = '<div class="empty">sin quests' + (currentFilter !== 'all' ? ' ' + currentFilter : '') + '</div>';
    return;
  }
  box.innerHTML = list.map(t => renderTask(t)).join('');
}

function renderTask(t) {
  const status = t.status || 'open';
  const prio   = t.priority || 'normal';
  const progress = t.progress || [];
  const created = (t.ts_created || '').slice(0, 10);
  const updated = (t.ts_updated || '').slice(0, 16).replace('T', ' ');
  const addedBy = t.added_by || 'stella';
  const summary = (t.summary || '').trim();

  // Resumen vivo si existe, si no fallback a últimos 3 avances
  let progHtml = '';
  if (summary && progress.length >= 3) {
    progHtml = '<div class="progress"><div class="progress-hdr">hilo (' + progress.length + ' avances · resumen vivo)</div>' +
      '<div class="progress-item reflection">' + esc(summary) + '</div>';
    progHtml += '<details><summary>VER TODOS LOS AVANCES</summary>' +
      progress.map(p => {
        const k = p.kind || 'thought';
        const t = (p.ts||'').slice(0, 16).replace('T',' ');
        return '<div class="progress-item ' + esc(k) + '" style="margin-top:6px;">' +
          '<div class="pi-meta">' + esc(k.toUpperCase()) + ' · ' + t + '</div>' +
          esc(p.content || '') +
        '</div>';
      }).join('') + '</details>';
    progHtml += '</div>';
  } else if (progress.length) {
    const recent = progress.slice(-3);
    progHtml = '<div class="progress"><div class="progress-hdr">avances (' + progress.length + ')</div>' +
      recent.map(p => {
        const k = p.kind || 'thought';
        const t = (p.ts||'').slice(11, 16);
        return '<div class="progress-item ' + esc(k) + '">' +
          '<div class="pi-meta">' + esc(k.toUpperCase()) + ' · ' + t + '</div>' +
          esc(p.content || '') +
        '</div>';
      }).join('');
    if (progress.length > 3) {
      progHtml += '<details><summary>VER ' + (progress.length - 3) + ' AVANCES ANTERIORES</summary>' +
        progress.slice(0, -3).map(p => {
          const k = p.kind || 'thought';
          const t = (p.ts||'').slice(0, 16).replace('T',' ');
          return '<div class="progress-item ' + esc(k) + '" style="margin-top:6px;">' +
            '<div class="pi-meta">' + esc(k.toUpperCase()) + ' · ' + t + '</div>' +
            esc(p.content || '') +
          '</div>';
        }).join('') + '</details>';
    }
    progHtml += '</div>';
  }

  // Botonera según estado
  let actionsHtml = '';
  if (status === 'pending_review') {
    actionsHtml = '<div class="task-actions">' +
      '<button class="btn primary" onclick="approveTask(\\'' + esc(t.id) + '\\')">✓ APROBAR</button>' +
      '<button class="btn danger"  onclick="deleteTask(\\'' + esc(t.id) + '\\')">RECHAZAR</button>' +
    '</div>';
  } else if (status === 'open') {
    actionsHtml = '<div class="task-actions">' +
      '<button class="btn primary" onclick="openMap(\\'' + esc(t.id) + '\\')">▭ VER MAPA</button>' +
      '<button class="btn" onclick="addProgress(\\'' + esc(t.id) + '\\')">+ AVANCE</button>' +
      '<button class="btn" onclick="closeTask(\\'' + esc(t.id) + '\\')">CERRAR QUEST</button>' +
      '<button class="btn danger" onclick="deleteTask(\\'' + esc(t.id) + '\\')">ELIMINAR</button>' +
    '</div>';
  } else {
    actionsHtml = '<div class="task-actions">' +
      '<button class="btn primary" onclick="openMap(\\'' + esc(t.id) + '\\')">▭ VER MAPA</button>' +
      '<button class="btn danger" onclick="deleteTask(\\'' + esc(t.id) + '\\')">ELIMINAR</button>' +
    '</div>';
  }

  // Etiqueta del status
  const statusLabel = status === 'pending_review' ? 'PROPUESTA' : status;

  const selfTag = t.self_approved ? '<span class="task-status" style="border-color:var(--amber);color:var(--amber);" title="Auto-aprobada por Stella">SELF</span>' : '';
  return '<div class="task' + (status === 'done' ? ' done' : '') + '">' +
    '<div class="task-hdr">' +
      '<div class="task-title">' + esc(t.title || '') + '</div>' +
      '<div class="task-meta">' +
        '<span class="task-id">' + esc(t.id || '') + '</span>' +
        (prio !== 'normal' ? '<span class="task-prio ' + esc(prio) + '">' + esc(prio) + '</span>' : '') +
        selfTag +
        '<span class="task-status ' + esc(status) + '">' + esc(statusLabel) + '</span>' +
      '</div>' +
    '</div>' +
    (t.description ? '<div class="task-desc">' + esc(t.description) + '</div>' : '') +
    '<div style="display:flex;justify-content:space-between;align-items:center;font-size:8px;color:var(--dim3);letter-spacing:2px;">' +
      '<span>creada ' + created + ' · actualizada ' + updated + '</span>' +
      '<span class="added-by ' + esc(addedBy) + '">añadida por ' + esc(addedBy) + '</span>' +
    '</div>' +
    progHtml +
    actionsHtml +
  '</div>';
}

async function approveTask(id) {
  if (!confirm('Aprobar esta quest? Stella podrá empezar a avanzarla en idle.')) return;
  try {
    const r = await fetch('/research/' + id + '/approve', {method:'POST'});
    const d = await r.json();
    if (d.ok) { loadStats(); loadTasks(); }
  } catch(e) {}
}

function openMap(id) {
  window.location.href = '/research/' + id + '/map';
}

async function createTask() {
  const title = document.getElementById('new-title').value.trim();
  const desc  = document.getElementById('new-desc').value.trim();
  const prio  = document.getElementById('new-prio').value;
  if (!title) { alert('Pon un título.'); return; }
  try {
    const r = await fetch('/research/create', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({title, description: desc, priority: prio})
    });
    const d = await r.json();
    if (d.ok) {
      document.getElementById('new-title').value = '';
      document.getElementById('new-desc').value = '';
      document.getElementById('new-prio').value = 'normal';
      loadStats(); loadTasks();
    }
  } catch(e) { alert('error: ' + e.message); }
}

async function addProgress(id) {
  const content = prompt('Avance para esta quest:');
  if (!content) return;
  try {
    const r = await fetch('/research/' + id + '/progress', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({content, kind:'thought'})
    });
    const d = await r.json();
    if (d.ok) { loadStats(); loadTasks(); }
  } catch(e) {}
}

async function closeTask(id) {
  const conclusion = prompt('Conclusión final (opcional):');
  if (conclusion === null) return;  // canceló
  try {
    const r = await fetch('/research/' + id + '/close', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({conclusion})
    });
    const d = await r.json();
    if (d.ok) { loadStats(); loadTasks(); }
  } catch(e) {}
}

async function deleteTask(id) {
  if (!confirm('Eliminar quest ' + id + ' permanentemente?')) return;
  try {
    const r = await fetch('/research/' + id + '/delete', {method:'POST'});
    const d = await r.json();
    if (d.ok) { loadStats(); loadTasks(); }
  } catch(e) {}
}

loadStats();
loadTasks();
setInterval(() => { loadStats(); loadTasks(); }, 10000);
</script>
</body>
</html>"""


@app.route("/research")
def research_page():
    return render_template_string(RESEARCH_HTML)


RESEARCH_MAP_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Stella // Research Map</title>
<style>
  :root {
    --bg:#000; --card:#050505; --b0:#1a1a1a; --b1:#2a2a2a; --b2:#555;
    --white:#fff; --dim:#aaa; --dim2:#666; --dim3:#444;
    --red:#d01818; --red2:#ff2a2a; --amber:#999;
    --cut:12px;
  }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--dim); font-family:'Consolas','Monaco',monospace; font-size:13px; margin:0; padding:0; letter-spacing:.3px; overflow:hidden; height:100vh; }
  header { padding:14px 22px; border-bottom:1px solid var(--b1); display:flex; align-items:center; justify-content:space-between; gap:20px; position:relative; z-index:10; background:var(--bg); }
  header h1 { margin:0; font-size:14px; letter-spacing:4px; font-weight:400; color:var(--white); }
  header h1 .quest-title { color:var(--dim); margin-left:14px; font-size:12px; letter-spacing:2px; }
  header h1 .quest-id    { color:var(--dim3); margin-left:10px; font-size:10px; letter-spacing:2px; }
  header .nav a { color:var(--dim2); text-decoration:none; font-size:10px; letter-spacing:3px; padding:4px 10px; border:1px solid var(--b1); margin-left:6px; text-transform:uppercase; }
  header .nav a:hover { color:var(--white); border-color:var(--b2); }

  /* fondo malla pixelado animado, estilo HUD */
  .map-container {
    position:relative;
    width:100%;
    height:calc(100vh - 53px);
    overflow:hidden;
    background:
      radial-gradient(ellipse at 30% 40%, rgba(255,255,255,0.025) 0%, transparent 55%),
      radial-gradient(ellipse at 70% 60%, rgba(208,24,24,0.018) 0%, transparent 50%),
      var(--bg);
  }
  .grid-bg {
    position:absolute; inset:0;
    background-image:
      linear-gradient(to right,  rgba(255,255,255,0.045) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(255,255,255,0.045) 1px, transparent 1px),
      linear-gradient(to right,  rgba(255,255,255,0.018) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(255,255,255,0.018) 1px, transparent 1px);
    background-size: 60px 60px, 60px 60px, 12px 12px, 12px 12px;
    background-position: 0 0, 0 0, 0 0, 0 0;
    animation: grid-pan 30s linear infinite;
    pointer-events:none;
  }
  .grid-dots {
    position:absolute; inset:0;
    background-image: radial-gradient(circle, rgba(255,255,255,0.18) 1px, transparent 1.5px);
    background-size: 60px 60px;
    background-position: 0 0;
    animation: grid-pan-slow 60s linear infinite;
    pointer-events:none;
  }
  .grid-scan {
    position:absolute; inset:0;
    background:
      repeating-linear-gradient(0deg,
        transparent 0,
        transparent 3px,
        rgba(255,255,255,0.012) 3px,
        rgba(255,255,255,0.012) 4px);
    pointer-events:none;
  }
  @keyframes grid-pan {
    from { background-position: 0 0, 0 0, 0 0, 0 0; }
    to   { background-position: 60px 60px, 60px 60px, 12px 12px, 12px 12px; }
  }
  @keyframes grid-pan-slow {
    from { background-position: 0 0; }
    to   { background-position: -60px 60px; }
  }
  /* glitch scan-line ocasional */
  .glitch {
    position:absolute; left:0; right:0; height:2px;
    background:linear-gradient(to right, transparent, rgba(255,255,255,0.04), transparent);
    pointer-events:none;
    animation: glitch-move 11s linear infinite;
  }
  @keyframes glitch-move {
    0%   { top:-2%; opacity:0; }
    8%   { opacity:1; }
    50%  { top:50%; opacity:0.6; }
    92%  { opacity:1; }
    100% { top:102%; opacity:0; }
  }

  svg.tree { position:absolute; inset:0; width:100%; height:100%; }

  /* nodos */
  .node-root rect, .node-leaf rect {
    fill:var(--card);
    stroke:var(--b1);
    stroke-width:1;
  }
  .node-root rect { stroke:var(--white); stroke-width:1.5; }
  .node-leaf.k-question rect { stroke:var(--amber); }
  .node-leaf.k-finding rect { stroke:var(--white); }
  .node-leaf.k-reflection rect { stroke:var(--white); stroke-width:1.5; fill:rgba(255,255,255,0.04); }
  .node-leaf.k-thought rect { stroke:var(--dim2); }
  .node-leaf.k-pivot rect { stroke:var(--red2); }
  .node-leaf.k-url_saved rect { stroke:var(--white); stroke-dasharray: 4 3; }

  .node text { fill:var(--white); font-family:'Consolas',monospace; font-size:11px; }
  .node .node-kind { fill:var(--dim3); font-size:8px; letter-spacing:2px; text-transform:uppercase; }
  .node .node-time { fill:var(--dim3); font-size:8px; letter-spacing:1px; }
  .node-root .node-title { font-size:12px; }
  .node-root .node-desc  { fill:var(--dim); font-size:10px; }

  /* conectores */
  .edge { fill:none; stroke:var(--b2); stroke-width:1; opacity:.55; }
  .edge.k-question   { stroke:var(--amber); }
  .edge.k-finding    { stroke:var(--white); }
  .edge.k-reflection { stroke:var(--white); }
  .edge.k-thought    { stroke:var(--dim2); }
  .edge.k-pivot      { stroke:var(--red2); }
  .edge.k-url_saved  { stroke:var(--white); stroke-dasharray: 5 4; }

  /* hover */
  .node:hover rect { fill:rgba(255,255,255,0.06); }
  .node:hover { cursor:pointer; }
  .node:hover .node-title { font-weight:bold; }

  /* tooltip flotante */
  #tooltip {
    position:absolute; pointer-events:none; opacity:0; transition:opacity .15s;
    background:var(--card); border:1px solid var(--b2); padding:10px 14px;
    max-width:420px; font-size:11px; color:var(--dim); line-height:1.5;
    clip-path:polygon(var(--cut) 0, 100% 0, 100% calc(100% - var(--cut)), calc(100% - var(--cut)) 100%, 0 100%, 0 var(--cut));
    z-index:50;
  }
  #tooltip .tt-kind { font-size:9px; letter-spacing:2px; color:var(--dim3); text-transform:uppercase; margin-bottom:6px; }
  #tooltip .tt-time { font-size:9px; color:var(--dim3); margin-top:6px; }
  #tooltip.k-finding,    #tooltip.k-finding .tt-kind    { color:var(--white); }
  #tooltip.k-question  .tt-kind { color:var(--amber); }
  #tooltip.k-pivot     .tt-kind { color:var(--red2); }
  #tooltip.k-reflection .tt-kind { color:var(--white); }

  /* leyenda */
  .legend {
    position:absolute; bottom:18px; left:22px;
    background:var(--card); border:1px solid var(--b1); padding:10px 14px;
    font-size:9px; letter-spacing:2px; text-transform:uppercase; color:var(--dim2);
    clip-path:polygon(var(--cut) 0, 100% 0, 100% calc(100% - var(--cut)), calc(100% - var(--cut)) 100%, 0 100%, 0 var(--cut));
    display:flex; gap:14px;
  }
  .legend span { display:flex; align-items:center; gap:6px; }
  .legend i { display:inline-block; width:10px; height:10px; border:1px solid; }
  .legend i.q { border-color:var(--amber); }
  .legend i.f { border-color:var(--white); }
  .legend i.r { border-color:var(--white); background:rgba(255,255,255,0.1); }
  .legend i.t { border-color:var(--dim2); }
  .legend i.p { border-color:var(--red2); }

  .empty {
    position:absolute; top:50%; left:50%; transform:translate(-50%, -50%);
    color:var(--dim3); font-style:italic; letter-spacing:3px;
  }
  svg.tree { cursor:grab; }
  svg.tree.dragging { cursor:grabbing; }
  .zoom-ctrl {
    position:absolute; bottom:18px; right:22px;
    display:flex; flex-direction:column; gap:4px; z-index:20;
  }
  .zoom-btn {
    background:var(--card); border:1px solid var(--b1); color:var(--dim2);
    font-family:'Consolas',monospace; font-size:13px; letter-spacing:1px;
    width:34px; height:28px; cursor:pointer; text-align:center; line-height:1;
  }
  .zoom-btn:hover { color:var(--white); border-color:var(--b2); }
  .zoom-label {
    font-size:8px; letter-spacing:2px; color:var(--dim3); text-align:center;
    margin-top:2px; text-transform:uppercase;
  }
</style>
</head>
<body>
<header>
  <h1>STELLA // RESEARCH MAP
    <span class="quest-title" id="q-title">cargando…</span>
    <span class="quest-id" id="q-id"></span>
  </h1>
  <div class="nav">
    <a href="/research">← RESEARCH</a>
    <a href="/">DASHBOARD</a>
  </div>
</header>
<div class="map-container">
  <div class="grid-bg"></div>
  <div class="grid-dots"></div>
  <div class="grid-scan"></div>
  <div class="glitch"></div>

  <svg class="tree" id="tree-svg"></svg>

  <div class="legend">
    <span><i class="q"></i> QUESTION</span>
    <span><i class="t"></i> THOUGHT</span>
    <span><i class="f"></i> FINDING</span>
    <span><i class="r"></i> REFLECTION</span>
    <span><i class="p"></i> PIVOT</span>
  </div>
  <div class="zoom-ctrl">
    <button class="zoom-btn" onclick="zoomStep(1.25)" title="Zoom in">+</button>
    <button class="zoom-btn" onclick="fitAll()"       title="Fit all">⊡</button>
    <button class="zoom-btn" onclick="zoomStep(0.8)"  title="Zoom out">−</button>
    <div class="zoom-label" id="zoom-pct">100%</div>
  </div>
  <div id="tooltip"></div>
  <div class="empty" id="empty" style="display:none;">— sin avances aún en esta quest —</div>
</div>

<script>
const QUEST_ID = window.location.pathname.split('/').filter(Boolean)[1];
const ns = 'http://www.w3.org/2000/svg';

function esc(t) { return String(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function truncate(t, n) { t = String(t||''); return t.length > n ? t.slice(0, n).trim() + '…' : t; }
function pathRoot(t) {
  if (!t) return t;
  const lines = String(t).split('\\n');
  return lines[0].slice(0, 80);
}

// -- pan / zoom state --------------------------------------------------------
let vpTx = 0, vpTy = 0, vpScale = 1;
let dragging = false, dragStartX = 0, dragStartY = 0, dragTx = 0, dragTy = 0;

const svg = document.getElementById('tree-svg');
let viewport = null;   // <g id="viewport"> created in render()

function applyTransform() {
  if (viewport) viewport.setAttribute('transform', `translate(${vpTx},${vpTy}) scale(${vpScale})`);
  document.getElementById('zoom-pct').textContent = Math.round(vpScale * 100) + '%';
}

function zoomAt(factor, cx, cy) {
  const newScale = Math.max(0.06, Math.min(5, vpScale * factor));
  const f = newScale / vpScale;
  vpTx = cx - f * (cx - vpTx);
  vpTy = cy - f * (cy - vpTy);
  vpScale = newScale;
  applyTransform();
}

function zoomStep(factor) {
  const rect = svg.getBoundingClientRect();
  zoomAt(factor, rect.width / 2, rect.height / 2);
}

// virtual canvas size — set during render, used by fitAll
let vcW = 1600, vcH = 800;

function fitAll() {
  const rect = svg.getBoundingClientRect();
  const W = rect.width, H = rect.height;
  const pad = 48;
  vpScale = Math.min((W - pad * 2) / vcW, (H - pad * 2) / vcH);
  vpScale = Math.max(0.06, Math.min(5, vpScale));
  vpTx = (W - vcW * vpScale) / 2;
  vpTy = (H - vcH * vpScale) / 2;
  applyTransform();
}

// wheel zoom
svg.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = svg.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  zoomAt(e.deltaY < 0 ? 1.12 : 0.89, cx, cy);
}, { passive: false });

// drag pan
svg.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  dragging = true;
  dragStartX = e.clientX; dragStartY = e.clientY;
  dragTx = vpTx; dragTy = vpTy;
  svg.classList.add('dragging');
  e.preventDefault();
});
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  vpTx = dragTx + (e.clientX - dragStartX);
  vpTy = dragTy + (e.clientY - dragStartY);
  applyTransform();
});
window.addEventListener('mouseup', () => { dragging = false; svg.classList.remove('dragging'); });

// touch pinch + drag
let touches = {};
svg.addEventListener('touchstart', (e) => {
  [...e.changedTouches].forEach(t => touches[t.identifier] = {x: t.clientX, y: t.clientY});
  if (e.touches.length === 1) {
    dragging = true;
    dragStartX = e.touches[0].clientX; dragStartY = e.touches[0].clientY;
    dragTx = vpTx; dragTy = vpTy;
  }
  e.preventDefault();
}, { passive: false });
svg.addEventListener('touchmove', (e) => {
  if (e.touches.length === 2) {
    dragging = false;
    const t0 = e.touches[0], t1 = e.touches[1];
    const prev0 = touches[t0.identifier] || t0, prev1 = touches[t1.identifier] || t1;
    const prevDist = Math.hypot(prev0.x - prev1.x, prev0.y - prev1.y);
    const curDist  = Math.hypot(t0.clientX - t1.clientX, t0.clientY - t1.clientY);
    if (prevDist > 0) {
      const rect = svg.getBoundingClientRect();
      const cx = (t0.clientX + t1.clientX) / 2 - rect.left;
      const cy = (t0.clientY + t1.clientY) / 2 - rect.top;
      zoomAt(curDist / prevDist, cx, cy);
    }
  } else if (e.touches.length === 1 && dragging) {
    vpTx = dragTx + (e.touches[0].clientX - dragStartX);
    vpTy = dragTy + (e.touches[0].clientY - dragStartY);
    applyTransform();
  }
  [...e.changedTouches].forEach(t => touches[t.identifier] = {x: t.clientX, y: t.clientY});
  e.preventDefault();
}, { passive: false });
svg.addEventListener('touchend', (e) => {
  [...e.changedTouches].forEach(t => delete touches[t.identifier]);
  if (e.touches.length === 0) dragging = false;
});

// -- render ------------------------------------------------------------------

async function loadTask() {
  try {
    const r = await fetch('/research/' + QUEST_ID);
    if (!r.ok) { document.getElementById('q-title').textContent = 'quest no encontrada'; return; }
    const d = await r.json();
    render(d.task);
  } catch(e) {
    document.getElementById('q-title').textContent = 'error: ' + e.message;
  }
}

function render(task) {
  document.getElementById('q-title').textContent = task.title || '';
  document.getElementById('q-id').textContent    = task.id   || '';

  // dimensiones del SVG en pantalla
  const rect = svg.getBoundingClientRect();
  const SW = rect.width  || window.innerWidth;
  const SH = rect.height || (window.innerHeight - 53);
  svg.setAttribute('width',  SW);
  svg.setAttribute('height', SH);
  svg.setAttribute('viewBox', `0 0 ${SW} ${SH}`);
  svg.innerHTML = '';

  // crear viewport group
  viewport = document.createElementNS(ns, 'g');
  viewport.setAttribute('id', 'viewport');
  svg.appendChild(viewport);

  const progress = task.progress || [];

  // -- virtual canvas --
  const rootW  = 340;
  const rootH  = task.description ? 100 : 66;
  const leafW  = 400;
  const leafH  = 68;
  const leafGap = 20;
  const n       = progress.length;

  // virtual height: enough for all leaves with generous spacing
  const minVH = Math.max(600, n * (leafH + leafGap) + 120);
  vcW = 900;
  vcH = minVH;

  const rootX = 40;
  const rootY = vcH / 2 - rootH / 2;

  if (!n) {
    document.getElementById('empty').style.display = '';
    drawNode(viewport, {
      cls: 'node-root', x: rootX, y: rootY, w: rootW, h: rootH,
      title: task.title || '', subtitle: task.description || '',
      kind: task.status === 'done' ? 'finding' : '', time: '',
      fullContent: task.description || '',
    });
    fitAll(); return;
  }
  document.getElementById('empty').style.display = 'none';

  const sorted   = [...progress].sort((a, b) => (a.ts||'').localeCompare(b.ts||''));
  const leafX    = vcW - leafW - 40;
  const blockH   = n * leafH + (n - 1) * leafGap;
  const leafStartY = vcH / 2 - blockH / 2;

  // edges first (behind nodes)
  sorted.forEach((p, i) => {
    const cy = leafStartY + i * (leafH + leafGap) + leafH / 2;
    drawEdge(viewport, rootX + rootW, rootY + rootH / 2, leafX, cy, p.kind || 'thought');
  });

  // root
  drawNode(viewport, {
    cls: 'node-root', x: rootX, y: rootY, w: rootW, h: rootH,
    title: task.title || '', subtitle: task.description || '',
    kind: task.status === 'done' ? 'finding' : '', time: '',
    fullContent: (task.title||'') + (task.description ? '\\n\\n' + task.description : ''),
  });

  // leaves
  sorted.forEach((p, i) => {
    const y = leafStartY + i * (leafH + leafGap);
    drawNode(viewport, {
      cls: 'node-leaf k-' + (p.kind || 'thought'),
      x: leafX, y: y, w: leafW, h: leafH,
      title: pathRoot(p.content || ''), subtitle: '',
      kind: p.kind || 'thought', time: (p.ts||'').slice(11, 16),
      fullContent: p.content || '',
    });
  });

  fitAll();
}

function drawEdge(parent, x1, y1, x2, y2, kind) {
  const ctrl = (x2 - x1) * 0.55;
  const d = `M ${x1} ${y1} C ${x1+ctrl} ${y1}, ${x2-ctrl} ${y2}, ${x2} ${y2}`;
  const path = document.createElementNS(ns, 'path');
  path.setAttribute('d', d);
  path.setAttribute('class', 'edge k-' + (kind || 'thought'));
  parent.appendChild(path);
}

function drawNode(parent, opts) {
  const g = document.createElementNS(ns, 'g');
  g.setAttribute('class', 'node ' + (opts.cls || ''));
  g.setAttribute('transform', `translate(${opts.x} ${opts.y})`);

  const rect = document.createElementNS(ns, 'rect');
  rect.setAttribute('width', opts.w);
  rect.setAttribute('height', opts.h);
  g.appendChild(rect);

  const padL = 12, padT = 16;

  if (opts.kind) {
    const k = document.createElementNS(ns, 'text');
    k.setAttribute('class', 'node-kind');
    k.setAttribute('x', padL); k.setAttribute('y', padT - 4);
    k.textContent = opts.kind.toUpperCase();
    g.appendChild(k);
  }
  if (opts.time) {
    const t = document.createElementNS(ns, 'text');
    t.setAttribute('class', 'node-time');
    t.setAttribute('text-anchor', 'end');
    t.setAttribute('x', opts.w - padL); t.setAttribute('y', padT - 4);
    t.textContent = opts.time;
    g.appendChild(t);
  }

  const tt = document.createElementNS(ns, 'text');
  tt.setAttribute('class', 'node-title');
  tt.setAttribute('x', padL); tt.setAttribute('y', padT + 14);
  const maxChars = Math.floor((opts.w - padL * 2) / 6.5);
  tt.textContent = truncate(opts.title, maxChars);
  g.appendChild(tt);

  if (opts.subtitle) {
    const desc = document.createElementNS(ns, 'text');
    desc.setAttribute('class', 'node-desc');
    desc.setAttribute('x', padL); desc.setAttribute('y', padT + 32);
    const maxChars2 = Math.floor((opts.w - padL * 2) / 6.0);
    desc.textContent = truncate(opts.subtitle.replace(/\\n/g, ' '), maxChars2);
    g.appendChild(desc);
  }

  g.addEventListener('mouseenter', (e) => showTooltip(e, opts));
  g.addEventListener('mousemove',  (e) => moveTooltip(e));
  g.addEventListener('mouseleave', hideTooltip);

  parent.appendChild(g);
}

const tooltip = document.getElementById('tooltip');
function showTooltip(e, opts) {
  if (!opts.fullContent) return;
  tooltip.innerHTML =
    (opts.kind ? '<div class="tt-kind">' + esc(opts.kind) + '</div>' : '') +
    esc(opts.fullContent).replace(/\\n/g, '<br>') +
    (opts.time ? '<div class="tt-time">' + esc(opts.time) + '</div>' : '');
  tooltip.className = opts.kind ? ('k-' + opts.kind) : '';
  tooltip.style.opacity = '1';
  moveTooltip(e);
}
function moveTooltip(e) {
  const pad = 14;
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + tw > window.innerWidth  - 10) x = e.clientX - tw - pad;
  if (y + th > window.innerHeight - 10) y = e.clientY - th - pad;
  tooltip.style.left = x + 'px'; tooltip.style.top = y + 'px';
}
function hideTooltip() { tooltip.style.opacity = '0'; }

loadTask();
window.addEventListener('resize', () => loadTask());
setInterval(loadTask, 12000);
</script>
</body>
</html>"""


@app.route("/research/<task_id>/map")
def research_map_page(task_id):
    return render_template_string(RESEARCH_MAP_HTML)


# -- Library endpoints + page -----------------------------------------------

@app.route("/library/stats")
def library_stats_ep():
    try:
        from library import vault_stats
        return jsonify(vault_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/library/list")
def library_list_ep():
    try:
        from library import vault_list
        n = request.args.get("n", type=int)
        return jsonify({"items": vault_list(n=n)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/library/summary")
def library_summary_ep():
    try:
        from library import read_library_summary
        return jsonify({"summary": read_library_summary() or ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/library/item/<item_id>/delete", methods=["POST"])
def library_item_delete(item_id):
    try:
        from library import vault_delete
        ok = vault_delete(item_id)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/library/item/<item_id>/file")
def library_item_file(item_id):
    """Sirve el archivo bruto de un item del vault."""
    try:
        from library import vault_get_item
        item = vault_get_item(item_id)
        if not item or not item.get("path"):
            return jsonify({"error": "not found"}), 404
        path = Path(item["path"])
        if not path.exists():
            return jsonify({"error": "file missing"}), 404
        return send_from_directory(path.parent, path.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/library/regenerate", methods=["POST"])
def library_regenerate():
    try:
        from library import regenerate_library_summary
        text = regenerate_library_summary(
            llm_endpoint=LLM_ENDPOINT, llm_model=LLM_MODEL, llm_api_key=LLM_API_KEY,
        )
        return jsonify({"ok": True, "summary": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


LIBRARY_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Stella // Library</title>
<style>
  :root {
    --bg:    #000;
    --card:  #050505;
    --b0:    #1a1a1a;
    --b1:    #2a2a2a;
    --b2:    #555;
    --white: #fff;
    --dim:   #aaa;
    --dim2:  #666;
    --dim3:  #444;
    --red:   #d01818;
    --red2:  #ff2a2a;
    --amber: #999;
    --cut:   12px;
  }
  * { box-sizing: border-box; }
  body { background:var(--bg); color:var(--dim); font-family:'Consolas','Monaco',monospace; font-size:13px; margin:0; padding:0; letter-spacing:.3px; }
  header { padding:14px 22px; border-bottom:1px solid var(--b1); display:flex; align-items:center; justify-content:space-between; gap:20px; }
  header h1 { margin:0; font-size:14px; letter-spacing:4px; font-weight:400; color:var(--white); }
  header .nav a { color:var(--dim2); text-decoration:none; font-size:10px; letter-spacing:3px; padding:4px 10px; border:1px solid var(--b1); margin-left:6px; text-transform:uppercase; }
  header .nav a:hover { color:var(--white); border-color:var(--b2); }
  .container { padding:22px; max-width:1400px; margin:0 auto; }
  .top-row { display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:18px; }
  .panel { background:var(--card); border:1px solid var(--b1); padding:14px; clip-path:polygon(var(--cut) 0, 100% 0, 100% calc(100% - var(--cut)), calc(100% - var(--cut)) 100%, 0 100%, 0 var(--cut)); }
  .panel-hdr { font-size:9px; color:var(--dim3); letter-spacing:4px; text-transform:uppercase; margin-bottom:10px; }
  .stat-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; }
  .stat-cell { text-align:center; }
  .stat-num { font-size:22px; color:var(--white); font-weight:400; }
  .stat-num.warn { color:var(--amber); }
  .stat-num.crit { color:var(--red2); }
  .stat-lbl { font-size:8px; color:var(--dim3); letter-spacing:3px; text-transform:uppercase; margin-top:4px; }
  .quota-bar { height:4px; background:var(--b0); margin-top:12px; position:relative; overflow:hidden; }
  .quota-fill { height:100%; background:var(--white); transition:width .3s, background .3s; }
  .quota-fill.warn { background:var(--amber); }
  .quota-fill.crit { background:var(--red2); }
  .summary-box { font-size:11px; line-height:1.7; color:var(--dim); white-space:pre-wrap; }
  .summary-empty { color:var(--dim3); font-style:italic; }
  .btn { background:transparent; border:1px solid var(--b1); color:var(--dim2); font-family:inherit; font-size:9px; letter-spacing:3px; padding:5px 12px; cursor:pointer; text-transform:uppercase; }
  .btn:hover { color:var(--white); border-color:var(--b2); }
  .btn.active { color:var(--white); border-color:var(--white); }
  .items { display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap:12px; }
  .item { background:var(--card); border:1px solid var(--b1); padding:12px; position:relative; clip-path:polygon(var(--cut) 0, 100% 0, 100% calc(100% - var(--cut)), calc(100% - var(--cut)) 100%, 0 100%, 0 var(--cut)); }
  .item:hover { border-color:var(--b2); }
  .item .tp { font-size:8px; letter-spacing:3px; color:var(--dim3); text-transform:uppercase; }
  .item .ttl { font-size:12px; color:var(--white); margin:6px 0 8px; word-break:break-word; }
  .item .ttl a { color:inherit; text-decoration:none; border-bottom:1px dotted var(--b1); }
  .item .ttl a:hover { border-bottom-color:var(--white); }
  .item .why { font-size:11px; color:var(--dim); margin-bottom:8px; line-height:1.45; }
  .item .meta { font-size:8px; color:var(--dim3); display:flex; gap:10px; letter-spacing:2px; text-transform:uppercase; }
  .item .acts { position:absolute; top:8px; right:8px; display:flex; gap:4px; opacity:0; transition:opacity .15s; }
  .item:hover .acts { opacity:1; }
  .item .acts button { background:transparent; border:1px solid var(--b1); color:var(--dim2); font-size:11px; padding:2px 6px; cursor:pointer; line-height:1; }
  .item .acts button:hover { color:var(--red2); border-color:var(--red); }
  .empty { text-align:center; color:var(--dim3); padding:60px 0; font-style:italic; letter-spacing:2px; }
  .filter-row { display:flex; gap:6px; margin-bottom:14px; }
</style>
</head>
<body>
<header>
  <h1>STELLA // LIBRARY</h1>
  <div class="nav">
    <a href="/">← DASHBOARD</a>
    <a href="/chat">CHAT</a>
    <a href="/research">RESEARCH</a>
    <a href="/history">HISTORIAL</a>
  </div>
</header>
<div class="container">
  <div class="top-row">
    <div class="panel">
      <div class="panel-hdr">CUOTA</div>
      <div class="stat-grid">
        <div class="stat-cell"><div class="stat-num" id="s-items">0</div><div class="stat-lbl">ITEMS</div></div>
        <div class="stat-cell"><div class="stat-num" id="s-mb">0.0</div><div class="stat-lbl">MB USADOS</div></div>
        <div class="stat-cell"><div class="stat-num" id="s-quota">5000</div><div class="stat-lbl">MB MAX</div></div>
        <div class="stat-cell"><div class="stat-num" id="s-pct">0%</div><div class="stat-lbl">USO</div></div>
      </div>
      <div class="quota-bar"><div class="quota-fill" id="qfill" style="width:0%"></div></div>
    </div>
    <div class="panel">
      <div class="panel-hdr" style="display:flex; justify-content:space-between;">
        <span>RESUMEN VIVO</span>
        <button class="btn" onclick="regenSummary()">REGENERAR</button>
      </div>
      <div class="summary-box" id="summary"><span class="summary-empty">sin items aún</span></div>
    </div>
  </div>

  <div class="filter-row">
    <button class="btn active" data-filter="all"     onclick="setFilter('all')">TODO</button>
    <button class="btn"        data-filter="papers"  onclick="setFilter('papers')">PAPERS</button>
    <button class="btn"        data-filter="images"  onclick="setFilter('images')">IMAGES</button>
    <button class="btn"        data-filter="articles" onclick="setFilter('articles')">ARTICLES</button>
    <button class="btn"        data-filter="notes"   onclick="setFilter('notes')">NOTES</button>
  </div>

  <div class="items" id="items">
    <div class="empty">cargando...</div>
  </div>
</div>

<script>
let currentFilter = 'all';
let allItems = [];

function esc(t) { return String(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function loadStats() {
  try {
    const s = await (await fetch('/library/stats')).json();
    document.getElementById('s-items').textContent = s.total_items || 0;
    document.getElementById('s-mb').textContent    = (s.total_mb || 0).toFixed(1);
    document.getElementById('s-quota').textContent = s.quota_mb || 5000;
    const pct = s.pct_used || 0;
    document.getElementById('s-pct').textContent = pct.toFixed(0) + '%';
    const pctEl = document.getElementById('s-pct');
    pctEl.className = 'stat-num' + (pct > 85 ? ' crit' : pct > 65 ? ' warn' : '');
    const fill = document.getElementById('qfill');
    fill.style.width = Math.min(100, pct) + '%';
    fill.className = 'quota-fill' + (pct > 85 ? ' crit' : pct > 65 ? ' warn' : '');
  } catch(e) {}
}

async function loadSummary() {
  try {
    const s = await (await fetch('/library/summary')).json();
    const box = document.getElementById('summary');
    if (s.summary && s.summary.trim()) {
      box.textContent = s.summary;
    } else {
      box.innerHTML = '<span class="summary-empty">sin resumen aún — guarda items y se generará solo</span>';
    }
  } catch(e) {}
}

async function loadItems() {
  try {
    const d = await (await fetch('/library/list')).json();
    allItems = d.items || [];
    renderItems();
  } catch(e) {
    document.getElementById('items').innerHTML = '<div class="empty">error: ' + e.message + '</div>';
  }
}

function setFilter(f) {
  currentFilter = f;
  document.querySelectorAll('.filter-row .btn').forEach(b => {
    b.classList.toggle('active', b.dataset.filter === f);
  });
  renderItems();
}

function renderItems() {
  const filtered = currentFilter === 'all' ? allItems : allItems.filter(it => it.type === currentFilter);
  const box = document.getElementById('items');
  if (!filtered.length) {
    box.innerHTML = '<div class="empty">sin items' + (currentFilter !== 'all' ? ' de tipo ' + currentFilter : '') + '</div>';
    return;
  }
  box.innerHTML = filtered.map(it => {
    const sizeKB = (it.size_bytes / 1024).toFixed(0);
    const date   = it.ts ? it.ts.slice(0, 10) : '';
    const titleEsc = esc(it.title || '(sin título)');
    const titleHtml = it.source_url
      ? '<a href="' + esc(it.source_url) + '" target="_blank" style="color:inherit;text-decoration:none;">' + titleEsc + '</a>'
      : titleEsc;
    return '<div class="item">' +
      '<div class="acts">' +
        (it.type === 'images' || it.type === 'papers' || it.type === 'notes' || it.type === 'articles'
          ? '<button onclick="openFile(\\'' + it.id + '\\')" title="Abrir archivo">↗</button>' : '') +
        '<button onclick="delItem(\\'' + it.id + '\\')" title="Eliminar">×</button>' +
      '</div>' +
      '<div class="tp">' + esc(it.type) + '</div>' +
      '<div class="ttl">' + titleHtml + '</div>' +
      (it.why ? '<div class="why">' + esc(it.why) + '</div>' : '') +
      '<div class="meta">' +
        '<span>' + date + '</span>' +
        '<span>' + sizeKB + ' KB</span>' +
        (it.starred ? '<span style="color:#ffb84a">★</span>' : '') +
      '</div>' +
    '</div>';
  }).join('');
}

function openFile(id) {
  window.open('/library/item/' + id + '/file', '_blank');
}

async function delItem(id) {
  if (!confirm('Eliminar este item de la biblioteca?')) return;
  try {
    const r = await fetch('/library/item/' + id + '/delete', {method:'POST'});
    const d = await r.json();
    if (d.ok) { loadStats(); loadItems(); loadSummary(); }
  } catch(e) {}
}

async function regenSummary() {
  const box = document.getElementById('summary');
  box.innerHTML = '<span class="summary-empty">regenerando con LLM...</span>';
  try {
    const r = await fetch('/library/regenerate', {method:'POST'});
    const d = await r.json();
    box.textContent = d.summary || '(vacío)';
  } catch(e) {
    box.innerHTML = '<span class="summary-empty">error: ' + e.message + '</span>';
  }
}

loadStats();
loadItems();
loadSummary();
setInterval(() => { loadStats(); loadItems(); loadSummary(); }, 15000);
</script>
</body>
</html>"""


@app.route("/library")
def library_page():
    return render_template_string(LIBRARY_HTML)


if __name__ == "__main__":
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n  [Stella Dashboard]  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)
