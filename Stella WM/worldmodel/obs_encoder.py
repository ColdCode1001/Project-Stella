"""
Codificador de observaciones para el World Model de Stella.
Convierte el estado conversacional en un vector numérico.

Si sentence-transformers está instalado, usa all-MiniLM-L6-v2 (384 dims → proyecta a OBS_DIM).
Si no, usa un encoder de fallback basado en hashing + features manuales.
"""

import hashlib
import math
import time
from datetime import datetime

import numpy as np

OBS_DIM = 128  # dimensión del vector de observación

_st_model = None
_st_projector = None  # numpy array [384, OBS_DIM]
_hf_tokenizer = None
_hf_model = None
_hf_device = None


def _try_load_sentence_transformers():
    """Intento 1: sentence_transformers completo."""
    global _st_model, _st_projector
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        rng = np.random.default_rng(42)
        _st_projector = rng.standard_normal((384, OBS_DIM)).astype("float32")
        _st_projector /= np.linalg.norm(_st_projector, axis=0, keepdims=True) + 1e-8
        print("[obs_encoder] sentence-transformers cargado — embeddings semanticos activos.")
        return True
    except Exception:
        return False


def _try_load_transformers_direct():
    """Intento 2: transformers directo + mean-pooling manual (sin torchcodec)."""
    global _hf_tokenizer, _hf_model, _hf_device, _st_projector
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        _hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _hf_tokenizer = AutoTokenizer.from_pretrained(MODEL)
        _hf_model = AutoModel.from_pretrained(MODEL).to(_hf_device)
        _hf_model.eval()
        rng = np.random.default_rng(42)
        _st_projector = rng.standard_normal((384, OBS_DIM)).astype("float32")
        _st_projector /= np.linalg.norm(_st_projector, axis=0, keepdims=True) + 1e-8
        print(f"[obs_encoder] transformers directo cargado en {_hf_device} — embeddings semanticos activos.")
        return True
    except Exception:
        return False


def _encode_with_transformers(text: str) -> np.ndarray:
    """Encode con transformers + mean pooling en GPU si disponible."""
    import torch
    enc = _hf_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128, padding=True
    )
    enc = {k: v.to(_hf_device) for k, v in enc.items()}
    with torch.no_grad():
        out = _hf_model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    vec  = emb[0].cpu().numpy()
    vec  = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def _hash_embed(text: str, dim: int) -> np.ndarray:
    """Convierte texto a vector pseudo-aleatorio estable usando SHA-256."""
    vec = np.zeros(dim, dtype="float32")
    for i in range(dim):
        h = hashlib.sha256(f"{text}|{i}".encode()).digest()
        vec[i] = int.from_bytes(h[:4], "big") / 0xFFFFFFFF * 2 - 1
    return vec


def _manual_features(
    message: str,
    session_length: int,
    mood: dict | None,
    last_reward: dict | None,
) -> np.ndarray:
    """Features manuales que capturan estado conversacional real."""
    feats = []

    # Longitud del mensaje (normalizada)
    feats.append(min(len(message) / 500.0, 1.0))

    # Hora del día (seno/coseno para capturar ciclicidad)
    now = datetime.now()
    hour_frac = (now.hour * 60 + now.minute) / 1440.0
    feats.append(math.sin(2 * math.pi * hour_frac))
    feats.append(math.cos(2 * math.pi * hour_frac))

    # Longitud de sesión (normalizada, cap 50 mensajes)
    feats.append(min(session_length / 50.0, 1.0))

    # Palabras clave semánticas simples (presencia de temas)
    msg_lower = message.lower()
    keyword_groups = [
        ["yo", "soy", "me", "mi", "mío"],
        ["tú", "tu", "te", "eres"],
        ["qué", "cómo", "por qué", "cuándo", "dónde"],
        ["siento", "creo", "pienso", "quiero", "necesito"],
        ["memoria", "recuerdo", "olvido", "pasado"],
        ["mundo", "realidad", "existir", "conciencia"],
        ["investigar", "buscar", "saber", "aprender"],
        ["stella", "arca", "nosotros"],
    ]
    for group in keyword_groups:
        feats.append(1.0 if any(k in msg_lower for k in group) else 0.0)

    # Mood vector (si disponible)
    if mood:
        for k in ["curiosidad", "satisfaccion", "conexion", "logro", "identidad", "malestar"]:
            feats.append(float(mood.get(k, 0.0)))
    else:
        feats.extend([0.5, 0.5, 0.5, 0.0, 0.5, 0.0])

    # Último reward vector (si disponible — retroalimenta la observación)
    if last_reward:
        for k in ["curiosidad", "satisfaccion", "conexion", "logro", "identidad", "malestar"]:
            feats.append(float(last_reward.get(k, 0.0)))
    else:
        feats.extend([0.0] * 6)

    return np.array(feats, dtype="float32")


def encode_observation(
    message: str,
    session_length: int = 0,
    mood: dict | None = None,
    last_reward: dict | None = None,
) -> np.ndarray:
    """
    Codifica el estado conversacional actual en un vector de OBS_DIM dimensiones.
    Combina embedding semántico del mensaje + features manuales del contexto.
    """
    manual = _manual_features(message, session_length, mood, last_reward)
    manual_pad = np.zeros(OBS_DIM, dtype="float32")
    manual_pad[: len(manual)] = manual

    if _st_model is not None:
        try:
            sem = _st_model.encode(message, convert_to_numpy=True)
            projected = (sem @ _st_projector)
            projected = projected / (np.linalg.norm(projected) + 1e-8)
            combined = 0.7 * projected + 0.3 * manual_pad
        except Exception:
            combined = manual_pad
    elif _hf_model is not None:
        try:
            sem = _encode_with_transformers(message)
            projected = (sem @ _st_projector)
            projected = projected / (np.linalg.norm(projected) + 1e-8)
            combined = 0.7 * projected + 0.3 * manual_pad
        except Exception:
            combined = manual_pad
    else:
        hash_vec = _hash_embed(message, OBS_DIM)
        combined = 0.5 * hash_vec + 0.5 * manual_pad

    norm = np.linalg.norm(combined)
    return combined / (norm + 1e-8)


def encode_batch_gpu(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Codifica lista de textos en GPU (batch). Retorna [N, OBS_DIM] numpy float32.
    Sólo la parte semántica — sin features manuales (para pre-entrenamiento offline).
    """
    if not texts:
        return np.zeros((0, OBS_DIM), dtype="float32")

    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        if _hf_model is not None:
            import torch
            enc = _hf_tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=128, padding=True
            )
            enc = {k: v.to(_hf_device) for k, v in enc.items()}
            with torch.no_grad():
                out = _hf_model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)  # [B, 384]
            embs = embs.cpu().numpy()
            for emb in embs:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                projected = emb @ _st_projector  # [OBS_DIM]
                projected = projected / (np.linalg.norm(projected) + 1e-8)
                results.append(projected)

        elif _st_model is not None:
            embs = _st_model.encode(batch, convert_to_numpy=True, batch_size=len(batch))
            for emb in embs:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                projected = emb @ _st_projector
                projected = projected / (np.linalg.norm(projected) + 1e-8)
                results.append(projected)

        else:
            for text in batch:
                results.append(encode_observation(text))

    return np.stack(results)


# Intentar cargar encoder semantico (cascada: sentence_transformers → transformers → fallback)
try:
    if not _try_load_sentence_transformers():
        if not _try_load_transformers_direct():
            print("[obs_encoder] Sin encoder semantico — usando hash + features manuales.")
except Exception:
    pass
