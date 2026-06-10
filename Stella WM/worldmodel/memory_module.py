"""
Módulo de memoria vectorial para el World Model de Stella.
Funciona como el hipocampo: almacena experiencias como vectores,
las recupera por similitud semántica, olvida lo que no se usa.

Principio: el conocimiento vive AQUÍ, no en los pesos del RSSM.
  - Los pesos del RSSM = cómo procesar el mundo (procedural)
  - Esta memoria = qué ha vivido esta instancia (experiencial)

Integración con RSSM:
  obs_enriched = obs_vec + MEMORY_WEIGHT * memory_context
  El RSSM recibe obs_enriched → h_t ya lleva contexto de memoria
  Sin cambios de arquitectura. Sin reentrenamiento.
"""

import json
import time
from pathlib import Path

import numpy as np

MEMORY_PATH   = Path("worldmodel/weights/memory.npz")
MEMORY_WEIGHT = 0.3   # cuánto influye la memoria en obs_vec
MAX_MEMORIES  = 2000  # límite antes de consolidación


class MemoryStore:
    def __init__(self, path: Path = MEMORY_PATH):
        self.path = Path(path)
        self.embeddings:  np.ndarray = np.zeros((0, 128), dtype="float32")
        self.texts:       list[str]  = []
        self.timestamps:  list[float]= []
        self.importance:  list[float]= []
        self.access_count:list[int]  = []

        if self.path.exists():
            self._load()
            print(f"[memory] Cargada: {len(self.texts)} recuerdos desde {self.path}")
        else:
            print(f"[memory] Memoria nueva — sin recuerdos previos.")

    # ── Escritura ──────────────────────────────────────────────

    def add(self, embedding: np.ndarray, text: str, importance: float = 1.0):
        """Almacena una nueva experiencia. Si es muy similar a una existente, la refuerza."""
        emb = self._normalize(embedding.flatten()[:128])

        if len(self.texts) > 0:
            sims = self.embeddings @ emb
            best_idx = int(np.argmax(sims))
            if sims[best_idx] > 0.92:
                # Recuerdo muy similar ya existe — reforzar importancia
                self.importance[best_idx] = min(2.0, self.importance[best_idx] + 0.2)
                self.access_count[best_idx] += 1
                return

        self.embeddings = (
            np.vstack([self.embeddings, emb[np.newaxis, :]])
            if len(self.texts) > 0
            else emb[np.newaxis, :]
        )
        self.texts.append(text[:500])
        self.timestamps.append(time.time())
        self.importance.append(float(importance))
        self.access_count.append(0)

        if len(self.texts) > MAX_MEMORIES:
            self._consolidate()

        self.save()

    # ── Recuperación ───────────────────────────────────────────

    def query(self, embedding: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Recupera los k recuerdos más relevantes.
        Retorna array [k, 128D] — vectores puros, no texto.
        El RSSM los consume directamente como contexto.
        """
        if len(self.texts) == 0:
            return np.zeros((k, 128), dtype="float32")

        emb = self._normalize(embedding.flatten()[:128])
        sims = self.embeddings @ emb  # [N]

        # Penalizar recuerdos muy viejos (decay temporal)
        now = time.time()
        ages = np.array([(now - ts) / 86400 for ts in self.timestamps])  # días
        decay = np.exp(-0.05 * ages)  # mitad de relevancia a los ~14 días
        scores = sims * decay * np.array(self.importance)

        top_k = min(k, len(self.texts))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        # Registrar acceso (refuerza la memoria usada)
        for idx in top_idx:
            self.access_count[idx] += 1

        return self.embeddings[top_idx]  # [k, 128D]

    def enrich_observation(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        Enriquece el vector de observación con contexto de memoria.
        Llamar ANTES del paso RSSM. Sin cambios de arquitectura.

        obs_enriched = obs_vec + MEMORY_WEIGHT * mean(top-5 memorias)
        """
        if len(self.texts) == 0:
            return obs_vec

        mem_vecs = self.query(obs_vec, k=5)          # [5, 128D]
        mem_ctx  = mem_vecs.mean(axis=0)              # [128D]
        enriched = obs_vec + MEMORY_WEIGHT * mem_ctx
        norm = np.linalg.norm(enriched)
        return enriched / (norm + 1e-8)

    # ── Olvido / consolidación ─────────────────────────────────

    def forget(self, max_age_days: float = 60.0, min_importance: float = 0.1):
        """Elimina recuerdos viejos e irrelevantes. Llama periódicamente."""
        if len(self.texts) == 0:
            return 0

        now = time.time()
        keep = []
        for i, (ts, imp, acc) in enumerate(
            zip(self.timestamps, self.importance, self.access_count)
        ):
            age_days = (now - ts) / 86400
            # Mantener si: reciente O importante O muy accedido
            if age_days < max_age_days or imp >= min_importance or acc >= 3:
                keep.append(i)

        removed = len(self.texts) - len(keep)
        if removed > 0:
            self.embeddings   = self.embeddings[keep]
            self.texts        = [self.texts[i]        for i in keep]
            self.timestamps   = [self.timestamps[i]   for i in keep]
            self.importance   = [self.importance[i]   for i in keep]
            self.access_count = [self.access_count[i] for i in keep]
            self.save()
            print(f"[memory] Olvidados {removed} recuerdos. Quedan {len(self.texts)}.")

        return removed

    def _consolidate(self):
        """Cuando la memoria está llena, elimina los menos importantes."""
        scores = np.array(self.importance) * (np.array(self.access_count) + 1)
        keep_n = int(MAX_MEMORIES * 0.8)
        keep = np.argpartition(scores, -keep_n)[-keep_n:].tolist()
        keep.sort()

        self.embeddings   = self.embeddings[keep]
        self.texts        = [self.texts[i]        for i in keep]
        self.timestamps   = [self.timestamps[i]   for i in keep]
        self.importance   = [self.importance[i]   for i in keep]
        self.access_count = [self.access_count[i] for i in keep]

    # ── Persistencia ───────────────────────────────────────────

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            embeddings   = self.embeddings,
            timestamps   = np.array(self.timestamps,   dtype="float64"),
            importance   = np.array(self.importance,   dtype="float32"),
            access_count = np.array(self.access_count, dtype="int32"),
        )
        # Textos por separado (npz no maneja strings bien)
        texts_path = self.path.with_suffix(".texts.json")
        texts_path.write_text(json.dumps(self.texts, ensure_ascii=False), encoding="utf-8")

    def _load(self):
        data = np.load(self.path)
        self.embeddings   = data["embeddings"].astype("float32")
        self.timestamps   = data["timestamps"].tolist()
        self.importance   = data["importance"].tolist()
        self.access_count = data["access_count"].tolist()

        texts_path = self.path.with_suffix(".texts.json")
        if texts_path.exists():
            self.texts = json.loads(texts_path.read_text(encoding="utf-8"))
        else:
            self.texts = [""] * len(self.timestamps)

    # ── Info ───────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self.texts:
            return {"total": 0}
        now = time.time()
        ages = [(now - ts) / 86400 for ts in self.timestamps]
        return {
            "total":        len(self.texts),
            "age_avg_days": round(sum(ages) / len(ages), 1),
            "age_max_days": round(max(ages), 1),
            "importance_avg": round(sum(self.importance) / len(self.importance), 2),
            "most_accessed": sorted(
                zip(self.access_count, self.texts), reverse=True
            )[:3],
        }

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(vec)
        return vec / (n + 1e-8)


# Instancia global (cargada una vez al importar)
_store: MemoryStore | None = None


def get_memory() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
