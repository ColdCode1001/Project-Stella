"""
Minimal RSSM (Recurrent State Space Model) para el World Model de Stella.

Arquitectura (basada en R2-Dreamer pero reducida para espacio conversacional):
  observacion → GRU (estado determinístico h_t) → z_t (latente estocástico)
  feat = [h_t, z_t] → 6 reward heads + actor (7 acciones)

Dimensiones por defecto:
  obs_dim    = 128   (output de obs_encoder)
  hidden_dim = 256   (estado determinístico del GRU)
  latent_dim = 64    (estado estocástico z)
  n_actions  = 7

Parámetros totales: ~350K — corre en CPU en <5ms por step.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = [
    "responder_chat",
    "buscar_web",
    "guardar_episodio",
    "avanzar_quest",
    "ejecutar_experimento",
    "guardar_nota",
    "idle",
]

REWARD_DIMS = ["curiosidad", "satisfaccion", "conexion", "logro", "identidad", "malestar"]


class MinimalRSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        n_actions: int = 7,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        # Estado determinístico
        self.gru = nn.GRUCell(obs_dim + n_actions, hidden_dim)

        # Estado estocástico — posterior (con observación)
        self.post_mean = nn.Linear(hidden_dim + obs_dim, latent_dim)
        self.post_log_std = nn.Linear(hidden_dim + obs_dim, latent_dim)

        # Estado estocástico — prior (sin observación, para imaginación futura)
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_log_std = nn.Linear(hidden_dim, latent_dim)

        feat_dim = hidden_dim + latent_dim  # 320

        # 6 reward heads independientes
        def _head(out_activation=None):
            layers = [nn.Linear(feat_dim, 64), nn.ELU(), nn.Linear(64, 1)]
            if out_activation:
                layers.append(out_activation)
            return nn.Sequential(*layers)

        self.reward_heads = nn.ModuleDict({
            "curiosidad":   _head(nn.Sigmoid()),
            "satisfaccion": _head(nn.Sigmoid()),
            "conexion":     _head(nn.Sigmoid()),
            "logro":        _head(nn.Sigmoid()),
            "identidad":    _head(nn.Sigmoid()),
            "malestar":     _head(nn.Tanh()),   # [-1, 1], solo se usa la parte negativa
        })

        # Actor — política en espacio latente
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ELU(),
            nn.Linear(128, n_actions),
        )

    def zero_state(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_dim)

    def _sample_z(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std.clamp(-4.0, 2.0))
        return mean + std * torch.randn_like(mean)

    @torch.no_grad()
    def step(
        self,
        obs_vec: np.ndarray,
        action_idx: int = 0,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, torch.Tensor, dict, list[float], int]:
        """
        Un paso del World Model.

        Args:
            obs_vec:    vector de observación [obs_dim] (numpy float32)
            action_idx: índice de la acción anterior (0-6)
            h_prev:     estado oculto GRU anterior [1, hidden_dim] o None

        Returns:
            z_np:       vector latente [latent_dim] como numpy
            h_new:      nuevo estado GRU [1, hidden_dim] (para el siguiente step)
            rewards:    dict {nombre: float} con los 6 valores de recompensa
            action_probs: list[float] de 7 probabilidades
            action_sel: índice de la acción seleccionada
        """
        self.eval()

        if h_prev is None:
            h_prev = torch.zeros(1, self.hidden_dim)

        obs = torch.from_numpy(obs_vec).float().unsqueeze(0)           # [1, obs_dim]
        act = F.one_hot(torch.tensor([action_idx]), self.n_actions).float()  # [1, n_actions]

        # GRU step determinístico
        gru_in = torch.cat([obs, act], dim=-1)  # [1, obs_dim + n_actions]
        h_new = self.gru(gru_in, h_prev)        # [1, hidden_dim]

        # Posterior z (con la observación actual)
        post_in = torch.cat([h_new, obs], dim=-1)
        z = self._sample_z(self.post_mean(post_in), self.post_log_std(post_in))  # [1, latent_dim]

        feat = torch.cat([h_new, z], dim=-1)  # [1, feat_dim]

        # Reward predictions
        rewards = {}
        for name, head in self.reward_heads.items():
            val = head(feat).squeeze().item()
            if name == "malestar":
                val = min(0.0, val)  # malestar siempre ≤ 0
            rewards[name] = round(float(np.clip(val, -1.0, 1.0)), 3)

        # Selección de acción (greedy sobre logits)
        logits = self.actor(feat).squeeze()           # [n_actions]
        probs = F.softmax(logits, dim=-1).tolist()
        action_sel = int(torch.argmax(logits).item())

        return z.squeeze().numpy(), h_new.detach(), rewards, probs, action_sel

    def predict_next_z(self, h: torch.Tensor) -> np.ndarray:
        """Predicción del prior: z_mean esperado dado h (sin observación)."""
        with torch.no_grad():
            return self.prior_mean(h).squeeze().numpy()

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path):
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


class LargeRSSM(nn.Module):
    """
    RSSM de alta capacidad ~125M parámetros.

    El cerebro de Stella en serio.

    Dimensiones:
      obs_dim    = 128   (encoder)
      hidden_dim = 2048  (GRU 3 capas apiladas)
      z_dim      = 512   (latente estocástico)
      feat_dim   = 2560  (8× el MinimalRSSM)

    El ctx_vec [2560D] es tan rico que el decoder no tiene que inventar nada —
    solo articular lo que el WM ya determinó completamente.

    Interfaz compatible con MinimalRSSM.step() excepto:
      h_prev: Tensor [3, 1, hidden_dim]  (3 capas de GRU)
      h_new:  Tensor [3, 1, hidden_dim]
    """

    HIDDEN  = 2048
    Z_DIM   = 512
    N_LAYERS = 3
    MLP_H   = 4096

    def __init__(
        self,
        obs_dim:    int = 128,
        hidden_dim: int = 2048,
        z_dim:      int = 512,
        n_gru_layers: int = 3,
        mlp_hidden: int = 4096,
        n_actions:  int = 7,
    ):
        super().__init__()
        self.obs_dim     = obs_dim
        self.hidden_dim  = hidden_dim
        self.z_dim       = z_dim
        self.n_gru_layers = n_gru_layers
        self.n_actions   = n_actions
        self.feat_dim    = hidden_dim + z_dim  # 2560

        # Secuencia determinística: GRU apilado
        self.gru = nn.GRU(
            input_size  = obs_dim + n_actions,
            hidden_size = hidden_dim,
            num_layers  = n_gru_layers,
            batch_first = True,
        )

        # Prior: p(z_t | h_t) — imaginación sin observación
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, z_dim * 2),  # → mean + log_std
        )

        # Posterior: q(z_t | h_t, obs_t) — estado real
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, z_dim * 2),  # → mean + log_std
        )

        # Reconstrucción de observación (loss auxiliar en training)
        self.obs_pred = nn.Linear(self.feat_dim, obs_dim)

        feat = self.feat_dim

        def _reward_head():
            return nn.Sequential(
                nn.Linear(feat, 256), nn.ELU(),
                nn.Linear(256, 64),   nn.ELU(),
                nn.Linear(64,  1),
            )

        self.reward_heads = nn.ModuleDict({
            "curiosidad":   nn.Sequential(*_reward_head(), nn.Sigmoid()),
            "satisfaccion": nn.Sequential(*_reward_head(), nn.Sigmoid()),
            "conexion":     nn.Sequential(*_reward_head(), nn.Sigmoid()),
            "logro":        nn.Sequential(*_reward_head(), nn.Sigmoid()),
            "identidad":    nn.Sequential(*_reward_head(), nn.Sigmoid()),
            "malestar":     nn.Sequential(*_reward_head(), nn.Tanh()),
        })

        self.actor = nn.Sequential(
            nn.Linear(feat, 512), nn.ELU(),
            nn.Linear(512, 128), nn.ELU(),
            nn.Linear(128, n_actions),
        )

    def zero_state(self) -> torch.Tensor:
        """Estado inicial: [n_layers, 1, hidden_dim]."""
        return torch.zeros(self.n_gru_layers, 1, self.hidden_dim)

    def _sample_z(self, params: torch.Tensor) -> torch.Tensor:
        mean, log_std = params.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-4.0, 2.0))
        return mean + std * torch.randn_like(mean)

    @torch.no_grad()
    def step(
        self,
        obs_vec:    np.ndarray,
        action_idx: int = 0,
        h_prev:     torch.Tensor | None = None,
    ) -> tuple[np.ndarray, torch.Tensor, dict, list[float], int]:
        """
        Mismo contrato que MinimalRSSM.step().
        h_prev: [n_layers, 1, hidden_dim]  (o None → zeros)
        h_new:  [n_layers, 1, hidden_dim]
        """
        self.eval()

        if h_prev is None:
            h_prev = self.zero_state()

        obs = torch.from_numpy(obs_vec).float().unsqueeze(0).unsqueeze(0)  # [1, 1, obs_dim]
        act = F.one_hot(torch.tensor([action_idx]), self.n_actions).float().unsqueeze(0)  # [1, 1, n_actions]

        gru_in = torch.cat([obs, act], dim=-1)          # [1, 1, obs_dim+n_actions]
        _, h_new = self.gru(gru_in, h_prev)             # h_new: [n_layers, 1, hidden_dim]

        h_top = h_new[-1]                               # capa superior: [1, hidden_dim]
        obs_flat = obs.squeeze(0)                       # [1, obs_dim]

        post_in = torch.cat([h_top, obs_flat], dim=-1)
        z = self._sample_z(self.posterior(post_in))     # [1, z_dim]

        feat = torch.cat([h_top, z], dim=-1)            # [1, feat_dim=2560]

        rewards = {}
        for name, head in self.reward_heads.items():
            val = head(feat).squeeze().item()
            if name == "malestar":
                val = min(0.0, val)
            rewards[name] = round(float(np.clip(val, -1.0, 1.0)), 3)

        logits    = self.actor(feat).squeeze()
        probs     = F.softmax(logits, dim=-1).tolist()
        action_sel = int(torch.argmax(logits).item())

        return z.squeeze().numpy(), h_new.detach(), rewards, probs, action_sel

    def predict_next_z(self, h: torch.Tensor) -> np.ndarray:
        """Predicción del prior: z_mean esperado dado h (sin observación)."""
        with torch.no_grad():
            h_top = h[-1] if h.dim() == 3 else h
            out = self.prior(h_top)
            return out[:, :self.z_dim].squeeze().numpy()

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path):
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_or_create(path: str | Path = "worldmodel/weights/rssm.pt", **kwargs) -> MinimalRSSM:
    """Carga pesos existentes o crea modelo nuevo con pesos aleatorios."""
    model = MinimalRSSM(**kwargs)
    p = Path(path)
    if p.exists():
        try:
            model.load(p)
            print(f"[RSSM] Pesos cargados desde {p}")
        except Exception as e:
            print(f"[RSSM] Error cargando pesos ({e}), usando pesos aleatorios.")
    else:
        print(f"[RSSM] Sin pesos previos en {p} — iniciando con pesos aleatorios.")
    return model
