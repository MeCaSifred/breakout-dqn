"""
Dueling DQN + PER + Double DQN para ALE/Breakout-v5
Versión definitiva — incluye:
  - Arquitectura Dueling DQN (Value stream + Advantage stream)
  - Prioritized Experience Replay con SumTree corregido
  - Double DQN para reducir sobreestimación de valores Q

Uso:
    python dqn_breakout.py                    # entrenamiento completo
    python dqn_breakout.py --steps 5000000   # entrenar hasta 5M steps
    python dqn_breakout.py --eval --render   # ver agente jugar
"""

import argparse
import os
import random
import time
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gym.register_envs(ale_py)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
#  HIPERPARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════
HP = {
    "env_name"           : "ALE/Breakout-v5",
    "frame_skip"         : 4,
    "frame_stack"        : 4,
    "frame_size"         : 84,
    "noop_max"           : 30,
    "total_steps"        : 1_500_000,
    "learning_starts"    : 10_000,
    "train_frequency"    : 4,
    "target_update_freq" : 1_000,
    "batch_size"         : 32,
    "buffer_size"        : 100_000,
    "gamma"              : 0.99,
    "lr"                 : 1e-4,
    "eps_start"          : 1.0,
    "eps_end"            : 0.01,
    "eps_decay_steps"    : 300_000,
    "max_episode_steps"  : 2_000,
    "per_alpha"          : 0.6,
    "per_beta_start"     : 0.4,
    "per_beta_end"       : 1.0,
    "per_eps"            : 1e-6,
    "log_every"          : 10,       # imprimir cada N episodios
    "save_every"         : 50_000,
}


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESAMIENTO — igual a versiones anteriores
# ══════════════════════════════════════════════════════════════════════════════

class NoopResetWrapper(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(np.random.randint(1, self.noop_max + 1)):
            obs, _, term, trunc, info = self.env.step(0)
            if term or trunc:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(1)
        if term or trunc:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxFrameWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._buf = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        for i in range(2):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            self._buf[i] = obs
            if term or trunc:
                break
        return self._buf.max(axis=0), total_reward, term, trunc, info


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if term or trunc:
                break
        return obs, total_reward, term, trunc, info


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, size=84):
        super().__init__(env)
        self._size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self._size, self._size),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class FrameStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self._n = n_frames
        self._buf = deque(maxlen=n_frames)
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n_frames, h, w), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._n):
            self._buf.append(obs[:, :, 0])
        return np.array(self._buf, dtype=np.uint8), info

    def observation(self, obs):
        self._buf.append(obs[:, :, 0])
        return np.array(self._buf, dtype=np.uint8)


def make_env(render_mode=None):
    env = gym.make(HP["env_name"], frameskip=1,
                   repeat_action_probability=0.0, render_mode=render_mode)
    env = NoopResetWrapper(env, HP["noop_max"])
    env = FireResetWrapper(env)
    env = MaxFrameWrapper(env)
    env = FrameSkipWrapper(env, HP["frame_skip"])
    env = GrayscaleResizeWrapper(env, HP["frame_size"])
    env = FrameStackWrapper(env, HP["frame_stack"])
    return env


# ══════════════════════════════════════════════════════════════════════════════
#  RED NEURONAL — Dueling DQN
# ══════════════════════════════════════════════════════════════════════════════

class DuelingQNetwork(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4,  32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv(x).view(x.size(0), -1)
        V = self.value_stream(f)
        A = self.advantage_stream(f)
        return V + (A - A.mean(dim=1, keepdim=True))

    @staticmethod
    def preprocess(obs, device):
        return torch.FloatTensor(obs).unsqueeze(0).to(device) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
#  PRIORITIZED REPLAY BUFFER — versión corregida
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """
    Árbol binario de suma para muestreo eficiente por prioridad.

    Estructura (capacity=4):
                 [0]  ← nodo raíz = suma total de prioridades
                /   \\
             [1]     [2]  ← nodos internos
            /   \\   /   \\
          [3]  [4] [5]  [6]  ← hojas = prioridades individuales

    Las experiencias se guardan en las hojas.
    Para muestrear: se divide la suma total en N segmentos y
    se recorre el árbol de arriba a abajo para encontrar
    qué hoja corresponde a cada valor muestreado.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.ptr      = 0   # puntero circular a la siguiente hoja libre

    def _propagate(self, idx, delta):
        """Propaga el cambio de prioridad hacia la raíz."""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx, priority):
        """Actualiza la prioridad de la hoja idx."""
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority):
        """Agrega una nueva prioridad y devuelve el índice de hoja en el árbol."""
        tree_idx  = self.ptr + self.capacity - 1
        self.update(tree_idx, priority)
        leaf_ptr  = self.ptr
        self.ptr  = (self.ptr + 1) % self.capacity
        return leaf_ptr, tree_idx

    def get(self, value):
        """
        Recorre el árbol de arriba a abajo.
        Devuelve (leaf_data_idx, tree_idx, priority).
        """
        idx = 0
        while idx < self.capacity - 1:
            left  = 2 * idx + 1
            right = 2 * idx + 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx    = right
        data_idx = idx - (self.capacity - 1)
        return data_idx, idx, self.tree[idx]

    @property
    def total(self):
        return self.tree[0]

    @property
    def max_priority(self):
        return self.tree[self.capacity - 1:].max()


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha    = alpha
        self.tree     = SumTree(capacity)
        self.size     = 0

        self.states      = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        # Nueva experiencia recibe la máxima prioridad para asegurar
        # que se muestree al menos una vez
        max_p = self.tree.max_priority
        priority = (max_p if max_p > 0 else 1.0) ** self.alpha

        data_idx, _ = self.tree.add(priority)

        self.states[data_idx]      = state
        self.next_states[data_idx] = next_state
        self.actions[data_idx]     = action
        self.rewards[data_idx]     = np.clip(reward, -1, 1)
        self.dones[data_idx]       = float(done)
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta, device):
        data_idxs  = np.zeros(batch_size, dtype=np.int32)
        tree_idxs  = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = random.uniform(a, b)
            data_idx, tree_idx, priority = self.tree.get(v)
            data_idxs[i]  = data_idx
            tree_idxs[i]  = tree_idx
            priorities[i] = priority

        # Importance sampling weights
        probs   = priorities / (self.tree.total + 1e-8)
        weights = (self.size * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return (
            torch.FloatTensor(self.states[data_idxs]).to(device)      / 255.0,
            torch.LongTensor(self.actions[data_idxs]).to(device),
            torch.FloatTensor(self.rewards[data_idxs]).to(device),
            torch.FloatTensor(self.next_states[data_idxs]).to(device) / 255.0,
            torch.FloatTensor(self.dones[data_idxs]).to(device),
            torch.FloatTensor(weights).to(device),
            tree_idxs,
        )

    def update_priorities(self, tree_idxs, td_errors):
        for idx, err in zip(tree_idxs, td_errors):
            p = (float(abs(err)) + HP["per_eps"]) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════════════════════════
#  AGENTE
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgentV3:
    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_steps = 0

        self.q_net      = DuelingQNetwork(n_actions).to(self.device)
        self.target_net = DuelingQNetwork(n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=HP["lr"])
        self.buffer    = PrioritizedReplayBuffer(HP["buffer_size"], HP["per_alpha"])

        n_params = sum(p.numel() for p in self.q_net.parameters())
        print(f"\n  Dispositivo : {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  Parámetros  : {n_params:,}")
        print(f"  Arquitectura: Dueling DQN + PER + Double DQN")

    def epsilon(self):
        p = min(self.total_steps / HP["eps_decay_steps"], 1.0)
        return HP["eps_start"] + p * (HP["eps_end"] - HP["eps_start"])

    def beta(self):
        p = min(self.total_steps / HP["total_steps"], 1.0)
        return HP["per_beta_start"] + p * (HP["per_beta_end"] - HP["per_beta_start"])

    def select_action(self, state):
        if random.random() < self.epsilon():
            return random.randint(0, self.n_actions - 1)
        s = DuelingQNetwork.preprocess(state, self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax().item())

    def optimize(self):
        if len(self.buffer) < HP["batch_size"]:
            return None
        states, actions, rewards, next_states, dones, weights, tree_idxs = \
            self.buffer.sample(HP["batch_size"], self.beta(), self.device)

        q_curr = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.q_net(next_states).argmax(1)
            q_next = self.target_net(next_states).gather(
                        1, best_a.unsqueeze(1)).squeeze(1)
            target = rewards + HP["gamma"] * q_next * (1 - dones)

        td_errors = (q_curr - target).detach().abs().cpu().numpy()
        self.buffer.update_priorities(tree_idxs, td_errors)

        loss = (weights * nn.HuberLoss(reduction='none')(q_curr, target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_net"      : self.q_net.state_dict(),
            "optimizer"  : self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        print(f"  ✅ Checkpoint cargado (step {self.total_steps:,})")


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING MEJORADO
# ══════════════════════════════════════════════════════════════════════════════

def progress_bar(current, total, width=30):
    """Barra de progreso visual: [████████░░░░░░░░] 45%"""
    pct   = current / total
    filled = int(width * pct)
    bar   = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:.1f}%"


def print_header():
    print(f"\n{'='*75}")
    print(f"{'Campo':<18} {'Descripción'}")
    print(f"{'─'*75}")
    print(f"{'Ep':<18} Número de episodio")
    print(f"{'Step':<18} Pasos totales acumulados")
    print(f"{'Avg(20)':<18} Recompensa PROMEDIO últimos 20 episodios")
    print(f"{'Max(20)':<18} Recompensa MÁXIMA últimos 20 episodios")
    print(f"{'Min(20)':<18} Recompensa MÍNIMA últimos 20 episodios")
    print(f"{'BestEver':<18} Mejor recompensa en toda la historia")
    print(f"{'ε (epsilon)':<18} Probabilidad de acción aleatoria")
    print(f"{'β (beta)':<18} Corrección de sesgo PER (0.4→1.0)")
    print(f"{'Buffer':<18} Experiencias en replay buffer")
    print(f"{'Steps/s':<18} Velocidad de entrenamiento")
    print(f"{'ETA':<18} Tiempo estimado restante")
    print(f"{'='*75}\n")
    # Cabecera de la tabla de logs
    print(f"{'Ep':>6} {'Step':>10} │ "
          f"{'Avg(20)':>8} {'Max(20)':>8} {'Min(20)':>7} {'BestEver':>9} │ "
          f"{'ε':>6} {'β':>5} │ "
          f"{'Buffer':>7} {'Steps/s':>8} {'ETA':>7}")
    print("─" * 100)


def print_log(ep, step, rewards, best_ever, epsilon, beta, buf_size, sps, eta_h):
    """Imprime una línea de log con todas las métricas."""
    window  = rewards[-20:] if len(rewards) >= 20 else rewards
    avg     = np.mean(window)
    mx      = np.max(window)
    mn      = np.min(window)

    # Color en terminal: verde si avg mejora, rojo si baja
    trend = "↑" if len(rewards) > 40 and avg > np.mean(rewards[-40:-20]) else "↓"

    print(f"{ep:>6} {step:>10,} │ "
          f"{avg:>8.1f} {mx:>8.1f} {mn:>7.1f} {best_ever:>9.1f} │ "
          f"{epsilon:>6.3f} {beta:>5.2f} │ "
          f"{buf_size:>7,} {sps:>8.0f} {eta_h:>6.1f}h  {trend}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def train(total_steps=HP["total_steps"]):
    env   = make_env()
    agent = DQNAgentV3(n_actions=env.action_space.n)

    ckpt = "checkpoints/dqn_breakout_latest.pth"
    if os.path.exists(ckpt):
        agent.load(ckpt)

    # ── Historial completo ─────────────────────────────────────────────────
    ep_rewards   = []   # recompensa por episodio
    ep_max_rwds  = []   # máximo por ventana
    ep_avg_rwds  = []   # promedio por ventana
    ep_lengths   = []   # duración por episodio
    ep_epsilons  = []
    ep_betas     = []
    losses_log   = []
    best_ever    = 0.0  # mejor recompensa individual histórica

    ep_reward, ep_steps = 0.0, 0
    ep_num = 0
    state, _ = env.reset(seed=SEED)
    t_start  = time.time()

    print_header()

    for step in range(agent.total_steps, total_steps):
        agent.total_steps = step

        action = agent.select_action(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        agent.buffer.push(state, action, reward, next_state, float(done))
        state      = next_state
        ep_reward += reward
        ep_steps  += 1

        if step >= HP["learning_starts"] and step % HP["train_frequency"] == 0:
            loss = agent.optimize()
            if loss is not None:
                losses_log.append(loss)

        if step % HP["target_update_freq"] == 0:
            agent.update_target()

        if step > 0 and step % HP["save_every"] == 0:
            agent.save(ckpt)
            agent.save(f"checkpoints/dqn_breakout_{step//1000}k.pth")
            # Mostrar barra de progreso al guardar
            print(f"\n  💾 Guardado | {progress_bar(step, total_steps)}\n")

        if done or ep_steps >= HP["max_episode_steps"]:
            ep_num += 1
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_steps)
            ep_epsilons.append(agent.epsilon())
            ep_betas.append(agent.beta())

            if ep_reward > best_ever:
                best_ever = ep_reward

            # Calcular métricas de ventana
            window = ep_rewards[-20:] if len(ep_rewards) >= 20 else ep_rewards
            ep_avg_rwds.append(float(np.mean(window)))
            ep_max_rwds.append(float(np.max(window)))

            if ep_num % HP["log_every"] == 0:
                elapsed = time.time() - t_start
                sps     = step / max(elapsed, 1)
                eta_h   = (total_steps - step) / max(sps, 1) / 3600
                print_log(ep_num, step, ep_rewards, best_ever,
                          agent.epsilon(), agent.beta(),
                          len(agent.buffer), sps, eta_h)

            ep_reward, ep_steps = 0.0, 0
            state, _ = env.reset()

    env.close()
    agent.save(ckpt)

    print(f"\n{'='*75}")
    print(f"  ENTRENAMIENTO COMPLETADO")
    print(f"  Mejor recompensa histórica : {best_ever:.1f}")
    print(f"  Promedio últimos 50 eps    : {np.mean(ep_rewards[-50:]):.1f}")
    print(f"  Tiempo total               : {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*75}\n")

    _plot_results(ep_rewards, ep_avg_rwds, ep_max_rwds,
                  ep_lengths, ep_epsilons, ep_betas, losses_log, best_ever)
    return agent, ep_rewards


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(agent=None, n_episodes=10, render=False):
    env = make_env(render_mode="human" if render else None)

    if agent is None:
        agent = DQNAgentV3(n_actions=env.action_space.n)
        agent.load("checkpoints/dqn_breakout_latest.pth")

    agent.q_net.eval()
    saved = agent.total_steps
    agent.total_steps = 999_999_999  # epsilon mínimo

    rewards = []
    print(f"\n{'─'*45}")
    print(f"  {'Ep':>4}  {'Recompensa':>12}  {'Estado'}")
    print(f"{'─'*45}")

    for ep in range(n_episodes):
        state, _ = env.reset(seed=2000 + ep)
        total, steps = 0.0, 0
        while steps < HP["max_episode_steps"]:
            action = agent.select_action(state)
            state, r, term, trunc, _ = env.step(action)
            total += r
            steps += 1
            if term or trunc:
                break
        rewards.append(total)
        bar = "█" * int(total / 5)
        print(f"  {ep+1:>4}  {total:>12.1f}  {bar}")

    env.close()
    agent.total_steps = saved

    print(f"{'─'*45}")
    print(f"  {'Media':<12}: {np.mean(rewards):.1f}")
    print(f"  {'Std':<12}: {np.std(rewards):.1f}")
    print(f"  {'Máximo':<12}: {np.max(rewards):.1f}")
    print(f"  {'Mínimo':<12}: {np.min(rewards):.1f}")
    print(f"  {'Mediana':<12}: {np.median(rewards):.1f}")
    print(f"{'─'*45}\n")
    return rewards


# ══════════════════════════════════════════════════════════════════════════════
#  GRÁFICAS — 6 paneles
# ══════════════════════════════════════════════════════════════════════════════

def _smooth(v, w=50):
    return np.convolve(v, np.ones(w) / w, mode='valid') if len(v) >= w else np.array(v)


def _plot_results(rewards, avg_rwds, max_rwds, lengths,
                  epsilons, betas, losses, best_ever):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Dueling DQN + PER + Double DQN — ALE/Breakout-v5\n"
        f"Mejor recompensa histórica: {best_ever:.0f} pts",
        fontsize=13, fontweight='bold'
    )

    # ── Panel 1: Recompensa cruda + media móvil ────────────────────────────
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.15, color='steelblue', label='Por episodio')
    if len(rewards) >= 50:
        ax.plot(range(49, len(rewards)), _smooth(rewards, 50),
                color='steelblue', lw=2.5, label='Media móvil (50 eps)')
    ax.axhline(best_ever, color='gold', linestyle='--', lw=1.5,
               label=f'Mejor ever: {best_ever:.0f}')
    ax.set_title("Recompensa por Episodio", fontweight='bold')
    ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 2: Avg y Max por ventana de 20 ──────────────────────────────
    ax = axes[0, 1]
    ax.plot(avg_rwds, color='royalblue',  lw=2,   label='Avg(20) por episodio')
    ax.plot(max_rwds, color='darkorange', lw=1.5,
            alpha=0.7, label='Max(20) por episodio')
    if len(avg_rwds) >= 50:
        ax.plot(range(49, len(avg_rwds)), _smooth(avg_rwds, 50),
                color='navy', lw=2.5, label='Avg(20) suavizado')
    ax.set_title("Promedio y Máximo — Ventana 20 eps", fontweight='bold')
    ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 3: Duración de episodios ────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(lengths, alpha=0.2, color='mediumseagreen')
    if len(lengths) >= 50:
        ax.plot(range(49, len(lengths)), _smooth(lengths, 50),
                color='mediumseagreen', lw=2.5)
    ax.set_title("Duración de Episodios (pasos)", fontweight='bold')
    ax.set_xlabel("Episodio"); ax.set_ylabel("Pasos"); ax.grid(alpha=0.3)

    # ── Panel 4: Epsilon ──────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epsilons, color='mediumpurple', lw=2)
    ax.fill_between(range(len(epsilons)), epsilons, alpha=0.2, color='mediumpurple')
    ax.set_title("Decaimiento de Epsilon (ε)", fontweight='bold')
    ax.set_xlabel("Episodio"); ax.set_ylabel("ε")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    # ── Panel 5: Beta PER ────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(betas, color='teal', lw=2)
    ax.fill_between(range(len(betas)), betas, alpha=0.2, color='teal')
    ax.set_title("Crecimiento de Beta PER (β)", fontweight='bold')
    ax.set_xlabel("Episodio"); ax.set_ylabel("β (importance sampling)")
    ax.set_ylim(0.3, 1.05); ax.grid(alpha=0.3)

    # ── Panel 6: Loss ────────────────────────────────────────────────────
    ax = axes[1, 2]
    if losses:
        ax.plot(losses, alpha=0.2, color='crimson')
        if len(losses) >= 200:
            ax.plot(range(199, len(losses)), _smooth(losses, 200),
                    color='crimson', lw=2.5)
    ax.set_title("Huber Loss (pesada por IS)", fontweight='bold')
    ax.set_xlabel("Paso de optimización"); ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out = "results/training_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  📊 Gráfica guardada en {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",   action="store_true", help="Solo evaluación")
    parser.add_argument("--render", action="store_true", help="Renderizar juego")
    parser.add_argument("--steps",  type=int, default=HP["total_steps"],
                        help="Total de pasos de entrenamiento")
    args = parser.parse_args()

    if args.eval:
        evaluate(render=args.render)
    else:
        agent, _ = train(total_steps=args.steps)
        evaluate(agent)
