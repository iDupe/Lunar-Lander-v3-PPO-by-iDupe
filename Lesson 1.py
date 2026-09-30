import os, math
import gymnasium as gym
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

HIDDEN_SIZE = 256
ACTIVATION = nn.LeakyReLU
GAMMA = 0.99
GAE_LAMBDA = 0.98
K_EPOCHS = 4
EPS_CLIP = 0.2
BATCH_SIZE = 256
ENTROPY_COEF = 0.003
MAX_GRAD_NORM = 0.5
BASE_LR = 3e-4
MIN_LR = 3e-6
LR_DECAY_ALPHA = 0.004
NUM_WORKERS = 12
NUM_ENVS = NUM_WORKERS
UPDATE_STEPS_PER_ENV = 256
MAX_STEPS = 5_000_000
TARGET_MEAN = 280.0
WINDOW_SIZE = 100
EVAL_ARGMAX = False
CHECKPOINT_EVERY = 500

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONTINUED_PATH = os.path.join(SCRIPT_DIR, "lunar_lander_ppo_continued.pth")
BEST_PATH      = os.path.join(SCRIPT_DIR, "lunar_lander_best.pth")
LL_PATH        = os.path.join(SCRIPT_DIR, "lunar_lander_ppo.pth")

LOAD_PATH      = CONTINUED_PATH
SAVE_PATH      = CONTINUED_PATH
INTERRUPT_PATH = CONTINUED_PATH
WIN_PATH       = LL_PATH
FINAL_PATH     = LL_PATH

DEVICE = t.device('cuda:0' if t.cuda.is_available() else 'cpu')


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size,
                 hidden_size=HIDDEN_SIZE, activation=ACTIVATION):
        super().__init__()
        def block(out):
            return nn.Sequential(
                nn.Linear(state_size, hidden_size), activation(),
                nn.Linear(hidden_size, hidden_size), activation(),
                nn.Linear(hidden_size, out))
        self.actor = block(action_size)
        self.critic = block(1)

    def forward(self, state):
        raise NotImplementedError("Позже дописать")

    def act(self, state, deterministic=False):
        logits = self.actor(state)
        if deterministic:
            action = t.argmax(logits, dim=-1)
            return action.detach(), t.zeros_like(action, dtype=t.float32)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate_actions(self, state, action):
        dist = Categorical(logits=self.actor(state))
        return dist.log_prob(action), self.critic(state), dist.entropy()


class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.is_terminals = [], []

    def clear(self):
        for lst in (self.states, self.actions, self.log_probs,
                    self.rewards, self.is_terminals):
            lst.clear()


class Agent:
    def __init__(self, state_size, action_size, num_envs=NUM_ENVS):
        self.gamma, self.gae_lambda = GAMMA, GAE_LAMBDA
        self.K_epochs, self.eps_clip = K_EPOCHS, EPS_CLIP
        self.batch_size, self.entropy_coef = BATCH_SIZE, ENTROPY_COEF
        self.max_grad_norm = MAX_GRAD_NORM
        self.num_envs = num_envs
        self.base_lr, self.min_lr = BASE_LR, MIN_LR
        self.lr_decay_alpha = LR_DECAY_ALPHA

        self.policy = ActorCritic(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.base_lr)
        self.policy_old = ActorCritic(state_size, action_size).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        pd = next(self.policy.parameters()).device
        print(f"policy_old на устройстве: {pd}")
        if DEVICE.type == 'cuda':
            assert pd.type == 'cuda', "Модель почему-то не на GPU!"
            print(f"   VRAM занято моделью: {t.cuda.memory_allocated(0) / 1024**2:.1f} MB")

        self.buffer = RolloutBuffer()
        self.MseLoss = nn.MSELoss()
        self.loss_actor_history, self.loss_critic_history = [], []
        self.entropy_history, self.lr_history = [], []
        self.current_lr = self.base_lr

    def adapt_lr(self, current_mean):
        if current_mean is None or np.isnan(current_mean):
            return self.base_lr
        scale = math.exp(-self.lr_decay_alpha * max(0.0, current_mean))
        new_lr = max(self.min_lr, self.base_lr * scale)
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr
        self.current_lr = new_lr
        return new_lr

    def select_action(self, states, deterministic=False):
        with t.no_grad():
            states_t = t.FloatTensor(states).to(DEVICE)
            actions, log_probs = self.policy_old.act(states_t, deterministic=deterministic)
        self.buffer.states.append(states_t)
        self.buffer.actions.append(actions)
        self.buffer.log_probs.append(log_probs)
        return actions.cpu().numpy()

    def update(self, last_values=None):
        old_states = t.cat(self.buffer.states, dim=0).detach()
        old_actions = t.cat(self.buffer.actions, dim=0).detach().view(-1)
        old_log_probs = t.cat(self.buffer.log_probs, dim=0).detach().view(-1)
        rewards_matrix = t.FloatTensor(np.array(self.buffer.rewards)).to(DEVICE)
        terminals_matrix = t.FloatTensor(np.array(self.buffer.is_terminals)).to(DEVICE)
        num_steps = rewards_matrix.size(0)

        with t.no_grad():
            state_values_old = self.policy.critic(old_states).view(num_steps, self.num_envs)

        last_values = (t.zeros(self.num_envs).to(DEVICE) if last_values is None
                       else last_values.to(DEVICE).view(-1))

        advantages_matrix = t.zeros_like(rewards_matrix).to(DEVICE)
        gae = t.zeros(self.num_envs).to(DEVICE)
        for i in reversed(range(num_steps)):
            is_terminal = terminals_matrix[i]
            next_value = last_values if i == num_steps - 1 else state_values_old[i + 1]
            next_value = next_value * (1.0 - is_terminal)
            delta = rewards_matrix[i] + self.gamma * next_value - state_values_old[i]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - is_terminal) * gae
            advantages_matrix[i] = gae

        advantages = advantages_matrix.view(-1)
        rewards = advantages + state_values_old.view(-1)

        eal, ecl, ee = [], [], []
        n = old_states.size(0)
        for _ in range(self.K_epochs):
            perm = t.randperm(n, device=DEVICE)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                b_states, b_actions = old_states[idx], old_actions[idx]
                b_log_probs, b_adv, b_rew = old_log_probs[idx], advantages[idx], rewards[idx]
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                log_probs, state_values, dist_entropy = self.policy.evaluate_actions(b_states, b_actions)
                log_probs = log_probs.view(-1)
                state_values = state_values.view(-1)
                dist_entropy = dist_entropy.mean()

                ratio = t.exp(log_probs - b_log_probs)
                surr1 = ratio * b_adv
                surr2 = t.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv

                actor_loss = -t.min(surr1, surr2).mean()
                critic_loss = 0.5 * self.MseLoss(state_values, b_rew)
                loss = actor_loss + critic_loss - self.entropy_coef * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                eal.append(actor_loss.item())
                ecl.append(critic_loss.item())
                ee.append(dist_entropy.item())

        self.loss_actor_history.append(np.mean(eal))
        self.loss_critic_history.append(np.mean(ecl))
        self.entropy_history.append(np.mean(ee))

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()


if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("=" * 60)
    print(f"Torch version: {t.__version__}")
    print(f"CUDA доступна: {t.cuda.is_available()}")
    if t.cuda.is_available():
        print(f"CUDA version: {t.version.cuda}")
        print(f"GPU: {t.cuda.get_device_name(0)}")
        print(f"VRAM: {t.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Устройство обучения: {DEVICE}")
    else:
        print("CUDA НЕ доступна - обучение пойдёт на CPU (будет ОЧЕНЬ медленно)")
    print("=" * 60)

    print("КОНФИГ:")
    print(f"  hidden={HIDDEN_SIZE}, activation={ACTIVATION.__name__}")
    print(f"  gamma={GAMMA}, lambda={GAE_LAMBDA}, K={K_EPOCHS}, clip={EPS_CLIP}")
    print(f"  batch={BATCH_SIZE}, entropy_coef={ENTROPY_COEF}, grad_clip={MAX_GRAD_NORM}")
    print(f"  base_lr={BASE_LR}, min_lr={MIN_LR}, alpha={LR_DECAY_ALPHA}")
    print(f"  num_workers={NUM_WORKERS}, num_envs={NUM_ENVS}, steps_per_env={UPDATE_STEPS_PER_ENV}")
    print(f"  target_mean={TARGET_MEAN}, eval_argmax={EVAL_ARGMAX}")
    print("=" * 60)

    envs = gym.vector.AsyncVectorEnv(
        [lambda: gym.make('LunarLander-v3') for _ in range(NUM_WORKERS)])
    state_size = envs.single_observation_space.shape[0]
    action_size = envs.single_action_space.n

    agent = Agent(state_size, action_size, num_envs=NUM_ENVS)

    if os.path.exists(LOAD_PATH):
        print(f"Загружаю веса из {LOAD_PATH}")
        state = t.load(LOAD_PATH, map_location=DEVICE)
        agent.policy.load_state_dict(state)
        agent.policy_old.load_state_dict(state)
        print("   Веса загружены. Продолжаю обучение с прошлой точки.")
    else:
        print(f"Файл {LOAD_PATH} не найден - обучение начнётся с нуля.")

    scores_window, all_scores, all_mean_scores = [], [], []
    current_episode_rewards = np.zeros(NUM_ENVS)
    episode_count = 0
    success = False
    best_mean = -np.inf

    states, _ = envs.reset()
    print(f"Продолжаю PPO на {NUM_WORKERS} воркерах. Целевой средний: > {TARGET_MEAN}!")

    try:
        for step in range(1, MAX_STEPS):
            actions = agent.select_action(states, deterministic=EVAL_ARGMAX)
            next_states, rewards, terminated, truncated, _ = envs.step(actions)
            dones = terminated | truncated

            agent.buffer.rewards.append(rewards)
            agent.buffer.is_terminals.append(dones)

            current_episode_rewards += rewards
            states = next_states

            for i in range(NUM_ENVS):
                if not dones[i]:
                    continue
                episode_count += 1
                final_score = current_episode_rewards[i]
                scores_window.append(final_score)
                all_scores.append(final_score)
                if len(scores_window) > WINDOW_SIZE:
                    scores_window.pop(0)

                current_mean = np.mean(scores_window)
                all_mean_scores.append(current_mean)

                print(f"Эпизод {episode_count}\tВоркер {i}\tСчёт: {final_score:.2f}\t"
                      f"Средний ({WINDOW_SIZE}): {current_mean:.2f}\tLR: {agent.current_lr:.2e}")
                current_episode_rewards[i] = 0

                if current_mean > best_mean:
                    best_mean = current_mean
                    t.save(agent.policy_old.state_dict(), BEST_PATH)

                if episode_count % CHECKPOINT_EVERY == 0:
                    t.save(agent.policy_old.state_dict(), SAVE_PATH)
                    print(f"   Чекпоинт сохранён (best_mean={best_mean:.2f})")

                if current_mean >= TARGET_MEAN and len(scores_window) >= WINDOW_SIZE:
                    print(f"\n\nПобеда! Средний {current_mean:.2f} на эпизоде {episode_count}!")
                    t.save(agent.policy_old.state_dict(), WIN_PATH)
                    t.save(agent.policy_old.state_dict(), CONTINUED_PATH)
                    success = True
                    break

            if success:
                break

            if step % UPDATE_STEPS_PER_ENV == 0:
                with t.no_grad():
                    last_values = agent.policy_old.critic(
                        t.FloatTensor(states).to(DEVICE)).squeeze(-1)

                current_mean_for_lr = np.mean(scores_window) if scores_window else None
                agent.adapt_lr(current_mean_for_lr)
                agent.lr_history.append(agent.current_lr)

                agent.update(last_values)

                if DEVICE.type == 'cuda':
                    alloc = t.cuda.memory_allocated(0) / 1024**2
                    reserv = t.cuda.memory_reserved(0) / 1024**2
                    print(f"   [GPU] alloc: {alloc:.0f} MB / reserved: {reserv:.0f} MB")

    except KeyboardInterrupt:
        print("\nПрервано пользователем. Сохраняю модель...")
        t.save(agent.policy_old.state_dict(), INTERRUPT_PATH)
    finally:
        envs.close()

    t.save(agent.policy_old.state_dict(), FINAL_PATH)
    t.save(agent.policy_old.state_dict(), CONTINUED_PATH)

    print("\nВывожу графики...")
    plt.figure(num="1. Награды агента", figsize=(10, 5))
    plt.plot(all_scores, color="skyblue", alpha=0.3, label="Счет за эпизод")
    plt.plot(all_mean_scores, color="royalblue", linewidth=2, label=f"Среднее за {WINDOW_SIZE} эп.")
    plt.axhline(y=TARGET_MEAN, color="green", linestyle="--", alpha=0.6, label=f"Порог ({TARGET_MEAN})")
    plt.title("Динамика вознаграждений"); plt.xlabel("Эпизоды"); plt.ylabel("Награда")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.figure(num="2. Энтропия политики", figsize=(10, 5))
    plt.plot(agent.entropy_history, color="mediumpurple", linewidth=1.5)
    plt.title("Энтропия политики (Exploration)"); plt.xlabel("Обновления"); plt.ylabel("Энтропия")
    plt.grid(True, alpha=0.3)

    plt.figure(num="3. Learning Rate", figsize=(10, 5))
    plt.plot(agent.lr_history, color="darkorange", linewidth=1.5)
    plt.title("Адаптивный Learning Rate"); plt.xlabel("Обновления"); plt.ylabel("LR")
    plt.yscale('log'); plt.grid(True, alpha=0.3)

    plt.figure(num="4. Ошибка Критика", figsize=(10, 5))
    plt.plot(agent.loss_critic_history, color="seagreen", linewidth=1.5)
    plt.title("Ошибка Критика (Value Loss MSE)"); plt.xlabel("Обновления"); plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.show()