import os
import gymnasium as gym
import numpy as np
import torch as t
import torch.nn as nn
import pygame
from torch.distributions import Categorical

gpu = t.device('cuda:0' if t.cuda.is_available() else 'cpu')


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, action_size))
        self.critic = nn.Sequential(
            nn.Linear(state_size, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 1))

    def act(self, state, deterministic=False):
        logits = self.actor(state)
        if deterministic:
            return t.argmax(logits, dim=-1).detach(), None
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a.detach(), dist.log_prob(a).detach()


def load_model(path, state_size, action_size):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    model = ActorCritic(state_size, action_size).to(gpu)
    state = t.load(path, map_location=gpu)
    print("=" * 60)
    print(f"Файл чтения: {path}")
    for k, v in state.items():
        print(f"   {k}: {tuple(v.shape)}")
    print("=" * 60)
    model.load_state_dict(state)
    model.eval()
    print("Веса загружены.")
    return model


class EngineHUD:
    ACTION_TO_ENGINE = {0: None, 1: "left", 2: "main", 3: "right"}
    COLORS = {"main": (255, 180, 60), "left": (90, 190, 255), "right": (255, 120, 120)}
    LABELS = {"main": "MAIN ", "left": "LEFT ", "right": "RIGHT"}
    SMOOTH = 0.10
    BORDER_ATTACK = 0.5
    BORDER_DECAY = 0.06
    BORDER_COLOR = (255, 255, 255)

    def __init__(self):
        pygame.font.init()
        self.font_small = pygame.font.SysFont("consolas", 12, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 13, bold=True)
        self.reset()

    def reset(self):
        self.smooth_probs = {k: 0.0 for k in ("main", "left", "right")}
        self.border_glow = {k: 0.0 for k in ("main", "left", "right")}

    def draw(self, screen, engine_probs, current_action):
        if screen is None:
            return
        for k in self.smooth_probs:
            self.smooth_probs[k] += (engine_probs[k] - self.smooth_probs[k]) * self.SMOOTH
        active = self.ACTION_TO_ENGINE.get(current_action)
        for k in self.border_glow:
            tgt = 1.0 if k == active else 0.0
            rate = self.BORDER_ATTACK if tgt > self.border_glow[k] else self.BORDER_DECAY
            self.border_glow[k] += (tgt - self.border_glow[k]) * rate

        panel = pygame.Surface((240, 118), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        screen.blit(panel, (8, 8))
        screen.blit(self.font_title.render("ENGINE MONITOR", True, (220, 220, 220)), (16, 12))

        bar_x, bar_w, bar_h, radius, row_h, y0 = 68, 130, 12, 6, 26, 36
        for i, eng in enumerate(("main", "left", "right")):
            y = y0 + i * row_h
            glow = self.border_glow[eng]
            v = int(140 + 115 * glow)
            screen.blit(self.font_small.render(self.LABELS[eng], True, (v, v, v)), (16, y))
            pygame.draw.rect(screen, (50, 50, 50), (bar_x, y, bar_w, bar_h), border_radius=radius)
            ratio = max(0.0, min(1.0, self.smooth_probs[eng]))
            fill_w = int(bar_w * ratio)
            if fill_w > 0:
                pygame.draw.rect(screen, self.COLORS[eng],
                                 (bar_x, y, max(fill_w, bar_h), bar_h),
                                 border_radius=radius)
            if glow > 0.01:
                ov = pygame.Surface((bar_w + 4, bar_h + 4), pygame.SRCALPHA)
                pygame.draw.rect(ov, (*self.BORDER_COLOR, int(255 * glow)),
                                 (0, 0, bar_w + 4, bar_h + 4),
                                 width=2, border_radius=radius + 2)
                screen.blit(ov, (bar_x - 2, y - 2))
            screen.blit(self.font_small.render(
                f"{int(round(ratio * 100)):>3}%", True, (220, 220, 220)),
                (bar_x + bar_w + 6, y - 1))


def run_forever(model, deterministic=True, render=True):
    pygame.init()
    hud = EngineHUD()
    env = gym.make('LunarLander-v3', render_mode='human' if render else None)
    scores, ep, stop_reason = [], 0, None
    try:
        while True:
            ep += 1
            state, _ = env.reset()
            done, total_reward, steps, aborted = False, 0.0, 0, False
            hud.reset()
            while not done:
                if render:
                    pygame.event.pump()
                    if any(e.type == pygame.QUIT for e in pygame.event.get()):
                        stop_reason, aborted = "Окно pygame закрыто", True
                        break
                s = t.FloatTensor(state).unsqueeze(0).to(gpu)
                with t.no_grad():
                    logits = model.actor(s)
                    probs = t.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    action = (int(t.argmax(logits, -1).item()) if deterministic
                              else int(Categorical(logits=logits).sample().item()))
                engine_probs = {"main": float(probs[2]),
                                "left": float(probs[1]),
                                "right": float(probs[3])}
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                if render:
                    screen = pygame.display.get_surface()
                    if screen is not None:
                        hud.draw(screen, engine_probs, action)
                        pygame.display.flip()
            if aborted:
                break
            scores.append(total_reward)
            print(f"Эпизод {ep:>4} | [{'ARGMAX' if deterministic else 'SAMPLE'}] "
                  f"Счёт: {total_reward:>8.2f} | Шагов: {steps}")
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
    except pygame.error as e:
        stop_reason = f"Окно pygame закрыто ({e})"

    try:
        env.close()
    except Exception:
        pass

    print()
    print("=" * 60)
    print(f"Остановка: {stop_reason}")
    print(f"Всего полных эпизодов: {len(scores)}")
    if scores:
        arr = np.array(scores)
        print(f"Средний:  {arr.mean():.2f}")
        print(f"Мин:      {arr.min():.2f}")
        print(f"Макс:     {arr.max():.2f}")
        print(f"Медиана:  {np.median(arr):.2f}")
        print(f"Std:      {arr.std():.2f}")
        if len(arr) >= 100:
            print(f"Средний (последние 100):  {arr[-100:].mean():.2f}")
        if len(arr) >= 500:
            print(f"Средний (последние 500):  {arr[-500:].mean():.2f}")
        if len(arr) >= 1000:
            print(f"Средний (последние 1000): {arr[-1000:].mean():.2f}")
    print("=" * 60)
    return scores


if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PTH_PATH = os.path.join(SCRIPT_DIR, "lunar_lander_ppo.pth")
    if not os.path.exists(PTH_PATH):
        if os.path.exists("lunar_lander_ppo.pth"):
            PTH_PATH = "lunar_lander_ppo.pth"
        else:
            print("Файл не найден в папке. Ищу .pth в окрестностях...")
            for d in (SCRIPT_DIR, os.getcwd()):
                if os.path.isdir(d):
                    files = [f for f in os.listdir(d) if f.endswith('.pth')]
                    if files:
                        print(f"   Найдено в {d}: {files}")
            raise FileNotFoundError(f"Не нашёл {PTH_PATH},  шуруй обучать. ")

    model = load_model(PTH_PATH, 8, 4)
    run_forever(model, deterministic=True, render=True)