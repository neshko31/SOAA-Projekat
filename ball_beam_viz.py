"""
Ball & Beam — Jednostavna vizualizacija (samo greda i loptica)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from matplotlib.widgets import Button


# ── Fizički model ─────────────────────────────────────────────────────────────

class Enviroment:
    def __init__(self, m=0.113, r=0.015, l=0.4, g=9.81, kf=0.5, TIME_STEP=0.05):
        self.m, self.r, self.l = m, r, l
        self.J = 0.4 * m * r ** 2
        self.g, self.kf = g, kf
        self.TIME_STEP = TIME_STEP
        self.p = 1
        self.x = self.x_dot = self.alpha = self.alpha_dot = 0
        self.x_dot_max, self.alpha_max, self.alpha_dot_max = 0.5, 0.35, 1
        self.steps, self.max_steps = 0, 100
        self.trunc = self.term = False
        self.actions = [-5, -2.5, 0, 2.5, 5]

    def x_ddot(self):
        return (-self.x_dot * (self.kf * self.r**2) +
                np.sin(self.alpha) * self.m * self.g * self.r**2) / (self.m * self.r**2 + self.J)

    def take_action(self, action):
        self.steps += 1
        if self.steps > self.max_steps:
            self.trunc, self.term = True, False
            return [self.x, self.x_dot, self.alpha, self.alpha_dot], self.trunc, self.term
        if abs(self.x) > self.l / 2:
            self.term = True
            return [self.x, self.x_dot, self.alpha, self.alpha_dot], self.trunc, self.term
        alpha_ddot = self.actions[action]
        self.x      += self.TIME_STEP * self.x_dot
        self.x_dot  += self.TIME_STEP * self.x_ddot()
        self.alpha   += self.TIME_STEP * self.alpha_dot
        self.alpha_dot += self.TIME_STEP * alpha_ddot
        return [self.x, self.x_dot, self.alpha, self.alpha_dot], self.trunc, self.term

    def reset(self):
        self.x = self.x_dot = self.alpha = self.alpha_dot = 0
        self.steps, self.trunc, self.term = 0, False, False
        return [self.x, self.x_dot, self.alpha, self.alpha_dot]


# ── Diskretizacija ────────────────────────────────────────────────────────────

N = 20
x_bins        = np.linspace(-0.2,  0.2,  N + 1)
x_dot_bins    = np.linspace(-0.5,  0.5,  N + 1)
alpha_bins    = np.linspace(-0.35, 0.35, N + 1)
alpha_dot_bins = np.linspace(-1.0,  1.0,  N + 1)

def discretize(state):
    x, xd, a, ad = state
    def idx(v, bins): return max(0, min(np.digitize(v, bins) - 1, N - 1))
    return (idx(np.clip(x, -0.2, 0.2), x_bins),
            idx(np.clip(xd, -0.5, 0.5), x_dot_bins),
            idx(np.clip(a, -0.35, 0.35), alpha_bins),
            idx(np.clip(ad, -1.0, 1.0), alpha_dot_bins))


# ── Vizualizacija ─────────────────────────────────────────────────────────────

def run_visualization(Q):
    env = Enviroment()
    L = env.l / 2   # 0.20 m

    def random_state():
        s = env.reset()
        # Normalna raspodela, isečena na granice okruženja
        # env.x         = np.clip(np.random.normal(0, 0.10), -0.19, 0.19)
        # env.x_dot     = np.clip(np.random.normal(0, 0.15), -0.49, 0.49)
        # env.alpha     = np.clip(np.random.normal(0, 0.10), -0.34, 0.34)
        # env.alpha_dot = np.clip(np.random.normal(0, 0.30), -0.99, 0.99)
        return s
        # return [env.x, env.x_dot, env.alpha, env.alpha_dot]

    state = random_state()

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(-0.26, 0.26)
    ax.set_ylim(-0.13, 0.13)
    ax.set_aspect('equal')
    ax.axis('off')

    # Ciljna linija i granice
    ax.axvline(0,  color='#e3b341', linestyle='--', linewidth=0.9, alpha=0.5)
    ax.axvline(-L, color='#f85149', linestyle=':',  linewidth=0.9, alpha=0.5)
    ax.axvline( L, color='#f85149', linestyle=':',  linewidth=0.9, alpha=0.5)

    # Pivot
    ax.plot(0, 0, 'o', color='#2ea043', markersize=8, zorder=5)

    # Greda
    beam_line, = ax.plot([], [], color='#238636', linewidth=7,  solid_capstyle='round', zorder=2)

    # Loptica
    ball      = plt.Circle((0, 0), 0.012, color='#58a6ff', zorder=6)
    ax.add_patch(ball)

    # Info tekst (gore levo)
    info = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                   fontsize=8, color='#8b949e', va='top', fontfamily='monospace')

    ep = [1]
    step = [0]
    paused = [True]

    # Dugme za pauzu
    ax_btn = fig.add_axes([0.82, 0.03, 0.15, 0.08])
    btn = Button(ax_btn, '⏸  Pauza', color='#161b22', hovercolor='#21262d')
    btn.label.set_color('#c9d1d9')
    btn.label.set_fontsize(9)

    def toggle_pause(event):
        paused[0] = not paused[0]
        btn.label.set_text('▶  Nastavi' if paused[0] else '⏸  Pauza')
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_pause)

    def update(_):
        nonlocal state

        if paused[0]:
            return beam_line, ball, info

        x, _, alpha, _ = state

        action = int(np.argmax(Q[discretize(state)]))
        state, trunc, term = env.take_action(action)

        if term or trunc:
            state = random_state()
            ep[0] += 1
            step[0] = 0
        else:
            step[0] += 1

        # Greda
        va = -alpha
        bx1, by1 = -L * np.cos(va), -L * np.sin(va)
        bx2, by2 =  L * np.cos(va),  L * np.sin(va)
        beam_line.set_data([bx1, bx2], [by1, by2])

        # Loptica
        ball_x = x * np.cos(va)
        ball_y = x * np.sin(va) + env.r
        ball.center = (ball_x, ball_y)

        info.set_text(f'Ep. {ep[0]}  |  Korak: {step[0]}/{env.max_steps}  |  x={x:+.3f}m  α={alpha:+.3f}rad')

        return beam_line, ball, info

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return ani


if __name__ == '__main__':
    Q = np.load('./q-tables/Q_table.npy')
    ani = run_visualization(Q)