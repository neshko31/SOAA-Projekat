"""
Ball & Beam — Vizualizacija sa odabirom algoritma (Q-Learning / SARSA)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
        self.steps, self.max_steps = 0, 500
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
        self.x         += self.TIME_STEP * self.x_dot
        self.x_dot     += self.TIME_STEP * self.x_ddot()
        self.alpha     += self.TIME_STEP * self.alpha_dot
        self.alpha_dot += self.TIME_STEP * alpha_ddot
        return [self.x, self.x_dot, self.alpha, self.alpha_dot], self.trunc, self.term

    def reset(self):
        self.x = self.x_dot = self.alpha = self.alpha_dot = 0
        self.steps, self.trunc, self.term = 0, False, False
        return [self.x, self.x_dot, self.alpha, self.alpha_dot]


# ── Diskretizacija ────────────────────────────────────────────────────────────

N = 20
x_bins         = np.linspace(-0.2,  0.2,  N + 1)
x_dot_bins     = np.linspace(-0.5,  0.5,  N + 1)
alpha_bins     = np.linspace(-0.35, 0.35, N + 1)
alpha_dot_bins = np.linspace(-1.0,  1.0,  N + 1)

def discretize(state):
    x, xd, a, ad = state
    def idx(v, bins): return max(0, min(np.digitize(v, bins) - 1, N - 1))
    return (idx(np.clip(x,  -0.2,  0.2),  x_bins),
            idx(np.clip(xd, -0.5,  0.5),  x_dot_bins),
            idx(np.clip(a,  -0.35, 0.35), alpha_bins),
            idx(np.clip(ad, -1.0,  1.0),  alpha_dot_bins))


# ── Meni za odabir algoritma ──────────────────────────────────────────────────

def show_menu(Q_ql, Q_sarsa):
    """
    Prikazuje početni ekran sa dva dugmeta.
    Vraća odabranu Q tabelu i naziv algoritma.
    """
    selected = [None]   # čuva rezultat odabira

    fig_menu, ax_menu = plt.subplots(figsize=(5, 3.2))
    fig_menu.patch.set_facecolor('#0d1117')
    ax_menu.set_facecolor('#0d1117')
    ax_menu.axis('off')
    fig_menu.canvas.manager.set_window_title('Ball & Beam — Odabir algoritma')

    # Naslov
    ax_menu.text(0.5, 0.82, 'Ball & Beam',
                 transform=ax_menu.transAxes, fontsize=18, fontweight='bold',
                 color='#c9d1d9', ha='center', va='center', fontfamily='monospace')
    ax_menu.text(0.5, 0.64, 'Odaberite algoritam:',
                 transform=ax_menu.transAxes, fontsize=10,
                 color='#8b949e', ha='center', va='center', fontfamily='monospace')

    # ── Dugme: Q-Learning ──
    ax_ql = fig_menu.add_axes([0.12, 0.18, 0.34, 0.24])
    btn_ql = Button(ax_ql, 'Q-Learning', color='#1f3a5f', hovercolor='#2d5a9e')
    btn_ql.label.set_color('#58a6ff')
    btn_ql.label.set_fontsize(11)
    btn_ql.label.set_fontfamily('monospace')

    # ── Dugme: SARSA ──
    ax_sa = fig_menu.add_axes([0.54, 0.18, 0.34, 0.24])
    btn_sa = Button(ax_sa, 'SARSA', color='#3a1f5f', hovercolor='#6b3a9e')
    btn_sa.label.set_color('#bc8cff')
    btn_sa.label.set_fontsize(11)
    btn_sa.label.set_fontfamily('monospace')

    def on_ql(event):
        selected[0] = ('Q-Learning', Q_ql)
        plt.close(fig_menu)

    def on_sa(event):
        selected[0] = ('SARSA', Q_sarsa)
        plt.close(fig_menu)

    btn_ql.on_clicked(on_ql)
    btn_sa.on_clicked(on_sa)

    plt.tight_layout()
    plt.show()   # blokira dok se prozor ne zatvori

    return selected[0]   # može biti None ako korisnik zatvori prozor


# ── Vizualizacija ─────────────────────────────────────────────────────────────

def run_visualization(Q, algo_name=''):
    env = Enviroment()
    x_th, x_dot_th, alpha_th, alpha_dot_th = 0.2, 0.5, 0.35, 1.0

    # Boja loptice i naslova zavisi od algoritma
    if 'SARSA' in algo_name:
        ball_color  = '#bc8cff'   # ljubičasta za SARSA
        title_color = '#bc8cff'
    else:
        ball_color  = '#58a6ff'   # plava za Q-Learning
        title_color = '#58a6ff'

    def random_state():
        env.reset()
        env.x         = np.clip(np.random.normal(0, x_th      / 4), -0.19, 0.19)
        env.x_dot     = np.clip(np.random.normal(0, x_dot_th  / 4), -0.49, 0.49)
        env.alpha     = np.clip(np.random.normal(0, alpha_th   / 4), -0.34, 0.34)
        env.alpha_dot = np.clip(np.random.normal(0, alpha_dot_th / 4), -0.99, 0.99)
        return [env.x, env.x_dot, env.alpha, env.alpha_dot]

    state = random_state()

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(-0.26, 0.26)
    ax.set_ylim(-0.13, 0.13)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.canvas.manager.set_window_title(f'Ball & Beam — {algo_name}')

    # Oznaka aktivnog algoritma (gore desno)
    ax.text(0.98, 0.97, algo_name,
            transform=ax.transAxes, fontsize=9, color=title_color,
            va='top', ha='right', fontfamily='monospace', fontweight='bold')

    # Ciljna linija i granice
    ax.axvline(0,             color='#e3b341', linestyle='--', linewidth=0.9, alpha=0.5)
    ax.axvline(-env.l / 2,   color='#f85149', linestyle=':',  linewidth=0.9, alpha=0.5)
    ax.axvline( env.l / 2,   color='#f85149', linestyle=':',  linewidth=0.9, alpha=0.5)

    # Pivot
    ax.plot(0, 0, 'o', color='#f9ff46', markersize=8, zorder=5)

    # Greda
    beam_line, = ax.plot([], [], color='#ac6807', linewidth=7,
                         solid_capstyle='round', zorder=2)

    # Loptica
    ball = plt.Circle((0, 0), 0.012, color=ball_color, zorder=6)
    ax.add_patch(ball)

    # Info tekst (gore levo)
    info = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=8,
                   color='#8b949e', va='top', fontfamily='monospace')

    ep     = [1]
    step   = [0]
    paused = [True]

    # ── Dugme: pauza ──
    ax_pause = fig.add_axes([0.68, 0.03, 0.15, 0.08])
    btn_pause = Button(ax_pause, '⏸  Pauza', color='#161b22', hovercolor='#21262d')
    btn_pause.label.set_color('#c9d1d9')
    btn_pause.label.set_fontsize(9)

    def toggle_pause(event):
        paused[0] = not paused[0]
        btn_pause.label.set_text('▶  Nastavi' if paused[0] else '⏸  Pauza')
        fig.canvas.draw_idle()

    btn_pause.on_clicked(toggle_pause)

    # ── Dugme: nazad na meni ──
    ax_back = fig.add_axes([0.84, 0.03, 0.14, 0.08])
    btn_back = Button(ax_back, '⬅  Meni', color='#161b22', hovercolor='#21262d')
    btn_back.label.set_color('#c9d1d9')
    btn_back.label.set_fontsize(9)

    def go_back(event):
        plt.close(fig)

    btn_back.on_clicked(go_back)

    # ── Animacija ──
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
        bx1, by1 = -env.l / 2 * np.cos(va), -env.l / 2 * np.sin(va)
        bx2, by2 =  env.l / 2 * np.cos(va),  env.l / 2 * np.sin(va)
        beam_line.set_data([bx1, bx2], [by1, by2])

        # Loptica
        ball.center = (x * np.cos(va), x * np.sin(va) + env.r)

        info.set_text(
            f'Ep. {ep[0]}  |  Korak: {step[0]}/{env.max_steps}'
            f'  |  x={x:+.3f}m  α={alpha:+.3f}rad'
        )

        return beam_line, ball, info

    ani = animation.FuncAnimation(fig, update, interval=50,
                                  blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return ani


# ── Glavni tok ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    Q_ql    = np.load('./q-tables/Q_table.npy')
    Q_sarsa = np.load('./q-tables/Q_S_table.npy')

    while True:
        result = show_menu(Q_ql, Q_sarsa)
        if result is None:
            # Korisnik zatvorio meni — izlaz
            break
        algo_name, Q = result
        ani = run_visualization(Q, algo_name)
        # Nakon što se zatvori vizualizacija, vraćamo se na meni