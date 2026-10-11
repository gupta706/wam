"""Build compact GIF previews and browser data from the repository simulations."""

import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from environment.kuramoto_sivashinsky import create_ks_params, get_initial_conditions_ks, integrate_ks


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
FRAME_COUNT = 72
GIF_FPS = 12


def sample_rows(values, count=FRAME_COUNT):
    indices = np.linspace(0, len(values) - 1, count, dtype=int)
    return values[indices]


def integrate_rk4(rhs, initial, params, dt, end_time, save_every):
    """NumPy-vectorized equivalent of environment.integrators.build_integrator."""
    steps = int(end_time / dt)
    ys = np.empty((steps // save_every + 1, len(initial)), dtype=initial.dtype)
    ts = np.empty(steps // save_every + 1)
    ys[0], ts[0] = initial, 0.0
    state, saved = initial.copy(), 1
    for step in range(steps):
        t = step * dt
        k1 = dt * rhs(t, state, params)
        k2 = dt * rhs(t + dt/2, state + k1/2, params)
        k3 = dt * rhs(t + dt/2, state + k2/2, params)
        k4 = dt * rhs(t + dt, state + k3, params)
        state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        if (step + 1) % save_every == 0:
            ys[saved], ts[saved] = state, (step + 1) * dt
            saved += 1
    return ts[:saved], ys[:saved]


def kuramoto_setup(n, populations):
    np.random.seed(42)
    sizes = [n // populations] * populations
    sizes[-1] += n - sum(sizes)
    coupling = np.full((n, n), 0.1)
    start = 0
    for size in sizes:
        coupling[start:start+size, start:start+size] = 5.0
        start += size
    coupling /= n
    omega = np.random.normal(0, 0.5, size=n)
    np.random.seed(99)
    initial = np.random.uniform(0, 2*np.pi, size=n)
    return coupling, omega, initial


def kuramoto_rhs(_t, theta, params):
    coupling, omega = params
    return omega + np.sum(coupling * np.sin(theta[None, :] - theta[:, None]), axis=1)


def stuart_setup(n, clusters):
    np.random.seed(42)
    sizes = [n // clusters] * clusters
    sizes[-1] += n - sum(sizes)
    coupling = np.full((n, n), 0.01)
    start = 0
    for size in sizes:
        coupling[start:start+size, start:start+size] = 2.0
        start += size
    coupling /= n
    lam = np.ones(n)
    omega = np.random.normal(2.0, 0.2, size=n)
    np.random.seed(99)
    radius = np.random.uniform(0.1, 0.2, size=n)
    theta = np.random.uniform(0, 2*np.pi, size=n)
    return (coupling, lam, omega), radius * np.exp(1j*theta), sizes


def stuart_rhs(_t, z, params):
    coupling, lam, omega = params
    return ((lam + 1j*omega)*z - np.abs(z)**2*z
            + coupling @ z - coupling.sum(axis=1)*z)


def save_kuramoto():
    n, populations = 200, 5
    coupling, omega, initial = kuramoto_setup(n, populations)
    times, phases = integrate_rk4(kuramoto_rhs, initial, (coupling, omega), 0.01, 50.0, 10)
    phases = sample_rows(phases)
    times = sample_rows(times)
    groups = np.repeat(np.arange(populations), n // populations)
    order = np.stack([
        np.abs(np.mean(np.exp(1j * phases[:, groups == group]), axis=1))
        for group in range(populations)
    ], axis=1)

    colors = plt.get_cmap("tab10")(np.arange(populations))
    fig, (ax_phase, ax_order) = plt.subplots(1, 2, figsize=(11.2, 6.3), dpi=144)
    fig.patch.set_facecolor("#f7f8f5")
    for ax in (ax_phase, ax_order):
        ax.set_facecolor("#f7f8f5")
        ax.spines[["top", "right"]].set_visible(False)
    image = ax_phase.imshow(np.full((n, FRAME_COUNT), np.nan), origin="lower", aspect="auto",
                            extent=[times[0], times[-1], 0, n], vmin=0, vmax=2*np.pi,
                            cmap="twilight", interpolation="nearest")
    ax_phase.set(title="Phase histories reveal group locking", xlabel="Time", ylabel="Oscillator")
    lines = [ax_order.plot([], [], color=colors[i], lw=2, label=f"Group {i+1}")[0]
             for i in range(populations)]
    ax_order.set(xlim=(times[0], times[-1]), ylim=(0, 1.05), title="Within-group synchrony",
                 xlabel="Time", ylabel="Order parameter  R")
    ax_order.legend(frameon=False, fontsize=8, ncol=2)

    def update(frame):
        image.set_data(np.where(np.arange(FRAME_COUNT)[None, :] <= frame,
                                phases.T, np.nan))
        for i, line in enumerate(lines):
            line.set_data(times[:frame+1], order[:frame+1, i])
        return [image, *lines]

    animation = FuncAnimation(fig, update, frames=FRAME_COUNT, interval=1000/GIF_FPS, blit=False)
    animation.save(os.path.join(OUT, "kuramoto_animation.gif"), writer=PillowWriter(fps=GIF_FPS))
    plt.close(fig)
    return {"times": times.tolist(), "phases": phases.tolist(), "order": order.tolist(),
            "groups": groups.tolist()}


def save_stuart_landau():
    n, clusters = 200, 7
    params, initial, sizes = stuart_setup(n, clusters)
    times_all, z_all = integrate_rk4(stuart_rhs, initial, params, 0.01, 50.0, 5)
    indices = np.linspace(0, len(times_all)-1, FRAME_COUNT, dtype=int)
    times, z = times_all[indices], z_all[indices]
    group_ids = np.repeat(np.arange(clusters), sizes)
    chosen = np.concatenate([np.arange(sum(sizes[:i]), sum(sizes[:i])+min(5, sizes[i]))
                             for i in range(clusters)])
    sample = z[:, chosen]
    group_means = np.stack([np.mean(np.abs(z[:, group_ids == i]), axis=1)
                            for i in range(clusters)], axis=1)

    colors = plt.get_cmap("tab10")(np.arange(clusters))
    fig, (ax_amp, ax_plane) = plt.subplots(1, 2, figsize=(11.2, 6.3), dpi=144)
    fig.patch.set_facecolor("#f7f8f5")
    for ax in (ax_amp, ax_plane):
        ax.set_facecolor("#f7f8f5")
        ax.spines[["top", "right"]].set_visible(False)
    amp_lines = [ax_amp.plot([], [], color=colors[i], lw=1.7, label=f"Cluster {i+1}")[0]
                 for i in range(clusters)]
    ax_amp.set(xlim=(times[0], times[-1]), ylim=(0, max(1.5, float(np.abs(z).max())*1.05)),
               title="Amplitude grows, then saturates", xlabel="Time", ylabel="Mean |z|")
    ax_amp.legend(frameon=False, fontsize=7, ncol=2)
    trails = [ax_plane.plot([], [], color=colors[i], alpha=.8, lw=1.5)[0]
              for i in range(clusters)]
    heads = [ax_plane.plot([], [], "o", color=colors[i], ms=4)[0]
             for i in range(clusters)]
    reach = max(1.4, float(np.abs(z).max())*1.1)
    ax_plane.set(xlim=(-reach, reach), ylim=(-reach, reach), title="Complex plane: Re(z) vs Im(z)",
                 xlabel="Re(z)", ylabel="Im(z)", aspect="equal")

    def update(frame):
        for i, line in enumerate(amp_lines):
            line.set_data(times[:frame+1], group_means[:frame+1, i])
            members = np.asarray(group_ids[chosen] == i)
            x, y = sample[:frame+1, members].real, sample[:frame+1, members].imag
            trails[i].set_data(x.flatten(), y.flatten())
            heads[i].set_data(x[-1], y[-1])
        return [*amp_lines, *trails, *heads]

    animation = FuncAnimation(fig, update, frames=FRAME_COUNT, interval=1000/GIF_FPS, blit=False)
    animation.save(os.path.join(OUT, "stuart_landau_animation.gif"), writer=PillowWriter(fps=GIF_FPS))
    plt.close(fig)
    return {"times": times.tolist(), "real": sample.real.tolist(), "imag": sample.imag.tolist(),
            "groups": group_ids[chosen].tolist(), "amplitude": group_means.tolist()}


def save_ks():
    n, length = 200, 22.0
    params = create_ks_params(N=n, L=length)
    times_all, spectrum = integrate_ks(get_initial_conditions_ks(n), (0.0, 100.0), params,
                                       dt=0.001, save_every=50)
    field_all = np.real(np.fft.ifft(spectrum, axis=1))
    indices = np.linspace(0, len(times_all)-1, FRAME_COUNT, dtype=int)
    times = times_all[indices]
    field = field_all[indices, ::2]
    x = np.linspace(0, length, field.shape[1], endpoint=False)

    fig, ax = plt.subplots(figsize=(11.2, 6.3), dpi=144)
    fig.patch.set_facecolor("#f7f8f5")
    ax.set_facecolor("#f7f8f5")
    image = ax.imshow(np.full((FRAME_COUNT, len(x)), np.nan), origin="lower", aspect="auto",
                      extent=[0, length, times[0], times[-1]], cmap="RdBu_r",
                      vmin=float(field.min()), vmax=float(field.max()), interpolation="nearest")
    ax.set(title="A nonlinear wave field builds through space and time", xlabel="Position x", ylabel="Time")
    fig.colorbar(image, ax=ax, label="u(x, t)")

    def update(frame):
        image.set_data(np.where(np.arange(FRAME_COUNT)[:, None] <= frame, field, np.nan))
        return [image]

    animation = FuncAnimation(fig, update, frames=FRAME_COUNT, interval=1000/GIF_FPS, blit=False)
    animation.save(os.path.join(OUT, "ks_animation.gif"), writer=PillowWriter(fps=GIF_FPS))
    plt.close(fig)
    return {"times": times.tolist(), "x": x.tolist(), "field": field.tolist(),
            "range": [float(field.min()), float(field.max())]}


def main():
    dataset = {
        "kuramoto": save_kuramoto(),
        "stuartLandau": save_stuart_landau(),
        "kuramotoSivashinsky": save_ks(),
    }
    path = os.path.join(OUT, "oscillator_data.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, separators=(",", ":"))
    print(f"Wrote interactive trajectory data to {path}")


if __name__ == "__main__":
    main()
