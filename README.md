# Dynamics, Time Series & World Models

A research and simulation lab connecting **dynamical systems**, **time-series foundation models**, and **world-action models**. This repository pairs NumPy/Numba oscillator simulations with an interactive field guide to what models take in, what they predict, and what the evidence does (and does not) establish.

## Start with the interactive field notes

| Field note | Question | Explore |
|---|---|---|
| **Time-Series Foundation Models** | How can a model turn numerical history into a useful forecast on a new series? | [Read the paper study](https://dipayandey1244.github.io/time-series-foundation-model/time-series-paper-study.html) |
| **World-Action Models** | How can an agent connect incomplete observations, possible actions, and their consequences? | [Open the World Model Lab](https://dipayandey1244.github.io/time-series-foundation-model/kairos_time_series_lab.html) |

The [homepage](https://dipayandey1244.github.io/time-series-foundation-model/) introduces time-series, vision, and world-action models, then links the two research paths. The field notes use animated, step-by-step explanations. The equations below are compact maps of each field, not claims that every model shares one architecture.

## The equations behind the field notes

### 1. Time-series forecasting

$$
\hat{y}_{t+1:t+H} = f_\theta(y_{1:t})
$$

The observed context $y_{1:t}$ ends at time $t$. A learned forecasting function $f_\theta$, with parameters $\theta$, maps that history to estimates for the next $H$ time steps. Some foundation forecasters also represent uncertainty or return quantiles; this expression shows the basic history-to-horizon relationship.

### 2. Visiion Models

$$
\hat{M}_{1:T} = f_\theta(X_{1:T}, q)
$$

Here $X_{1:T}$ is an image or sequence of video frames, $q$ is a prompt specifying the object or concept, and $\hat{M}_{1:T}$ is the predicted segmentation mask sequence. A mask marks selected pixels; it is not itself a complete scene understanding.

### 3. World-action models

$$
(\hat{A}_{t:t+H}, \hat{O}_{t+1:t+H}) \sim
p_\theta(A_{t:t+H}, O_{t+1:t+H} \mid O_{1:t}, g)
$$

A model samples a candidate action sequence $\hat{A}$ together with future observations $\hat{O}$, conditioned on observation history $O_{1:t}$ and goal $g$. This is **schematic family-level notation**: particular world-action models predict different combinations of actions, states, video, or other outputs.

The common thread is prediction from context. A time-series model asks what values may come next; a world-action model asks what may happen under a possible action.

## Oscillator environments

The numerical examples are 200-state systems with structure intended to make collective behavior easier to see. Each oscillator remains an individual state; the lower-dimensional descriptions below are hypotheses about emergent group behavior, not a guarantee that every run lands on an exact low-dimensional manifold.

### Open the animated labs

Choose a model to open its interactive walkthrough. Each page combines the high-definition simulation, a scrub-through animation, the governing equation, and a sourced real-world research example.

| Interactive walkthrough | Research connection |
|---|---|
| [Open Multi-population Kuramoto →](https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#kuramoto) | Power-grid synchronization |
| [Open Cluster-synchronized Stuart–Landau →](https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#stuart-landau) | Neural population dynamics |
| [Open Kuramoto–Sivashinsky →](https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#kuramoto-sivashinsky) | Flame-front instabilities |

### 1. Multi-population Kuramoto model

[**▶ Open the animated Kuramoto walkthrough**](https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#kuramoto)

Implemented in [environment/kuramoto.py](environment/kuramoto.py).

$$
\dot{\theta}_i = \omega_i +
\sum_{j=1}^{N} K_{ij}\sin(\theta_j-\theta_i),
\qquad i=1,\ldots,N
$$

**Equation intuition.** Imagine each oscillator as a clock hand on a circle. The first term, $\omega_i$, is the speed that hand would keep on its own. The sum is the network tug: a neighbor ahead pulls it forward, one behind pulls it back, and $K_{ij}$ sets how strong that connection is. The sine makes the pull depend on angular difference and keeps it bounded. If the tugs are strong enough compared with differences in natural speed, phases can align; coupling does not guarantee that they will.

- $N=200$ phase oscillators are divided into five populations of 40.
- The coupling matrix is block-structured: intra-population coupling is stronger than inter-population coupling. The stored matrix is divided by $N$ before integration.
- Natural frequencies are drawn from a normal distribution; random initial phases spread oscillators around the circle.
- The population order parameter
  $$
  R_c(t)=\left|\frac{1}{N_c}\sum_{j\in c}e^{i\theta_j(t)}\right|
  $$
  summarizes synchrony within population $c$: values near 1 indicate aligned phases; lower values indicate dispersion.

[Open the Kuramoto result image](visualizations/kuramoto_results.png) · [simulation script](visualizations/vis_kuramoto.py)

<p align="center"><a href="https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#kuramoto"><img src="visualizations/kuramoto_animation.gif" alt="High-definition animated Kuramoto phase histories and synchrony curves; click to explore the simulation" width="100%"></a></p>

<p align="center"><a href="https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#kuramoto">Open the interactive Kuramoto lab: play, pause, scrub time, and inspect each population's synchrony.</a></p>

**What the animation tells us.** In the left panel, each row is one oscillator and color encodes its phase around the circle; vertical color bands mean many oscillators are at similar phases at the same time. In the right panel, each colored curve is $R_c(t)$ for one population. A rising curve toward 1 is direct evidence of within-group phase alignment in this run. Different groups can each synchronize while keeping different phases, so “synchronized” need not mean all 200 hands point in exactly the same direction.

**Why it is useful, and what it does not show.** The model isolates how network topology and coupling can create collective rhythms, which is why Kuramoto-like reductions are useful in power-system analysis. This particular plot is generated from synthetic frequencies, chosen couplings, and a fixed seed: it is not measured grid data, a stability certificate, or a prediction of a blackout. See the cited power-grid paper for the assumptions behind that application.

**Real-world connection:** Power-grid researchers have mapped generator and consumer dynamics to Kuramoto-like phase oscillators, with assumptions that make the frequency distribution bimodal. This is a model-analysis connection, not a claim that this five-group demonstration represents an operational grid. [Filatrella et al., *Analysis of a power grid using the Kuramoto-like model*](https://arxiv.org/abs/0705.1305).

<details>
<summary><strong>How to read and reproduce this figure</strong></summary>

The image has two views of the same trajectory:

1. **Build the communities.** Create the 200-by-200 block coupling matrix with five strongly coupled groups and weaker links between groups.
2. **Evolve the phases.** Draw natural frequencies and initial phases, then integrate the coupled ODE with the shared fourth-order Runge–Kutta integrator (dt=0.01, t=0…50). The visualization saves every 10 integration steps.
3. **Reveal the collective pattern.** The left panel plots each phase modulo $2\pi$. The right panel plots $R_c(t)$ for each population, so within-group synchronization can be compared directly.

The design encourages five coarse communities; inspect the order-parameter curves to see how strongly they synchronize in the actual run.

</details>

### 2. Cluster-synchronized Stuart–Landau network

[**▶ Open the animated Stuart–Landau walkthrough**](https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#stuart-landau)

Implemented in [environment/stuart_landau.py](environment/stuart_landau.py).

$$
\dot{z}_i = (\lambda_i + i\omega_i)z_i - |z_i|^2 z_i + \sum_{j=1}^{N} K_{ij}(z_j-z_i)
$$

The complex state $z_i$ encodes amplitude and phase. The linear term supports growth and rotation, the cubic term limits amplitude near a Hopf bifurcation, and coupling communicates state between oscillators.

**Equation intuition.** Write $z_i=r_i e^{i\phi_i}$: its distance $r_i$ from the origin is amplitude, and its angle $\phi_i$ is phase. The linear term makes a small signal grow at rate $\lambda_i$ while rotating at angular speed $\omega_i$. The negative cubic term grows stronger as $r_i$ grows, balancing that initial growth and producing a bounded oscillation. The final sum pulls each complex state toward connected neighbors. Unlike the phase-only Kuramoto model, this equation lets both amplitude and phase respond.

- $N=200$ oscillators are arranged in seven groups (six groups of 28 and one of 32).
- The parameters use $\lambda_i=1$ and heterogeneous frequencies centered near 2; the coupling matrix is block-structured and normalized by $N$.
- The visualization uses intra-group coupling 2.0 and inter-group coupling 0.01 (the environment function's default inter-group value is 0.05).
- The shared RK4 integrator uses dt=0.01 over t=0…50, saving every five steps.

[Open the Stuart–Landau result image](visualizations/stuart_landau_results.png) · [simulation script](visualizations/vis_stuart_landau.py)

<p align="center"><a href="https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#stuart-landau"><img src="visualizations/stuart_landau_animation.gif" alt="High-definition animation of Stuart–Landau amplitudes and complex-plane trajectories; click to explore" width="100%"></a></p>

<p align="center"><a href="https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#stuart-landau">Open the interactive Stuart–Landau lab: follow amplitude growth and rotating trajectories in the complex plane.</a></p>

**What the animation tells us.** The left panel tracks mean amplitude by cluster: growth away from zero followed by a plateau is the visual signature of nonlinear saturation. In the right panel, each trace is a trajectory in the complex plane $(\Re z,\Im z)$; a loop means the state is oscillating, while traces of the same color that approach similar loops suggest cluster-level similarity. The curves show this chosen simulation, not a measurement of brain activity.

**Why it is useful, and what it does not show.** Stuart–Landau is a canonical local model near the onset of oscillation, making it useful for studying how coupling or control changes collective rhythm. It is a reduced dynamical description, not a detailed neuron model. The cited neural-control result is a separate computational study; our 200-node animation does not reproduce its parameters, intervention, or conclusions.

**Real-world connection:** Stuart–Landau (Hopf) networks are used to model interacting neural populations. One computational proof-of-concept modeled 100 neuronal patches: when coupling rose, the uncontrolled network synchronized, while a simulated controller acting through 20 nodes kept it desynchronized. That study is not a clinical trial, and its controller is not part of our 200-oscillator animation. [PLOS Computational Biology study](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006296).

<details>
<summary><strong>How to read and reproduce this figure</strong></summary>

1. **Initialize seven groups.** Create the block coupling matrix and random complex starting states with small amplitudes.
2. **Integrate amplitude and phase together.** Run the coupled complex ODE with RK4; each point in the complex plane is one oscillator's $z_i(t)$.
3. **Read both projections.** The left panel plots $\operatorname{Re}(z_i)$ over time. The right panel traces $(\operatorname{Re}z_i,\operatorname{Im}z_i)$ for late-time samples, coloring oscillators by group. Similar loops within a color indicate cluster-like behavior.

</details>

### 3. Kuramoto–Sivashinsky equation

[**▶ Open the animated Kuramoto–Sivashinsky walkthrough**](https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#kuramoto-sivashinsky)

Implemented in [environment/kuramoto_sivashinsky.py](environment/kuramoto_sivashinsky.py).

$$
u_t + u\,u_x + u_{xx} + u_{xxxx}=0
$$

The nonlinear advection term competes with a destabilizing second derivative and stabilizing fourth-order hyper-diffusion, producing rich spatiotemporal dynamics. The solver uses $N=200$ Fourier grid points on a periodic domain of length $L=22$.

**Equation intuition.** Think of $u(x,t)$ as the height of a one-dimensional interface. The time derivative $u_t$ asks how that shape changes. The nonlinear transport $u u_x$ moves steep and shallow parts at different speeds, so waves distort one another. The $u_{xx}$ term amplifies some long-wave disturbances, while $u_{xxxx}$ damps very short scales; their competition can create and rearrange spatial structure instead of simply smoothing everything away. Periodic boundaries mean the left and right ends of the simulated domain join.

In Fourier space, the linear operator has symbol $k^2-k^4$. The code advances that linear part implicitly and the quadratic term explicitly (first-order IMEX Euler):

$$
\hat{u}^{\,n+1}_k =
\frac{\hat{u}^{\,n}_k
-\frac{\Delta t}{2} i k\,\widehat{u^2}^{\,n}_k}
{1-\Delta t(k^2-k^4)}.
$$

[Open the Kuramoto–Sivashinsky result image](visualizations/ks_results.png) · [simulation script](visualizations/vis_ks.py)

<p align="center"><a href="https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#kuramoto-sivashinsky"><img src="visualizations/ks_animation.gif" alt="High-definition animated Kuramoto–Sivashinsky field across space and time; click to explore" width="100%"></a></p>

<p align="center"><a href="https://raw.githack.com/gupta706/wam/main/oscillator_lab.html?v=20261011-fullplots#kuramoto-sivashinsky">Open the interactive Kuramoto–Sivashinsky lab: scrub through the evolving field and inspect wave formation.</a></p>

**What the animation tells us.** The heatmap is a space–time record: horizontal position is $x$, vertical position is elapsed time, and color is the value of $u$. Each new row adds the next saved numerical state. Bands or cells that bend, merge, or break indicate evolving spatial patterns; the displayed color range is shared across frames so color changes remain comparable through time.

**Why it is useful, and what it does not show.** The KS equation is a compact testbed for instability, nonlinear transport, and pattern formation, and related Sivashinsky equations appear in flame-front research. Our 1-D periodic field is a numerical solution of this model, not a flame photograph or an experimentally validated forecast. Resolution, time-step, domain, and initial-condition checks are needed before treating fine-scale features as robust. The advertised Lyapunov-dimension range is a target only; these scripts do not estimate it.

**Real-world connection:** Flame-front researchers compared experiments with solutions of the related Sivashinsky equation and reproduced cell creation and fusion during a deterministic stage of front evolution. Our periodic one-dimensional KS run is an equation demonstration, not flame-camera data. [Almarcha et al., *Interface dynamics, pole trajectories, and cell size statistics*](https://doi.org/10.1103/PhysRevE.98.030202).

<details>
<summary><strong>How to read and reproduce this figure</strong></summary>

1. **Set the periodic grid.** Choose $L=22$, $N=200$, and initialize small mean-zero noise in Fourier space.
2. **Advance the Fourier modes.** Use dt=0.001 over t=0…100; save every 50 steps. The linear Fourier term is handled implicitly while the nonlinear product is evaluated in physical space and transformed back.
3. **Return to physical space.** Apply the inverse FFT to each saved state. The heatmap shows $u(x,t)$: horizontal position is space, vertical position is time, and color is field value.

The project description targets a chaotic attractor with an estimated Lyapunov dimension around 5–10. The plotting script does **not** calculate Lyapunov dimension; treat that range as a modeling target until a dedicated estimate is added.

</details>

## Runtime and installation

The ordinary ODE loops use NumPy-compatible code and Numba's <code>@njit</code> compilation; this avoids requiring a JAX binary compatible with a particular CPU architecture. The Kuramoto–Sivashinsky solver uses NumPy FFTs and a Fourier-space update. Numba compilation can make the first run slower while the compiled integrator is prepared.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Generate the three static figures:

    python visualizations/vis_kuramoto.py
    python visualizations/vis_stuart_landau.py
    python visualizations/vis_ks.py

Each script writes a PNG into <code>visualizations/</code>. Click a result link above to inspect the full-resolution image; expand its walkthrough to see the simulation and plotting steps. For animated, interactive explanations of the foundation-model equations, use the [field notes](https://dipayandey1244.github.io/time-series-foundation-model/).

### Animated, interactive simulation lab

The GIFs above are generated from the same numerical trajectories as the static plots. To rebuild the animations and the compact data used by the interactive controls, run:

    python3 visualizations/build_animations.py

Then serve the repository root (for example, `python3 -m http.server`) and open [`oscillator_lab.html`](oscillator_lab.html), or open the interactive link above. The lab can play, pause, scrub, and change playback speed; charts and readouts update together from saved simulation frames. No browser-side simulation or external JavaScript library is required. Reduced-motion preferences are respected.

## Reading the claims carefully

- **Designed dimensions are not measured dimensions.** Five or seven clusters are structural choices; use synchrony statistics and trajectories to judge the realized dynamics.
- **The KS dimension estimate is not computed by the current scripts.** A Lyapunov-dimension claim needs a dedicated numerical estimator and convergence checks.
- **Integration choices are model-specific.** The two oscillator networks use fixed-step RK4; KS uses a first-order IMEX Euler step. Neither method makes accuracy automatic: verify time-step and resolution sensitivity for quantitative conclusions.
- **A foundation-model forecast is not a guarantee.** Evaluate on the target domain and compare with simple baselines.

## Repository map

    environment/
      kuramoto.py                 Multi-population phase oscillators
      stuart_landau.py            Coupled complex oscillators
      kuramoto_sivashinsky.py     Fourier pseudo-spectral KS solver
      integrators.py              Numba-compiled RK4 integrator
    visualizations/
      vis_kuramoto.py             Simulation and Kuramoto figure
      vis_stuart_landau.py        Simulation and Stuart–Landau figure
      vis_ks.py                   Simulation and KS figure
      build_animations.py         GIF and interactive trajectory-data generator
      *_results.png               Generated result figures
      *_animation.gif             Animated simulation previews embedded in README
      oscillator_data.json        Downsampled trajectories for the interactive lab
    oscillator_lab.html           Interactive play/pause and scrub-through lab
