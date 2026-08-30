# Recap — running a mission end to end, from simulation to report PDF

Step-by-step runbook for the A\* + nonlinear MPC navigation stack on the Unitree G1.
Every command below is meant to be run from the repository root, `~/NOC_project`.

Related documents: [`roadmap_teorica_noc.md`](roadmap_teorica_noc.md) (the theory behind each
measurement), [`visualizzazione_ottimizzazione.md`](visualizzazione_ottimizzazione.md) (the two panels).

---

## 0. Requirements

### Simulation and stack

| component | version in use | notes |
|---|---|---|
| ROS 2 | **Humble** | `source /opt/ros/humble/setup.bash` |
| MuJoCo | **3.9.0** | `pip install mujoco` — the G1 model uses `MjSpec`, which needs ≥ 3.2 |
| Python | **3.10.12** | the ROS 2 Humble system interpreter |

### Offline analysis (`viz/`)

These do **not** need ROS, except for the tools that read a bag:

| package | version in use |
|---|---|
| CasADi | **3.7.2** |
| NumPy | **1.26.4** |
| SciPy | **1.8.0** |
| Matplotlib | **3.10.7** |
| PyYAML | any |

Reading a rosbag additionally needs `rosbag2_py`, which comes with the ROS 2 installation, so bag-based
tools must be run in a shell where ROS has been sourced.

### LaTeX → PDF

The report is compiled locally, so a TeX distribution is required. On Ubuntu:

```bash
sudo apt install latexmk texlive-latex-extra texlive-latex-recommended \
                 texlive-science texlive-fonts-recommended texlive-font-utils
```

`latexmk` drives the multi-pass compilation (LaTeX → BibTeX → LaTeX → LaTeX) so that cross-references and
citations resolve. `texlive-font-utils` provides **`epstopdf`**, which is required because the PoliMi title
page includes two `.eps` logos that `pdflatex` cannot embed directly.

---

## 1. Build and source the workspace

```bash
cd ~/NOC_project
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` matters: with it, edits to Python files under `src/` take effect without rebuilding.
A rebuild is only needed after changing `package.xml`, `setup.py`, or adding new files.

---

## 2. Launch the simulation (MuJoCo + RViz)

**Terminal A:**

```bash
cd ~/NOC_project
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch g1_sim g1_a_star_mpc.launch.py
```

Two windows open: the **MuJoCo viewer** and **RViz**. Wait until the G1 model appears in RViz without a
status error before doing anything else.

Useful arguments:

| argument | default | effect |
|---|---|---|
| `people:=default` | `''` | adds dynamic obstacles (moving people) |
| `use_rviz:=false` | `true` | headless run |
| `viewer:=false` | `true` | no MuJoCo window (faster) |
| `nav_graph:=true` | `false` | enables the topological global memory |
| `params_file:=<path>` | `planner_params_g1.yaml` | a different planner profile |

---

## 3. Start recording — **before** assigning the goal

**Terminal B:**

```bash
cd ~/NOC_project
source /opt/ros/humble/setup.bash
source install/setup.bash
./viz/record_run.sh industrial_plant_v4
```

The bag is written to `viz/bags/<name>/`. If the name is omitted, a timestamped one is used.

**Start the recorder before sending the goal.** Two things are lost otherwise: the single `/global_goal`
message, which is published once and never again, and the latched `/tf_static` transforms. Without them the
bag cannot be replayed by the analysis tools.

Recorded topics: `/robot_pose`, `/lidar/points_filtered`, `/a_star/path`, `/mpc/predicted_path`,
`/mpc/next_setpoint`, `/mpc/diagnostics`, `/global_goal`, `/cmd_vel`, `/odom`, `/tf`, `/tf_static`.
The occupancy grid is deliberately **not** recorded: it is large, and the panels recompute it from the cost
field anyway.

---

## 4. Assign the goal

In **RViz**, use the **2D Goal Pose** tool and click a target on the map.

The tool publishes on `/goal_pose`; a relay node forwards it to `/global_goal`, which is what the stack
listens to. The robot starts planning and walking immediately.

---

## 5. Stop the run

When the robot reaches the goal:

1. **Ctrl-C in Terminal B first** (the recorder), so the bag is closed cleanly.
2. Then Ctrl-C in Terminal A (the stack).

---

## 6. Check the bag before spending time on it

```bash
python3 viz/bag_source.py viz/bags/industrial_plant_v4
```

This prints the message count per topic and, most importantly, the MPC statistics:

```
cicli di controllo: 762  (430.3 s)
  successi: 100%
  solve_ms: media 100.6  p95 148.9  max 5991.8
  iterazioni IPOPT: media 12.1  max 32
  cicli con riferimento A*: 762/762
```

**Do not proceed unless the success rate is ~99–100 %.** Metrics extracted from a run with failed solver
cycles are meaningless — a latched fallback once produced a run at 30 % success, and every downstream number
was invalid.

Two things that look alarming but are not:

- a very large `solve_ms` **max** is normally cycle 0, which pays for CasADi code generation and NLP
  construction. Verify it is isolated: everything else should sit near the median;
- because that one sample inflates the mean, quote the **p95** in the report, not the mean.

---

## 7. Extract the metrics

### 7.1 Main generator

```bash
python3 viz/make_results.py --bag viz/bags/industrial_plant_v4
```

Runs three classes of measurement (~40 s) and writes `viz/out/results.json`, `viz/out/results.md` and the
whole `viz/out/tex/` tree.

> **This overwrites the previous results.** To keep an old campaign, add `--out viz/out_<name>`.

### 7.2 Satellite scripts that depend on the bag

```bash
python3 viz/robust_constraints.py --bag viz/bags/industrial_plant_v4 --no-show
python3 viz/solver_compare.py     --bag viz/bags/industrial_plant_v4
```

### 7.3 Satellite scripts that do **not** depend on the bag

These depend only on the planner profile. **Re-run them only if the tuning or the profile changed**, not
because a new bag was recorded:

```bash
python3 viz/horizon_sweep.py      # slowest, a few minutes
python3 viz/control_horizon.py
python3 viz/shooting_compare.py
python3 viz/pareto_front.py
```

---

## 8. Generate the figures

Six figures enter the report. Four are tied to the bag and must be regenerated for every new run.

```bash
# panel 1 — the navigation cost landscape
python3 viz/cost_field.py --bag viz/bags/industrial_plant_v4 --res 0.08

# panel 2 — the decision space, merit function (do NOT pass --objective here)
python3 viz/decision_plane.py --bag viz/bags/industrial_plant_v4 --no-show

# prediction error along the horizon
python3 viz/prediction_error.py viz/bags/industrial_plant_v4 --no-show
```

Two traps worth knowing:

- **`--res 0.08` on panel 1 is not optional on a small machine.** At the default `0.04` the grid times the
  LiDAR points can allocate ~5.5 GB and the process is killed by the OOM killer. At `0.08` the peak is
  ~1.4 GB, and nothing is lost visually because the Gaussian inflation is already smooth at 0.15 m.
- **panel 2 must be generated without `--objective`.** The report includes the *merit function* variant
  (`..._merit.pdf`); `--objective` writes `..._obj.pdf`, which the report does not pick up.

The remaining two figures (`bifurcation`, from `bifurcation_sweep.py`, and `horizon_sweep` / `pareto_front`)
come from the profile-only scripts of §7.3.

All figures are written to `viz/out/` as both `.png` and `.pdf`. The file names embed the bag name, so new
runs never overwrite old figures.

---

## 9. Regenerate the LaTeX

```bash
python3 viz/results_tex.py --check    # validate without writing
python3 viz/results_tex.py            # write
```

`make_results.py` already generates the LaTeX, so this step is needed **after** running any satellite script
of §7.2–7.3, to pull their JSON into the tables.

Output in `viz/out/tex/`:

| file | purpose |
|---|---|
| `metrics_macros.tex` | one macro per scalar — this is what makes the report self-updating |
| `metrics_body.tex` | sections and tables, to be `\input{}` into a report |
| `metrics_standalone.tex` | minimal wrapper to compile the metrics alone |
| `tab/*.tex` | one file per table, pulled in with `\restab{name}` |

The rule that holds this together: **no number is ever typed by hand in the report.** One writes
`$\resPredDivergence$` and the value follows the code.

---

## 10. Build the report PDF into `report_draft/`

### 10.1 Quick option — metrics only

```bash
cd viz/out/tex
latexmk -pdf metrics_standalone.tex
cp metrics_standalone.pdf ~/NOC_project/report_draft/Metrics_<bagname>.pdf
cd ~/NOC_project
```

A ~14-page document with every number and table, no figures. Good for checking a campaign.

### 10.2 Full report

The full report lives in a separate repository (`Relo02/NOC_report`, under `Latex_noc/`) because it needs
`Configuration_files/`, `bibliography.bib` and the PoliMi logos in `Images/`. A ready build tree is kept in
`report_draft/build_v3/`, so only the generated parts need refreshing:

```bash
B=~/NOC_project/report_draft/build_v3

# refresh macros and tables
cp viz/out/tex/metrics_macros.tex "$B"/Metrics/
cp viz/out/tex/tab/*.tex          "$B"/Metrics/tab/

# refresh figures, renaming to the stable names the report references
BAG=industrial_plant_v4
cp viz/out/errore_predizione_$BAG.pdf                   "$B"/Metrics/fig/prediction_error.pdf
cp viz/out/pannello1_bag_${BAG}_planner_params_g1.pdf   "$B"/Metrics/fig/cost_landscape.pdf
cp viz/out/pannello2_${BAG}_merit.pdf                   "$B"/Metrics/fig/decision_plane.pdf
cp viz/out/biforcazione_centred_pillar.pdf              "$B"/Metrics/fig/bifurcation.pdf
cp viz/out/horizon_sweep.pdf                            "$B"/Metrics/fig/horizon_sweep.pdf
cp viz/out/pareto_front.pdf                             "$B"/Metrics/fig/pareto_front.pdf

# compile
cd "$B"
latexmk -pdf -bibtex -interaction=nonstopmode Report_metrics_v3.tex
cp Report_metrics_v3.pdf ~/NOC_project/report_draft/Report_metrics_$BAG.pdf
cd ~/NOC_project
```

The figure renaming is deliberate: the report references **stable** names, so a new bag cannot silently
leave a stale figure in place. The identity of the run stays in the caption, through the `\resBag` macro.

After compiling, check the **last pass** rather than the accumulated log — the first pass always reports
undefined references:

```bash
grep -c "undefined" "$B"/Report_metrics_v3.log      # must be 0
```

---

# How the optimization metrics are extracted

*(summary of [`roadmap_teorica_noc.md`](roadmap_teorica_noc.md), §10)*

The project solves a **nonlinear finite-horizon optimal control problem** in receding horizon: 6 states,
3 inputs, N = 15 steps at dt = 0.20 s, giving **141 decision variables and 156 constraints**, built once
symbolically in CasADi and solved by IPOPT at every cycle. Every measurement below is produced by importing
*the deployed modules*, never a re-implementation — a visualization that draws a different function from the
one being optimized would be worthless.

Measurements fall into three classes, which is also how `make_results.py` is organised.

### Class 1 — properties of the formulation *(no bag needed)*

Depend only on the planner profile, so they are identical across runs and only need re-running when the
tuning changes.

- **Truncation order** (course §2.1.3): the Euler and mid-point schemes are integrated against a reference
  solution and the order is fitted on a grid of step sizes. Measured 1.00 and 2.00 exactly.
- **NLP structure and sparsity**: variable and constraint counts follow the closed form
  `n_var = 6(N+1) + 3N`, `n_con = 6N + 6 + 4N`, verified against the built problem.
- **AD against finite differences** (§5.2–5.3): the cost of one gradient is timed against one function
  evaluation, with a warm-up and a min-of-blocks estimator. Reverse-mode AD costs ~1.4 evaluations; finite
  differences would cost `n+1 = 142`.
- **Exact ℓ¹ penalty** (Thm 6.3.1): the slack of the relaxed obstacle constraint is driven to exactly zero
  once the penalty weight exceeds `max|μ*|`, and the ℓ² comparison shows the expected −1 slope.

### Class 2 — properties of the instance *(needs a bag)*

Evaluated on the hardest recorded cycle, selected among the *successful* ones.

- **KKT conditions** (§6.1): LICQ, strict complementarity and the second-order condition are checked on the
  actual multipliers returned by IPOPT. The residual is computed as `g − lbg`, because CasADi's `Opti`
  absorbs parameters into the bounds — reading `|g|` alone returns the state, not the residual.
- **Bifurcation threshold** (§4.4.5): the obstacle weight is swept until the left/right solutions separate,
  locating the value at which the solution loses regularity with respect to the parameter.

### Class 3 — closed-loop performance *(needs a bag)*

- **Prediction error** (§7.2.5): the trajectory predicted at time *t* is compared with the poses actually
  reached at *t + kΔt*. The residual at k = 0 is time misalignment, not model error, and is subtracted. The
  remaining divergence is the mismatch between a planar model and a 29-DoF humanoid walking — and it is more
  than an order of magnitude larger than the integration error, which is why upgrading the integrator does
  not help the closed loop.
- **Constraint tightening** (§7.2.5): the back-off β(k) is *measured* from that same error distribution
  (95th percentile, monotone, zero at k = 0) rather than guessed, and then imposed on the clearance
  constraint.

### Satellite studies

Each writes its own JSON, collected by `results_tex.py` if present: horizon sweep over N × dt, Pareto front
of the multi-objective scalarisation, interior point against active set, single against multiple shooting,
control horizon `N_c < N`, and robust constraints.

### Two invariants

1. **The deployed behaviour never changes when a study is added.** Every alternative formulation is an
   option in `MPCConfig` defaulting to the deployed choice, and a regression check confirms the solve on a
   real cycle returns a bit-identical `J*`.
2. **Claims are conditional on the data.** The LaTeX generator refuses to assert a conclusion the numbers do
   not support: if the ℓ² slope does not reach −1, or the sweep shows no degrading horizon, the text says so
   instead of printing the expected result.
