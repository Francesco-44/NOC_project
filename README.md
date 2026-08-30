# Map-free navigation of the Unitree G1 — rolling-horizon A\* + nonlinear MPC

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros)](https://docs.ros.org/en/humble/) [![MuJoCo](https://img.shields.io/badge/MuJoCo-3.9-orange)](https://mujoco.org/) [![Robot](https://img.shields.io/badge/Robot-Unitree_G1-yellow)](https://www.unitree.com/g1/) [![Solver](https://img.shields.io/badge/CasADi-IPOPT-blue)](https://web.casadi.org/)

Autonomous navigation in an **unknown environment**, with no prior map, for the **Unitree G1**
humanoid. At every control cycle a LiDAR-based **Gaussian occupancy grid** is rebuilt around the
robot, a rolling-horizon **A\*** search produces a collision-free reference on it, and a
**nonlinear MPC** written in CasADi and solved by IPOPT turns that reference into body-frame
velocity commands.

The project is also the object of study: the deployed NLP is instrumented and measured — NLP
structure and sparsity, KKT conditions and multipliers, exact ℓ¹ penalty, automatic
differentiation against finite differences, horizon and control-horizon sweeps, Pareto front of
the scalarisation, interior point against active set, and the prediction error that drives the
robust constraint tightening. Every number in the report is generated from *the deployed
modules*, never re-implemented.

---

## What runs

```
mujoco_sim  --/odom-----------> odom_to_pose_node --/robot_pose--+
            --/livox/lidar--> lidar_filter_node                  |
                                  |                              |
                                  +--/lidar/points_filtered------+
                                                 |               |
                                                 v               v
                                           a_star_node -----> mpc_node
                                           /a_star/path    /mpc/next_setpoint
                                                                 |
                                                 setpoint_to_cmd_vel_node
                                                                 |
                                                            /cmd_vel --> mujoco_sim
```

The plant is **MuJoCo**: in Gazebo the G1 would have no source of motion without writing one from
scratch. The gain is methodological as much as practical — MuJoCo's kinematic plant *is* the model
the MPC optimises over, so the model/plant mismatch is nil and the experiments measure the solver
instead of the gait.

The robot enters the chain in exactly two places: a parameter file and the name of the pose topic.
No node of the algorithmic stack is platform-specific.

---

## Repository structure

```
NOC_project/
├── src/
│   ├── a_star_mpc_planner/     # A* on a Gaussian grid + nonlinear MPC (CasADi/IPOPT)
│   ├── g1_sim/                 # MuJoCo plant, G1 assets, worlds, missions, RViz, launch
│   ├── robot_real_lidar/       # point cloud: sensor frame → planning frame (range/height/voxel)
│   └── robot_real_goal_manager/# RViz /goal_pose relay + waypoint mission runner
├── viz/                        # offline measurement of the optimization problem → figures + LaTeX
├── guides/                     # runbook, theoretical roadmap, panel documentation
├── tests/                      # integrator truncation order
├── report_draft/               # LaTeX build tree of the report
└── tuning/, bag_gp_tuning/     # offline Bayesian weight tuning, non eseguito su questo profilo
```

---

## Requirements

| component | version in use |
|---|---|
| ROS 2 | **Humble** |
| MuJoCo | **3.9.0** (`pip install mujoco`; the G1 model uses `MjSpec`, needs ≥ 3.2) |
| Python | **3.10.12** (the ROS 2 Humble system interpreter) |
| CasADi | **3.7.2** |
| NumPy / SciPy / Matplotlib | 1.26.4 / 1.8.0 / 3.10.7 |

The `viz/` tools do **not** need ROS, except the ones that read a rosbag (`rosbag2_py`).
For the report PDF a TeX distribution is required — see [`guides/Recap.md`](guides/Recap.md) §0.

---

## Build

```bash
cd ~/NOC_project
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` matters: edits to Python files under `src/` then take effect without
rebuilding. A rebuild is only needed after touching `package.xml`, `setup.py`, or adding files.

---

## Quick start

```bash
ros2 launch g1_sim g1_a_star_mpc.launch.py
```

The MuJoCo viewer and RViz open. Send the goal with the **2D Goal Pose** tool in RViz: it publishes
on `/goal_pose`, a relay forwards it to `/global_goal`, which is what the stack listens to.

Useful arguments:

| argument | default | effect |
|---|---|---|
| `world:=<name>` | `industrial` | MuJoCo world geometry (also sets the spawn pose) |
| `people:=default` | `''` | dynamic obstacles (moving people) |
| `use_rviz:=false` | `true` | headless run |
| `viewer:=false` | `true` | no MuJoCo window (faster) |
| `nav_graph:=true` | `false` | topological global memory (Dijkstra) |
| `use_mission:=true mission_file:=<yaml>` | — | repeatable waypoint mission |
| `params_file:=<path>` | `planner_params_g1.yaml` | a different planner profile |
| `planner_overlay:=<path>` | `overlay_none.yaml` | a few parameters merged on top of the profile |

### Worlds

`industrial` is the warehouse. The others are the **non-convex obstacle** cases:
`long_wall`, `long_wall_south`, `long_wall_false_north`, `horseshoe`, `dead_end`, `l_corridor`,
`open_corridor`, `zigzag`, `door_room` — defined in `WORLDS` of
[`g1_sim/mujoco_world.py`](src/g1_sim/g1_sim/mujoco_world.py), with a matching
`config/mission_<world>.yaml` regenerated by `src/g1_sim/scripts/gen_missions.py`.

Concave worlds need the wider planning window:

```bash
ros2 launch g1_sim g1_a_star_mpc.launch.py world:=horseshoe \
  planner_overlay:=$(ros2 pkg prefix a_star_mpc_planner)/share/a_star_mpc_planner/config/overlay_nonconvex.yaml
```

The rationale — a local goal projected onto the window boundary points straight into the trap, and
the measured `grid_half_width` at which each world stops blocking — is documented at the top of
[`overlay_nonconvex.yaml`](src/a_star_mpc_planner/config/overlay_nonconvex.yaml).

---

## Record a run, extract the metrics, build the report

Full step-by-step runbook: **[`guides/Recap.md`](guides/Recap.md)**. In short:

```bash
./viz/record_run.sh <name>                  # BEFORE sending the goal (/global_goal fires once)
python3 viz/bag_source.py viz/bags/<name>   # sanity check: solver success must be ~99–100 %
python3 viz/make_results.py --bag viz/bags/<name>
```

`make_results.py` writes `viz/out/results.json`, `results.md` and the whole `viz/out/tex/` tree —
one LaTeX macro per scalar. The rule that holds this together: **no number in the report is ever
typed by hand.** One writes `$\resPredDivergence$` and the value follows the code.

Measurements are grouped by class, because the class decides when they must be redone:

- **class 1 — properties of the formulation**, profile-only: truncation order, NLP structure and
  sparsity, AD against finite differences, exact ℓ¹ penalty;
- **class 2 — properties of the instance**, needs a bag: KKT/LICQ/complementarity on the actual
  IPOPT multipliers, bifurcation threshold;
- **class 3 — closed-loop performance**, needs a bag: prediction error along the horizon and the
  constraint back-off β(k) measured from it.

---

## Diagnostics

```bash
ros2 topic echo /mpc/diagnostics
# data: [success(0/1), cost, solve_time_ms, avg_solve_ms, total_failures]
```

| Topic | Type | Meaning |
|---|---|---|
| `/a_star/occupancy_grid` | `OccupancyGrid` | Gaussian obstacle map |
| `/a_star/path` | `Path` | A\* waypoint path |
| `/mpc/predicted_path` | `Path` | MPC prediction over the horizon |
| `/mpc/next_setpoint` | `PoseStamped` | current lookahead setpoint |

---

## Component overview

| Package | Description |
|---|---|
| [`a_star_mpc_planner/`](src/a_star_mpc_planner/) | Rolling-horizon A\* on a Gaussian occupancy grid + nonlinear MPC tracker (CasADi/IPOPT), geodesic local-target metric and tabu memory for non-convex obstacles, optional topological navigation graph |
| [`g1_sim/`](src/g1_sim/) | MuJoCo plant of the G1 (29 DoF, simulated Mid-360 LiDAR), world geometries, missions, RViz config, and the top-level launch |
| [`robot_real_lidar/`](src/robot_real_lidar/) | Point-cloud adapter: range + height filter, voxel downsample, TF into the planning frame. Fully parametric — the robot enters through the YAML only |
| [`robot_real_goal_manager/`](src/robot_real_goal_manager/) | RViz `/goal_pose` → `/global_goal` relay + YAML-driven waypoint mission runner for repeatable trials |
| [`viz/`](viz/) | Offline measurement of the optimization problem: panels, sweeps, KKT analysis, and the LaTeX generator |

---

## Documentation

| Document | Content |
|---|---|
| [`guides/Recap.md`](guides/Recap.md) | End-to-end runbook: simulation → rosbag → metrics → report PDF |
| [`guides/roadmap_teorica_noc.md`](guides/roadmap_teorica_noc.md) | The theory behind each measurement, and what is still open |
| [`guides/visualizzazione_ottimizzazione.md`](guides/visualizzazione_ottimizzazione.md) | The two panels: cost landscape and decision space |
| [`viz/README.md`](viz/README.md) | The offline tools and what each one produces |
| [`guides/Todo_and_summary.md`](guides/Todo_and_summary.md) | Historical — refers to the earlier, robot-agnostic stack |

---

## Acknowledgements

- [CasADi](https://web.casadi.org/) — symbolic framework for nonlinear optimization
- [IPOPT](https://coin-or.github.io/Ipopt/) — interior-point NLP solver
- [MuJoCo](https://mujoco.org/) — the simulation plant
- [WildOS](https://github.com/nasa-jpl/nebula2-wildos) — inspiration for the topological navigation graph
