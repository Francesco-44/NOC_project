# g1_sim — impianto MuJoCo dell'Unitree G1

MuJoCo al posto di Gazebo: in Gazebo il G1 non avrebbe alcuna **sorgente di
moto** senza scriverne una da zero — o una policy RL (che parla il contratto DDS
Unitree, non Gazebo) o un plugin che teletrasporta la base, cioè lo stesso
impianto cinematico che MuJoCo dà già fatto.

Derivato dal materiale del laboratorio CIHR (pacchetto ROS `policypilot`),
ridotto alla sola simulazione: Nav2, slam_toolbox e le mappe non servono, perché
questo stack pianifica **senza mappa a priori** con A\* a orizzonte mobile sulla
griglia gaussiana.

---

## Perché la modalità cinematica

In modalità cinematica (default) `mujoco_sim` integra la posa della base da
`/cmd_vel`:

```
x_{k+1}   = x_k   + (vx·cos(yaw_k) − vy·sin(yaw_k))·dt
y_{k+1}   = y_k   + (vx·sin(yaw_k) + vy·cos(yaw_k))·dt
yaw_{k+1} = yaw_k + wz·dt
```

che è **esattamente** il modello SE(2) olonomico su cui l'MPC ottimizza, con
costanti di tempo dell'attuatore nulle. Il disadattamento modello/impianto è
quindi nullo per costruzione.

È una scelta deliberata, non un ripiego: rende gli esperimenti di ottimizzazione
(iterazioni IPOPT, condizionamento, warm start, penalità esatta, active-set vs
interior-point — vedi [`guides/roadmap_teorica_noc.md`](../../guides/roadmap_teorica_noc.md))
misure del **solutore**, non del rumore dell'andatura.

La modalità `physics:=true` (camminata sotto fisica con la policy RL AMO sul bus
DDS Unitree) è presente nel codice ma **non usata**: richiede torch,
`unitree_sdk2py` e `cyclonedds` in un ambiente separato, e sul materiale di
origine la camminata non è verificata.

---

## Interfaccia ROS

| direzione | topic | tipo |
|---|---|---|
| IN | `/cmd_vel` | `geometry_msgs/Twist` (vx, vy, wz nel corpo) |
| OUT | `/odom` | `nav_msgs/Odometry` |
| OUT | `/livox/lidar` | `sensor_msgs/PointCloud2`, frame `mid360_link` |
| OUT | `/clock` | `rosgraph_msgs/Clock` |
| OUT | `/joint_states` | `sensor_msgs/JointState` |
| TF | `odom → base_link` | posa della base |
| TF | `odom → mid360_link` | presa dal site MuJoCo che genera i raggi |

La TF del sensore è pubblicata da questo nodo e non da `robot_state_publisher`:
la nuvola e la trasformazione sono così coerenti per costruzione, e non servono
l'URDF a 29 DoF né le sue mesh (~19 MB risparmiati).

Il Mid-360 è simulato con `mj_multiRay` dal site `mid360`, con ray-cast limitato
al gruppo geometrico dell'ambiente: **il robot non mappa sé stesso**, quindi in
simulazione il self-filtering non serve. Misurato: 8640 raggi in ~6 ms, cioè il
6 % di un core al rate di 10 Hz.

---

## Uso

### Solo impianto, guidato a mano

```bash
ros2 launch g1_sim g1_sim.launch.py
ros2 run g1_sim key_teleop          # in un secondo terminale
```

### Stack completo di navigazione autonoma

```bash
ros2 launch g1_sim g1_a_star_mpc.launch.py
```

Poi si manda un goal con lo strumento **2D Goal Pose** di RViz, oppure:

```bash
ros2 topic pub --once /global_goal geometry_msgs/PoseStamped \
  '{header: {frame_id: "odom"}, pose: {position: {x: 5.0, y: 3.0}}}'
```

Argomenti utili:

```bash
# ostacoli dinamici (3 persone di prova)
ros2 launch g1_sim g1_a_star_mpc.launch.py people:=default

# senza finestra MuJoCo e senza RViz (headless, per le campagne di misura)
ros2 launch g1_sim g1_a_star_mpc.launch.py viewer:=false use_rviz:=false

# memoria globale topologica
ros2 launch g1_sim g1_a_star_mpc.launch.py nav_graph:=true

# missione a waypoint ripetibile
ros2 launch g1_sim g1_a_star_mpc.launch.py use_mission:=true \
    mission_file:=/percorso/della/missione.yaml
```

---

## Contenuto

```
g1_sim/
├── g1_sim/
│   ├── mujoco_sim.py         nodo ROS: impianto + Mid-360 simulato + TF
│   ├── mujoco_world.py       costruzione del modello: G1 + magazzino + persone
│   ├── lowlevel_bridge.py    ponte DDS Unitree (solo physics:=true, non usato)
│   ├── key_teleop.py         guida da tastiera
│   └── cloud_self_filter.py  rimozione del rig di sostegno (serve sul robot reale)
├── assets/
│   ├── g1/g1_29dof_rev_1_0.xml + meshes/   MJCF del G1 (serve a MuJoCo)
│   └── industrial.sdf        geometria del magazzino, replicata in mujoco_world.py
├── config/
│   ├── g1_sim.yaml           parametri del simulatore
│   └── lidar_filter_g1.yaml  parametri dell'adattatore LiDAR
├── launch/
│   ├── g1_sim.launch.py      solo impianto
│   └── g1_a_star_mpc.launch.py   stack completo
└── rviz/g1_nav.rviz
```

## Requisiti

```bash
pip install mujoco        # verificato con 3.9.0
```
