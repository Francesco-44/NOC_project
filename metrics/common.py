"""
Infrastruttura condivisa dai due pannelli di visualizzazione.

Regola di progetto: **il costo non viene mai reimplementato a mano**. Il termine
di ostacolo e' replicato riga per riga da MPCTracker._build_nlp (ed e' verificato
da metrics/test_fidelity.py), e il costo completo del pannello 2 e' estratto
direttamente dall'espressione CasADi che IPOPT minimizza. Una visualizzazione
che disegna una funzione diversa da quella ottimizzata non serve a niente.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src", "a_star_mpc_planner"))

from a_star_mpc_planner.mpc_tracker import MPCConfig, MPCTracker  # noqa: E402
from a_star_mpc_planner.a_star_planner import AStarPlanner  # noqa: E402
from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap  # noqa: E402

_HERE_COMMON = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROFILE = os.path.join(
    _REPO, "src", "a_star_mpc_planner", "config", "planner_params_g1.yaml")


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
def load_profile(path: str = DEFAULT_PROFILE,
                 overrides: list[str] | None = None) -> tuple[MPCConfig, dict]:
    """
    Legge un planner_params*.yaml e ne ricava un MPCConfig.

    `overrides` e' una lista "chiave=valore" con le chiavi del YAML, per gli
    studi parametrici senza duplicare il file (es. mpc_W_obs_sigmoid=600).
    """
    raw = yaml.safe_load(open(path))["/**"]["ros__parameters"]
    for item in overrides or []:
        k, _, v = item.partition("=")
        k = k.strip()
        if k not in raw:
            raise SystemExit(f"parametro sconosciuto: {k}")
        raw[k] = yaml.safe_load(v)
    cfg = MPCConfig(
        N=int(raw["mpc_N"]), dt=float(raw["mpc_dt"]),
        tau_v=float(raw["mpc_tau_v"]), tau_w=float(raw["mpc_tau_w"]),
        vx_max=float(raw["mpc_vx_max"]), vy_max=float(raw["mpc_vy_max"]),
        vx_min=float(raw.get("mpc_vx_min", 0.0)),
        omega_max=float(raw["mpc_omega_max"]), v_ref=float(raw["mpc_v_ref"]),
        Q_x=float(raw["mpc_Q_x"]), Q_y=float(raw["mpc_Q_y"]),
        Q_yaw=float(raw["mpc_Q_yaw"]), Q_terminal=float(raw["mpc_Q_terminal"]),
        R_vx=float(raw["mpc_R_vx"]), R_vy=float(raw["mpc_R_vy"]),
        R_omega=float(raw["mpc_R_omega"]), R_jerk=float(raw["mpc_R_jerk"]),
        W_obs_sigmoid=float(raw["mpc_W_obs_sigmoid"]),
        obs_alpha=float(raw["mpc_obs_alpha"]), obs_r=float(raw["mpc_obs_r"]),
        max_obs_constraints=int(raw["mpc_max_obs_constraints"]),
        obs_check_radius=float(raw["mpc_obs_check_radius"]),
        max_iter=int(raw["mpc_max_iter"]), warm_start=bool(raw["mpc_warm_start"]),
        integrator=str(raw.get("mpc_integrator", "euler")),
        N_c=(int(raw["mpc_N_c"]) if raw.get("mpc_N_c") is not None else None),
        path_mode=str(raw.get("mpc_path_mode", "time")),
        theta_progress_weight=float(raw.get("mpc_theta_progress_weight", 50.0)),
        terminal_constraint=str(raw.get("mpc_terminal_constraint", "none")),
        terminal_rho=float(raw.get("mpc_terminal_rho", 5.0e3)),
    )
    return cfg, raw


# ---------------------------------------------------------------------------
# Micro-benchmark
# ---------------------------------------------------------------------------
def time_call(fn, repeats: int = 200, blocks: int = 5, warmup: int = 20) -> float:
    """
    Tempo per chiamata [s], robusto al rumore.

    Un singolo ciclo cronometrato da' misure inaffidabili: la prima chiamata
    paga allocazioni e cache fredde, e lo scheduler introduce code lunghe. Con
    la media si e' arrivati a misurare un gradiente AD piu' veloce di una
    valutazione della funzione — un rapporto impossibile.

    Si scarta quindi un warm-up e si prende il MINIMO fra piu' blocchi: il
    minimo e' lo stimatore giusto per un tempo di calcolo, perche' il rumore
    puo' solo rallentare, mai accelerare.
    """
    import time as _t
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(blocks):
        t0 = _t.perf_counter()
        for _ in range(repeats):
            fn()
        best = min(best, (_t.perf_counter() - t0) / repeats)
    return best


# ---------------------------------------------------------------------------
# Termine di ostacolo — replica ESATTA di MPCTracker._build_nlp
# ---------------------------------------------------------------------------
def _blocchi(m: int, k: int, byte_max: float = 64e6):
    """Righe di P da trattare per volta perche' il temporaneo (blocco, K, 2)
    resti sotto byte_max.

    Serve perche' P[:, None, :] - obs[None, :, :] materializza un array (M, K, 2)
    INTERO prima di ridurlo: con la griglia del pannello 1 a res 0.04 su una bag
    del magazzino (M = 221806, K = 954) sono 3.4 GB per temporaneo, e la catena
    diff -> **2 -> sum ne tiene tre vivi insieme. Su una macchina da 7 GB
    l'OOM killer interviene (misurato: 6.05 GB di anon-rss prima del SIGKILL).
    A blocchi il risultato e' IDENTICO — la riduzione e' per riga di P, quindi
    separabile — e la memoria e' limitata a prescindere dalla griglia."""
    n = max(1, int(byte_max / max(1.0, k * 2 * 8)))
    return range(0, m, n), n


def obstacle_cost(P: np.ndarray, obs: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """
    Barriera ibrida sigmoide + hinge quadratica, valutata su un insieme di punti.

        J_obs(p) = W * [ 0.5*(1 - tanh(0.5*alpha*(d - r))) + 2*max(0, r - d)^2 ]

    con d = sqrt(dx^2 + dy^2 + 1e-6), esattamente come nell'NLP (l'epsilon serve
    a rendere la radice differenziabile nell'origine e va replicato).

    P   : (M, 2) punti     obs : (K, 2) ostacoli     ->  (M,) costi
    """
    P = np.atleast_2d(P)
    if obs is None or len(obs) == 0:
        return np.zeros(len(P))
    obs = np.atleast_2d(obs)
    out = np.empty(len(P))
    starts, n = _blocchi(len(P), len(obs))
    for i in starts:
        Pb = P[i:i + n]
        d = np.sqrt(((Pb[:, None, :] - obs[None, :, :]) ** 2).sum(-1) + 1e-6)
        s = cfg.obs_alpha * (d - cfg.obs_r)
        j = cfg.W_obs_sigmoid * 0.5 * (1.0 - np.tanh(0.5 * s))
        j += cfg.W_obs_sigmoid * 2.0 * np.maximum(0.0, cfg.obs_r - d) ** 2
        out[i:i + n] = j.sum(1)
    return out


def tracking_cost(P: np.ndarray, path: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """
    Costo di inseguimento ridotto alla posizione: per ogni punto, l'errore
    quadratico pesato rispetto al waypoint piu' vicino del riferimento.

    E' la restrizione alla posizione del termine ||x - x_ref||^2_Q dell'MPC:
    fedele, a differenza di una attrazione inventata verso il goal.
    """
    P = np.atleast_2d(P)
    path = np.atleast_2d(path)[:, :2]
    w = np.array([cfg.Q_x, cfg.Q_y])
    out = np.empty(len(P))
    starts, n = _blocchi(len(P), len(path))
    for i in starts:
        diff = P[i:i + n, None, :] - path[None, :, :]
        out[i:i + n] = ((diff ** 2) * w).sum(-1).min(1)
    return out


def goal_cost(P: np.ndarray, goal: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """Attrazione quadratica verso il goal, pesata come il tracking."""
    P = np.atleast_2d(P)
    w = np.array([cfg.Q_x, cfg.Q_y])
    return (((P - np.asarray(goal)[:2]) ** 2) * w).sum(1)


# ---------------------------------------------------------------------------
# Insieme raggiungibile: il termine di manovra NON si somma, definisce il dominio
# ---------------------------------------------------------------------------
def reach_time(P: np.ndarray, pose: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """
    Tempo minimo per raggiungere ogni punto con la politica "ruota, poi avanza",
    sotto i limiti di U_Sigma. Per il G1, che non puo' indietreggiare
    (vx >= 0 e' imposto nell'NLP) ne' traslare, questo rende visibile
    l'asimmetria dell'insieme ammissibile.
    """
    P = np.atleast_2d(P)
    rel = P - pose[:2]
    dist = np.linalg.norm(rel, axis=1)
    bearing = np.arctan2(rel[:, 1], rel[:, 0])
    dpsi = np.abs(np.arctan2(np.sin(bearing - pose[2]), np.cos(bearing - pose[2])))
    return dpsi / max(cfg.omega_max, 1e-9) + dist / max(cfg.vx_max, 1e-9)


def reachable_mask(P, pose, cfg: MPCConfig) -> np.ndarray:
    """True dove il punto e' raggiungibile entro un orizzonte."""
    return reach_time(P, pose, cfg) <= cfg.N * cfg.dt


# ---------------------------------------------------------------------------
# Scenari
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    pose: np.ndarray                      # (3,) [x, y, yaw]
    obstacles: np.ndarray                 # (K, 2)
    goal: np.ndarray                      # (2,)
    path: np.ndarray = field(default=None)  # (M, 2) riferimento A*; None -> retta
    extent: tuple = (-1.5, 5.0, -2.5, 2.5)

    def reference(self) -> np.ndarray:
        """Riferimento geometrico: il path A* se c'e', altrimenti la retta al goal."""
        if self.path is not None:
            return np.atleast_2d(self.path)[:, :2]
        n = 40
        t = np.linspace(0.0, 1.0, n)[:, None]
        return self.pose[:2] + t * (self.goal - self.pose[:2])


def _wall(p0, p1, spacing=0.12):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.linalg.norm(p1 - p0) / spacing) + 1)
    return p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0)


SCENARIOS = {}


def _reg(fn):
    SCENARIOS[fn.__name__] = fn
    return fn


@_reg
def u_trap() -> Scenario:
    """Ostacolo concavo aperto verso il robot: la trappola classica dei campi di
    potenziale. Il goal sta oltre il fondo della U."""
    obs = np.vstack([_wall((2.4, -1.2), (2.4, 1.2)),
                     _wall((1.2, 1.2), (2.4, 1.2)),
                     _wall((1.2, -1.2), (2.4, -1.2))])
    return Scenario("u_trap", np.array([0.0, 0.0, 0.0]), obs, np.array([4.0, 0.0]))


@_reg
def centred_pillar() -> Scenario:
    """
    Un pilastro esattamente sulla retta verso il goal: e' il caso in cui il costo
    DOVREBBE avere due minimi (passo a sinistra / passo a destra).

    Il pilastro sta a 0.55 m perche' l'orizzonte del profilo G1 era N=15,
    dt=0.20, v_ref=0.2, cioe' 0.60 m di percorso (0.9 m a vx_max): un ostacolo
    oltre quella distanza era semplicemente FUORI dall'orizzonte, la barriera
    non lo vedeva, e il paesaggio non biforcava per un motivo che non ha niente
    a che fare con il peso della barriera.

    [SIM] Col profilo attuale (dt = 0.35) l'orizzonte copre 1.05 m, quindi 0.55 m
    e' ora ben dentro e il test resta valido — ma non e' piu' AL LIMITE, che era
    il suo scopo. Il pilastro NON e' stato spostato di proposito: la figura
    biforcazione_centred_pillar entra nel report tramite results_tex.py, e
    cambiare la geometria dello scenario invaliderebbe silenziosamente il
    confronto con le versioni precedenti. Per ritarare la prova al nuovo
    orizzonte va spostato a ~0.95 m, rigenerando la figura.
    """
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    obs = np.stack([0.55 + 0.14 * np.cos(th), 0.14 * np.sin(th)], 1)
    return Scenario("centred_pillar", np.array([0.0, 0.0, 0.0]), obs,
                    np.array([2.5, 0.0]), extent=(-0.8, 3.0, -1.6, 1.6))


@_reg
def narrow_gap() -> Scenario:
    """Varco stretto fra due ostacoli: controprova, il minimo deve stare in mezzo."""
    obs = np.vstack([_wall((1.2, 0.45), (1.2, 2.0)),
                     _wall((1.2, -2.0), (1.2, -0.45))])
    return Scenario("narrow_gap", np.array([0.0, 0.0, 0.0]), obs,
                    np.array([3.0, 0.0]), extent=(-1.0, 3.5, -2.2, 2.2))


@_reg
def corridor() -> Scenario:
    """Corridoio con un pilastro sfalsato: caso realistico da magazzino."""
    obs = np.vstack([_wall((-0.5, 1.1), (4.0, 1.1)),
                     _wall((-0.5, -1.1), (4.0, -1.1)),
                     _wall((1.8, -0.35), (1.8, 0.35))])
    return Scenario("corridor", np.array([0.0, 0.0, 0.0]), obs,
                    np.array([3.6, 0.0]), extent=(-0.8, 4.2, -1.6, 1.6))


def _geom_footprint(ge, spacing=0.12, z_band=(0.15, 1.60)):
    """Impronta 2D di un geom di mujoco_world, campionata in punti.

    Si tiene solo cio' che interseca la fascia di quota che il filtro LiDAR
    lascia passare (z_min/z_max di lidar_filter_g1.yaml): un geom tutto sopra o
    tutto sotto quella fascia non arriva mai al pianificatore, e includerlo qui
    renderebbe l'harness piu' pessimista del sistema vero.

    Si campiona il PERIMETRO e non l'area: il LiDAR vede le superfici, non
    l'interno, e una nuvola piena falserebbe sia la griglia gaussiana di A* sia
    la barriera dell'MPC (che conta i punti piu' vicini).
    """
    # group 0 = decorazione (marcatori di goal/spawn): mujoco_world la tiene
    # fuori da LIDAR_GROUP, quindi il ray-cast non la vede e il pianificatore
    # nemmeno. Escluderla qui e' cio' che rende l'harness coerente col simulatore.
    if ge.get("group") is not None and int(ge["group"]) == 0:
        return np.zeros((0, 2))

    cx, cy, cz = ge["pos"]
    sz = ge["size"][-1]
    if cz + sz < z_band[0] or cz - sz > z_band[1]:
        return np.zeros((0, 2))

    if ge["shape"] == "cyl":
        r = ge["size"][0]
        n = max(8, int(2 * np.pi * r / spacing))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1)

    hx, hy = ge["size"][0], ge["size"][1]
    corners = np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])
    yaw = ge.get("yaw", 0.0)
    c, s_ = np.cos(yaw), np.sin(yaw)
    corners = corners @ np.array([[c, s_], [-s_, c]])
    pts = []
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        n = max(2, int(np.linalg.norm(p1 - p0) / spacing) + 1)
        pts.append(p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0))
    return np.vstack(pts) + np.array([cx, cy])


def world_scenario(name: str, spacing: float = 0.12) -> Scenario:
    """Uno Scenario dell'harness a partire da un mondo MuJoCo di g1_sim.

    Serve a provare i mondi non convessi (long_wall, horseshoe, dead_end) senza
    far girare MuJoCo, ROS e il LiDAR: stessa geometria, stessi spawn e goal.

    ATTENZIONE alla differenza di fedelta': qui gli ostacoli sono NOTI PER
    INTERO fin dal primo ciclo, mentre nel sistema vero il robot ne vede solo
    la porzione entro max_lidar_range (8 m) e in linea di vista, integrata da
    `_persistent_map` di a_star_node. Su un ostacolo concavo la differenza NON
    e' trascurabile: qui A* sa gia' che la U e' chiusa, sul robot deve
    scoprirlo. Un esito positivo qui e' quindi condizione NECESSARIA, non
    sufficiente — se fallisce anche con informazione perfetta, sul robot e'
    senza speranza.
    """
    import os
    import sys as _sys
    _pkg = os.path.join(os.path.dirname(_HERE_COMMON), "src", "g1_sim")
    if _pkg not in _sys.path:
        _sys.path.insert(0, _pkg)
    from g1_sim.mujoco_world import world_info

    info = world_info(name)
    chunks = [_geom_footprint(ge, spacing) for ge in info["geoms"]()]
    chunks = [c for c in chunks if len(c)]
    obs = np.vstack(chunks) if chunks else np.zeros((0, 2))
    sx, sy, syaw = info["spawn"]
    gx, gy = info["goal"]
    m = 1.5
    ext = (min(sx, gx, obs[:, 0].min()) - m, max(sx, gx, obs[:, 0].max()) + m,
           min(sy, gy, obs[:, 1].min()) - m, max(sy, gy, obs[:, 1].max()) + m)
    return Scenario(name, np.array([sx, sy, syaw]), obs,
                    np.array([gx, gy]), None, ext)


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise SystemExit(f"scenario sconosciuto: {name}. "
                         f"Disponibili: {', '.join(sorted(SCENARIOS))}")
    return SCENARIOS[name]()



# ---------------------------------------------------------------------------
# Riferimento globale: A* VERO, non una retta
# ---------------------------------------------------------------------------
def plan_astar(pose, goal, obstacles, raw: dict):
    """
    Riferimento calcolato con il pianificatore del repo (stessa griglia
    gaussiana, stessa A*), non con una retta. Senza questo, la visualizzazione
    mostrerebbe un uomo di paglia: l'MPC da solo, con un riferimento che
    attraversa gli ostacoli, non ha alcuna possibilita' di evitarli.
    """
    grid = FixedGaussianGridMap(reso=float(raw["grid_reso"]),
                                half_width=float(raw["grid_half_width"]),
                                std=float(raw["grid_std"]))
    pts = np.hstack([np.atleast_2d(obstacles),
                     np.zeros((len(obstacles), 1))]) if len(obstacles) else None
    grid.update(pts, np.asarray(pose[:2]))
    planner = AStarPlanner(obstacle_threshold=float(raw["obstacle_threshold"]),
                           obstacle_cost_weight=float(raw["obstacle_cost_weight"]))
    path = planner.plan(grid, np.asarray(pose[:2]), np.asarray(goal[:2]))
    return None if not path else np.asarray(path, dtype=float)[:, :2]


# Raggio d'ingombro del G1 [m] (planner_params_g1.yaml: "ingombro del robot
# ~0.35 m", da cui obs_r = 0.40).
BODY_RADIUS = 0.35
# Sotto questa distanza il centro del robot e' cosi' vicino alla SUPERFICIE
# campionata dell'ostacolo che la traiettoria l'ha attraversata: i punti sono
# campionati ogni 0.12 m, quindi 0.15 m e' dentro lo spessore del muro.
PENETRATION_EPS = 0.15


def check_collisions(traj_xy, obstacles) -> dict:
    """Verifica geometrica che una traiettoria dell'harness sia percorribile.

    SERVE PERCHE' `closed_loop` NON HA COLLISIONI: l'impianto integra il comando
    e basta, esattamente come mujoco_sim in modalita' cinematica (i geom hanno
    contype=conaffinity=0). Gli ostacoli agiscono solo come celle bloccate in A*
    e come penalita' MORBIDA nel costo dell'MPC, e nessuna delle due e' un
    vincolo rigido. Quando il ramo di ripiego di closed_loop punta l'ultimo
    waypoint di A* con un controllore proporzionale, quel proporzionale ignora
    gli ostacoli e il robot puo' passare DENTRO un muro arrivando al goal.

    Un "goal raggiunto" senza questo controllo non vuol dire niente: va sempre
    letto insieme a `attraversamento`.
    """
    d = clearance(traj_xy, obstacles)
    return {"clearance": d,
            "attraversamento": bool(d < PENETRATION_EPS),
            "contatto": bool(d < BODY_RADIUS)}


def clearance(traj_xy, obstacles) -> float:
    """Distanza minima fra la traiettoria percorsa e il piu' vicino ostacolo."""
    if obstacles is None or len(obstacles) == 0:
        return float("inf")
    d = np.linalg.norm(np.atleast_2d(traj_xy)[:, None, :]
                       - np.atleast_2d(obstacles)[None, :, :], axis=2)
    return float(d.min())


# ---------------------------------------------------------------------------
# Rollout dell'MPC sullo scenario
# ---------------------------------------------------------------------------
def make_tracker(cfg: MPCConfig, record_iterates: bool = False) -> MPCTracker:
    t = MPCTracker(cfg)
    if record_iterates:
        t.cfg.record_iterates = True
    return t


def solve_at(tracker: MPCTracker, pose: np.ndarray, sc: Scenario):
    """Un solve dell'MPC nella posa data, sul riferimento dello scenario."""
    state = np.array([pose[0], pose[1], pose[2], 0.0, 0.0, 0.0])
    ref = sc.reference()
    # il riferimento parte dal waypoint piu' vicino, come fa il nodo
    i = int(np.argmin(np.linalg.norm(ref - pose[:2], axis=1)))
    return tracker.solve(state, [tuple(p) for p in ref[i:]], sc.obstacles)


def closed_loop(tracker: MPCTracker, sc: Scenario, steps: int = 60,
                lookahead: float = None, kp: float = None, kp_yaw: float = None,
                raw: dict = None, replan_every: int = 5):
    """
    Simula l'anello chiuso come sul robot: l'MPC pubblica un setpoint a
    `lookahead` metri, un controllore proporzionale lo insegue, l'impianto e'
    lo stesso modello cinematico di mujoco_sim.

    lookahead/kp/kp_yaw a None (default) li prende dal PROFILO, rispettivamente
    da mpc_lookahead_dist, cmd_kp_xy, cmd_kp_yaw. Erano costanti fisse
    (0.9/1.0/1.5) e il valore 0.9 non era quello deployato (0.45): con un
    orizzonte che copre v_ref*N*dt = 0.60 m la predizione non raggiungeva MAI
    il lookahead, il ramo di fallback scattava sul 100% dei cicli e il setpoint
    veniva preso dall'ultimo waypoint di A*. In quelle condizioni l'uscita
    dell'MPC non entrava in anello chiuso e ogni sweep su N/dt misurava il
    passo dell'impianto, non l'orizzonte.
    """
    cfg = tracker.cfg
    if lookahead is None:
        lookahead = float((raw or {}).get("mpc_lookahead_dist", 0.9))
    if kp is None:
        kp = float((raw or {}).get("cmd_kp_xy", 1.0))
    if kp_yaw is None:
        kp_yaw = float((raw or {}).get("cmd_kp_yaw", 1.5))
    pose = sc.pose.astype(float).copy()
    hist = {"pose": [], "cost": [], "pred": [], "solve_ms": [], "success": [],
            "ref": [], "wz": []}
    ref = None
    for step in range(steps):
        if raw is not None and step % replan_every == 0:
            # orizzonte mobile: A* viene rilanciato periodicamente, come nel nodo
            new = plan_astar(pose, sc.goal, sc.obstacles, raw)
            if new is not None and len(new) >= 2:
                ref = new
        sc_step = sc if ref is None else Scenario(
            sc.name, pose, sc.obstacles, sc.goal, ref, sc.extent)
        res = solve_at(tracker, pose, sc_step)
        pred = res.predicted_xy
        # Selezione del setpoint, FEDELE a mpc_node: si cerca il primo nodo
        # predetto oltre `lookahead`; se l'orizzonte non ci arriva si ripiega
        # sull'ultimo waypoint di A*, puntandolo (non tenendo lo yaw corrente).
        d = np.linalg.norm(pred - pose[:2], axis=1)
        hit = np.nonzero(d >= lookahead)[0]
        if hit.size:
            tgt = pred[hit[0]]
            tgt_yaw = float(res.predicted_yaw[hit[0]])
        else:
            ref_i = sc_step.reference()
            tgt = np.asarray(ref_i[-1][:2], dtype=float)
            dv = tgt - pose[:2]
            tgt_yaw = (float(np.arctan2(dv[1], dv[0]))
                       if np.linalg.norm(dv) > 1e-6 else pose[2])
        hist["pose"].append(pose.copy()); hist["cost"].append(res.cost)
        hist["pred"].append(pred.copy()); hist["solve_ms"].append(res.solve_time_ms)
        hist["success"].append(res.success)
        hist["ref"].append(None if ref is None else ref.copy())
        # controllore proporzionale in corpo + saturazioni di U_Sigma
        e = tgt - pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        ex, ey = c * e[0] + s * e[1], -s * e[0] + c * e[1]
        # il nodo insegue l'ORIENTAMENTO del setpoint, non la direzione verso di esso
        eyaw = np.arctan2(np.sin(tgt_yaw - pose[2]), np.cos(tgt_yaw - pose[2]))
        vx = np.clip(kp * ex, min(cfg.vx_min, 0.0), cfg.vx_max)
        vy = np.clip(kp * ey, -cfg.vy_max, cfg.vy_max)
        wz = np.clip(kp_yaw * eyaw, -cfg.omega_max, cfg.omega_max)
        # impianto: identico a mujoco_sim in modalita' cinematica
        pose[0] += (vx * c - vy * s) * cfg.dt
        pose[1] += (vx * s + vy * c) * cfg.dt
        hist["wz"].append(wz)
        pose[2] = np.arctan2(np.sin(pose[2] + wz * cfg.dt), np.cos(pose[2] + wz * cfg.dt))
        if np.linalg.norm(pose[:2] - sc.goal) < 0.3:
            break
    for k in hist:
        hist[k] = np.array(hist[k], dtype=object if k in ("pred", "ref") else float)
    return hist


# ---------------------------------------------------------------------------
# Ambiente: matplotlib doppia
# ---------------------------------------------------------------------------
def ensure_mpl3d():
    """
    Su questa macchina convivono due matplotlib: 3.10.7 in ~/.local e 3.5.1 di
    sistema. `mpl_toolkits` di sistema e' un pacchetto REGOLARE, e per le regole
    di import di Python un pacchetto regolare trovato piu' avanti nel sys.path
    batte una porzione di namespace trovata prima: quindi `mpl_toolkits.mplot3d`
    viene risolto dalla 3.5.1 e fallisce contro l'API della 3.10
    (`cannot import name 'docstring'`).

    Qui si forza la risoluzione accanto alla matplotlib effettivamente attiva.
    La correzione vera e' ripulire l'ambiente, ma il tool non deve dipenderne.
    """
    import matplotlib

    site = os.path.dirname(os.path.dirname(os.path.abspath(matplotlib.__file__)))
    cand = os.path.join(site, "mpl_toolkits")
    if os.path.isdir(cand):
        for name in [m for m in sys.modules if m.split(".")[0] == "mpl_toolkits"]:
            del sys.modules[name]
        if site not in sys.path:
            sys.path.insert(0, site)
        import mpl_toolkits
        if cand not in list(mpl_toolkits.__path__):
            mpl_toolkits.__path__.insert(0, cand)
    from mpl_toolkits.mplot3d import Axes3D

    # Il registro delle proiezioni di matplotlib viene popolato all'import di
    # `matplotlib.projections`, dentro un try/except silenzioso: se al momento
    # dell'import mpl_toolkits era ancora quello sbagliato, '3d' resta assente
    # per sempre. Va quindi registrata esplicitamente.
    from matplotlib.projections import projection_registry, register_projection
    if "3d" not in projection_registry._all_projection_types:
        register_projection(Axes3D)
    return Axes3D


# ---------------------------------------------------------------------------
# Salvataggio delle figure
# ---------------------------------------------------------------------------
def save_figure(fig, out_png: str, dpi: int = 130) -> list[str]:
    """
    Scrive la figura in PNG **e** in PDF, e restituisce i percorsi.

    Il PNG serve per guardarla al volo; nel report va il PDF. Un raster a 130
    dpi messo a piena larghezza su A4 e' visibilmente morbido, e accanto alle
    figure vettoriali gia' presenti nel report la differenza si nota. Il PDF
    e' vettoriale, scala a qualunque dimensione e di solito pesa meno.
    """
    import os
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    out_pdf = os.path.splitext(out_png)[0] + ".pdf"
    # bbox_inches stretto: senza, il PDF porta i margini della figura e nel
    # report resta un bordo bianco che nessun \includegraphics puo' togliere.
    fig.savefig(out_pdf, bbox_inches="tight")
    return [out_png, out_pdf]
