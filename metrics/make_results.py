#!/usr/bin/env python3
"""
Generatore dei risultati per il report — un comando, un file di numeri.

Esegue tutte le misure della roadmap e scrive:

    metrics/out/results.json   tutti i numeri, in forma strutturata
    metrics/out/results.md     lo stesso, in tabelle pronte da leggere
    metrics/out/tex/           lo stesso, in LaTeX per il report (metrics/results_tex.py):
                           metrics_macros.tex, metrics_body.tex,
                           metrics_standalone.tex

Perche' esiste: i numeri di un report NON vanno copiati a mano dal terminale.
Appena si ritocca un parametro divergono dal codice in silenzio — e' gia'
successo in questo progetto con snippets/nlp_structure.py, rimasto a una
taratura superata. Qui ogni numero e' calcolato dagli
stessi identici moduli usati dagli strumenti interattivi, e il file porta con
se' la provenienza (commit git, profilo, versioni) per essere verificabile.

Le misure sono divise per CLASSE, perche' la classe decide se vanno rifatte:

  classe 1  proprieta' della formulazione. Non dipendono da nessuna run:
            si calcolano una volta e basta.
  classe 2  proprieta' dell'istanza (punto di lavoro). Variano ciclo per
            ciclo: si riportano come profilo lungo una missione, non come
            numero singolo.
  classe 3  prestazione in anello chiuso. Dipendono da run e mondo: qui
            servono davvero piu' missioni.

Uso:
    python3 metrics/make_results.py
    python3 metrics/make_results.py --quick                  # meno punti, per provare
    python3 metrics/make_results.py --only classe1 classe2
    python3 metrics/make_results.py --bag metrics/bags/altra_run
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import common  # noqa: E402


# ---------------------------------------------------------------------------
# Provenienza
# ---------------------------------------------------------------------------
def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def provenance(profile: str, cfg) -> dict:
    import casadi as ca
    dirty = bool(_git("status", "--porcelain"))
    return {
        "data_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Se l'albero e' sporco i numeri NON sono riproducibili da un commit:
        # va detto nel file, non scoperto dopo.
        "git_albero_sporco": dirty,
        "profilo": os.path.relpath(profile, _ROOT),
        "python": platform.python_version(),
        "casadi": ca.__version__,
        "numpy": np.__version__,
        "parametri_chiave": {
            "N": cfg.N, "dt": cfg.dt, "v_ref": cfg.v_ref,
            "vx_max": cfg.vx_max, "vx_min": cfg.vx_min,
            "vy_max": cfg.vy_max, "omega_max": cfg.omega_max,
            "Q_x": cfg.Q_x, "Q_y": cfg.Q_y, "Q_yaw": cfg.Q_yaw,
            "W_obs_sigmoid": cfg.W_obs_sigmoid, "obs_r": cfg.obs_r,
            "obs_alpha": cfg.obs_alpha,
            "obs_check_radius": cfg.obs_check_radius,
            "max_obs_constraints": cfg.max_obs_constraints,
            "tau_v": cfg.tau_v, "tau_w": cfg.tau_w,
            "integrator": cfg.integrator, "obstacle_mode": cfg.obstacle_mode,
            "path_mode": cfg.path_mode,
            "terminal_constraint": cfg.terminal_constraint,
            "hessian": cfg.hessian,
        },
    }


# ---------------------------------------------------------------------------
# CLASSE 1 — proprieta' della formulazione (indipendenti dalla run)
# ---------------------------------------------------------------------------
def classe1(cfg, raw, quick: bool) -> dict:
    out = {}

    # --- ordine di troncamento (§3.1) -----------------------------------
    import test_integrators as TI
    regimi = {
        "nominale (vx=0.2, w=0.3)": (0.20, 0.00, 0.30),
        "con deriva laterale": (0.20, 0.02, 0.30),
        "rotazione rapida (w=1.0)": (0.30, 0.00, 1.00),
    }
    # La griglia dei passi DERIVA dal dt deployato invece di essere fissa: con
    # una griglia costante il passo che il report cita (\resDt) puo' non esserci,
    # e "al dt deployato" finisce per riportare l'errore misurato a un altro
    # passo. T e' otto volte dt cosi' che ogni livello lo divida esattamente
    # (integrate() lo pretende, e con un numero non intero di passi l'errore
    # misurato sarebbe il disallineamento dell'orizzonte, non l'ordine).
    dts_dep = tuple(cfg.dt / 2 ** i for i in range(5))
    T_dep = 8.0 * cfg.dt
    integ = {}
    for nome, v in regimi.items():
        rows = TI.global_error_table(v, T=T_dep, dts=dts_dep)
        dts = [r[0] for r in rows]
        integ[nome] = {
            "dt": dts,
            "errore_euler": [r[1] for r in rows],
            "errore_midpoint": [r[2] for r in rows],
            "ordine_euler": TI.fit_order(dts, [r[1] for r in rows]),
            "ordine_midpoint": TI.fit_order(dts, [r[2] for r in rows]),
        }
    dep = integ["nominale (vx=0.2, w=0.3)"]
    i_dep = int(np.argmin(np.abs(np.array(dep["dt"]) - cfg.dt)))
    out["integratore"] = {
        "regimi": integ,
        "orizzonte_s": T_dep,
        "al_dt_deployato": {
            "dt": dep["dt"][i_dep],
            "errore_euler_m": dep["errore_euler"][i_dep],
            "errore_midpoint_m": dep["errore_midpoint"][i_dep],
            "guadagno": dep["errore_euler"][i_dep] / max(dep["errore_midpoint"][i_dep], 1e-300),
        },
    }

    # --- bersaglio locale: le tre metriche a confronto (§3.1) -----------
    # Sono i numeri annotati sulla figura fig_local_target: il report li cita nel
    # testo, e devono essere gli stessi. Calcolarli qui e' l'unico modo di
    # garantirlo — una figura e una frase che si contraddicono sono peggio di
    # nessuna delle due.
    import fig_local_target as FLT
    out["bersaglio_locale"] = FLT.measure(cfg, raw)

    # --- struttura e sparsita' dell'NLP (§0, §1.5) ----------------------
    sys.path.insert(0, os.path.join(_ROOT, "snippets"))
    import nlp_structure as NS
    horizons = [10, 15, 25] if quick else [10, 15, 25, 50]
    out["nlp"] = {"per_N": [NS.structure(cfg, N) for N in horizons]}

    # --- AD contro differenze finite (§4.1) -----------------------------
    import ad_vs_fd as AF
    sc = common.SCENARIOS["centred_pillar"]()
    tracker, res, f_fun, gf_fun, x, p = AF.build_point(cfg, sc)
    import casadi as ca
    R = 50 if quick else 200
    B = 3 if quick else 5
    xd, pd = ca.DM(x), ca.DM(p)
    g_ad = np.array(gf_fun(xd, pd)).ravel()
    # Il rapporto AD/f e' un micro-benchmark su tempi di ~100 us: una singola
    # coppia di misure e' inaffidabile (osservato oscillare fra 0.95 e 1.53,
    # dove < 1 e' fisicamente impossibile). Si ripete la coppia e si riporta la
    # MEDIANA con l'intervallo, invece di una cifra che finge precisione.
    K = 3 if quick else 5
    tads, tfs = [], []
    for _ in range(K):
        tads.append(common.time_call(lambda: gf_fun(xd, pd), R, B))
        tfs.append(common.time_call(lambda: f_fun(xd, pd), R, B))
    ratios = sorted(a / b for a, b in zip(tads, tfs))
    t_ad, t_f = float(np.median(tads)), float(np.median(tfs))
    scala = float(np.linalg.norm(g_ad))
    EPS = np.finfo(float).eps
    passi = [1e-4, np.cbrt(EPS), 1e-6, np.sqrt(EPS), 1e-10]
    tab = []
    for h in passi:
        gf_, nf = AF.fd_forward(f_fun, x, p, h)
        gc_, nc = AF.fd_central(f_fun, x, p, h)
        tab.append({
            "h": float(h),
            "err_avanti": float(np.linalg.norm(gf_ - g_ad) / scala),
            "err_centrate": float(np.linalg.norm(gc_ - g_ad) / scala),
        })
    bf = min(tab, key=lambda r: r["err_avanti"])
    bc = min(tab, key=lambda r: r["err_centrate"])
    rate = float(raw.get("mpc_rate_hz", 1.0 / cfg.dt))
    out["derivate"] = {
        "n_variabili": int(x.size),
        "t_f_us": t_f * 1e6,
        "t_grad_ad_us": t_ad * 1e6,
        "ad_in_valutazioni_di_f": float(np.median(ratios)),
        "ad_in_valutazioni_di_f_min": ratios[0],
        "ad_in_valutazioni_di_f_max": ratios[-1],
        "ad_ratio_attendibile": bool(ratios[0] >= 1.0),
        "valutazioni_fd_avanti": int(x.size + 1),
        "valutazioni_fd_centrate": int(2 * x.size),
        "tabella_passi": tab,
        "miglior_err_avanti": bf["err_avanti"], "h_ottimo_avanti": bf["h"],
        "miglior_err_centrate": bc["err_centrate"], "h_ottimo_centrate": bc["h"],
        "h_teorico_avanti": float(np.sqrt(EPS)),
        "h_teorico_centrate": float(np.cbrt(EPS)),
        "budget_ciclo_ms": 1000.0 / rate,
        "quota_budget_fd_centrate": (2 * x.size) * t_f * 1e3 / (1000.0 / rate),
    }

    # --- Hessiana esatta contro L-BFGS (§4.1) ---------------------------
    hess = {}
    for h in ("exact", "limited-memory"):
        t = common.make_tracker(dataclasses.replace(cfg, hessian=h, max_iter=500))
        r = common.solve_at(t, sc.pose, sc)
        hess[h] = {"iterazioni": int(r.iterations), "solve_ms": float(r.solve_time_ms),
                   "J": float(r.cost), "status": r.status}
    out["hessiana"] = hess

    # --- penalita' esatta l1 contro l2 (§2.2) ---------------------------
    import exact_penalty as EP
    sc2 = common.SCENARIOS["narrow_gap"]()
    x0 = np.array([sc2.pose[0], sc2.pose[1], sc2.pose[2], 0.0, 0.0, 0.0])
    path = [(float(q[0]), float(q[1]), 0.0) for q in sc2.reference()]
    d_safe = 1.1                      # scelto perche' il vincolo MORDA
    _, _, S_h = EP.solve_mode(cfg, "l1", 1e9, x0, path, sc2.obstacles, d_safe)
    trh, _, _ = EP.solve_mode(cfg, "l1", 1e9, x0, path, sc2.obstacles, d_safe)
    import casadi as ca2  # noqa: F401
    lam = np.abs(np.array(trh._opti.debug.value(trh._opti.lam_g)).ravel())
    n_oc = trh._n_obs_con
    mu_max = float(lam[0:n_oc:2].max()) if n_oc else 0.0
    rhos = [1e3, 1e4, 1e6] if quick else [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    rows = []
    for rho in rhos:
        _, _, S1 = EP.solve_mode(cfg, "l1", rho, x0, path, sc2.obstacles, d_safe)
        _, _, S2 = EP.solve_mode(cfg, "l2", rho, x0, path, sc2.obstacles, d_safe)
        rows.append({"rho": float(rho),
                     "slack_l1": float(S1.max()), "slack_l2": float(S2.max())})
    a = np.array([[r["rho"], r["slack_l2"]] for r in rows if r["slack_l2"] > 1e-12])
    pend = float(np.polyfit(np.log(a[-3:, 0]), np.log(a[-3:, 1]), 1)[0]) if len(a) >= 3 else float("nan")
    zero = [r["rho"] for r in rows if r["slack_l1"] < 1e-8]
    out["penalita_esatta"] = {
        "d_safe": d_safe,
        "ammissibile": bool(S_h.max() < 1e-6),
        "max_mu_vincolo_distanza": mu_max,
        "tabella": rows,
        "rho_slack_l1_nullo": min(zero) if zero else None,
        "pendenza_l2_coda": pend,
    }
    return out


# ---------------------------------------------------------------------------
# CLASSE 2 — proprieta' dell'istanza (variano ciclo per ciclo)
# ---------------------------------------------------------------------------
def classe2(cfg, raw, bagpath: str, quick: bool) -> dict:
    import bag_source
    import kkt_analysis as K
    import bifurcation_sweep as BS

    out = {}
    frs = bag_source.frames(bag_source.read_bag(bagpath))
    n = len(frs)
    idx = [int(i) for i in np.linspace(0, n - 1, 5 if quick else 9).astype(int)]

    # --- KKT lungo la missione (§2.1) -----------------------------------
    prof = []
    for k in idx:
        f = frs[k]
        if f.path is None or len(f.path) < 2:
            continue
        d = K.analyze(cfg, f.x0, [(float(p[0]), float(p[1]), 0.0) for p in f.path],
                      np.asarray(f.obstacles, float))
        d["ciclo"] = k
        d["t"] = float(f.t)
        prof.append(d)
    out["kkt"] = {
        "profilo": prof,
        "licq_sempre": all(d["licq"] for d in prof),
        "complementarita_stretta_sempre": all(d["complementarita_stretta"] for d in prof),
        "soc_c2_sempre": all(d["soc_c2"] for d in prof),
        "cono_critico_max": max(d["dim_cono_critico"] for d in prof),
        "cono_critico_min": min(d["dim_cono_critico"] for d in prof),
    }

    # --- soglia di biforcazione (§5.4) ----------------------------------
    pesi = [120, 200, 300, 600] if quick else [60, 120, 200, 300, 450, 600, 900, 1400]
    bif = {}
    sc = common.SCENARIOS["centred_pillar"]()
    rows = BS.sweep(cfg, sc, pesi, raw=raw)
    seps = np.array([r["sep"] for r in rows]); ws = np.array([r["W"] for r in rows])
    b = seps > BS.TOL_SEP
    bif["centred_pillar"] = {
        "tabella": rows,
        "soglia_inf": float(ws[~b].max()) if (~b).any() else None,
        "soglia_sup": float(ws[b].min()) if b.any() else None,
        "deployato_sotto_soglia": bool(cfg.W_obs_sigmoid < (ws[b].min() if b.any() else np.inf)),
    }
    k = bag_source.hardest_frame(frs)
    scb = bag_source.to_scenario(frs[k], name="bag")
    ref = np.atleast_2d(frs[k].path)[:, :2]
    rows = BS.sweep(cfg, scb, pesi, raw=raw, ref=ref)
    seps = np.array([r["sep"] for r in rows])
    bif["bag_ciclo_piu_impegnativo"] = {
        "ciclo": int(k), "tabella": rows,
        "biforca_mai": bool((seps > BS.TOL_SEP).any()),
    }
    out["biforcazione"] = bif

    # --- integratore sugli orizzonti VERI della bag (§3.2) --------------
    # Il test sintetico di classe1 misura l'ordine su un arco a velocita'
    # costante: e' il regime giusto per stimare un ordine, non quello dell'MPC.
    # Qui la stessa misura gira sulla sequenza di ingressi ottima di orizzonti
    # realmente risolti, dove omega cambia segno e gli errori si cancellano in
    # parte invece di sommarsi.
    import integrator_bag as IB
    out["integratore_bag"] = IB.measure(cfg, raw, bagpath,
                                        max_frames=6 if quick else 12,
                                        verbose=False)
    return out


# ---------------------------------------------------------------------------
# CLASSE 3 — prestazione in anello chiuso (dipende da run e mondo)
# ---------------------------------------------------------------------------
def classe3(cfg, raw, bagpath: str, quick: bool, integ: dict = None) -> dict:
    import bag_source
    import prediction_error as PE
    import formulation_compare as FC

    out = {}
    bag = bag_source.read_bag(bagpath)
    frs = bag_source.frames(bag)
    ts, ps = PE.pose_series(bag)

    # --- errore di predizione (§1.4) ------------------------------------
    N = cfg.N
    acc = [[] for _ in range(N + 1)]
    usati = 0
    for f in frs:
        if not f.success or f.pred is None or len(f.pred) < N + 1:
            continue
        pred = np.atleast_2d(f.pred); ok = False
        for k in range(N + 1):
            vera = PE.pose_at(ts, ps, f.t + k * cfg.dt)
            if vera is None:
                continue
            acc[k].append(float(np.linalg.norm(pred[k, :2] - vera[:2]))); ok = True
        usati += int(ok)
    med = [float(np.median(a)) if a else None for a in acc]
    p95 = [float(np.percentile(a, 95)) if a else None for a in acc]
    off = med[0] or 0.0
    out["errore_predizione"] = {
        "bag": os.path.basename(bagpath.rstrip("/")),
        "cicli_usati": usati,
        "mediana_per_k": med, "p95_per_k": p95,
        "offset_k0": off,
        "divergenza_fine_orizzonte": (med[N] - off) if med[N] is not None else None,
    }
    # Il confronto che spiega perche' RK2 non aiuta l'anello chiuso. I due
    # termini erano COSTANTI SCRITTE A MANO (1.74e-2 / 8.70e-5), misurate a
    # dt=0.20 e citate sotto l'etichetta del dt deployato: appena il profilo si
    # e' spostato a 0.35 il report ha cominciato a dire una cosa falsa senza che
    # niente se ne accorgesse. Ora vengono dalla misura sugli orizzonti veri.
    if integ:
        out["errore_predizione"]["errore_euler_orizzonte"] = \
            integ["al_dt_deployato"]["errore_euler_m"]
        out["errore_predizione"]["errore_midpoint_orizzonte"] = \
            integ["al_dt_deployato"]["errore_midpoint_m"]

    # --- integratore in anello chiuso (§3.2) ---------------------------
    # L'errore di predizione e' una cosa, il costo pagato in anello chiuso e'
    # un'altra: si applica solo il primo ingresso e A* ripianifica. Finora
    # l'affermazione ("<=1.1% sul costo mediano") non era generata da nessuno
    # script; ora lo e'.
    import integrator_bag as IB
    out["integratore_anello"] = IB.closed_loop(cfg, raw, verbose=False)

    # --- path following in theta e vincolo terminale (§1.1, §1.2) -------
    cand = FC.moving_frames(frs)
    if cand:
        m = 4 if quick else 6
        sel = [cand[i] for i in np.linspace(0, len(cand) - 1, min(m, len(cand))).astype(int)]
        rows = []
        for k in sel:
            f = frs[k]
            x0 = np.asarray(f.x0, float)
            path = [(float(p[0]), float(p[1]), 0.0) for p in f.path]
            obs = np.asarray(f.obstacles, float)
            tt, rt = FC._solve(cfg, x0, path, obs)
            th, rh = FC._solve(cfg, x0, path, obs, path_mode='theta')
            t1, r1 = FC._solve(cfg, x0, path, obs, terminal_constraint='equilibrium')
            Ut = np.array(tt._opti.debug.value(tt._U))
            Uh = np.array(th._opti.debug.value(th._U))
            Xt = np.array(tt._opti.debug.value(tt._X))
            Xh = np.array(th._opti.debug.value(th._X))
            ST = np.array(t1._opti.debug.value(t1._ST)).ravel()
            rows.append({
                "ciclo": int(k),
                "vx_media_time": float(Ut[0].mean()),
                "vx_media_theta": float(Uh[0].mean()),
                "spostamento_time": float(np.linalg.norm(Xt[:2, -1] - Xt[:2, 0])),
                "spostamento_theta": float(np.linalg.norm(Xh[:2, -1] - Xh[:2, 0])),
                "passi_saturi_theta": int((Uh[0] > cfg.vx_max - 1e-4).sum()),
                "iter_time": int(rt.iterations), "iter_theta": int(rh.iterations),
                "J_senza_terminale": float(rt.cost), "J_con_terminale": float(r1.cost),
                "slack_terminale": float(max(ST.max(), 0.0)),
            })
        a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "ciclo"}
        out["path_following"] = {
            "cicli": [r["ciclo"] for r in rows], "per_ciclo": rows,
            "vx_media_time": float(a["vx_media_time"].mean()),
            "vx_media_theta": float(a["vx_media_theta"].mean()),
            "spostamento_time": float(a["spostamento_time"].mean()),
            "spostamento_theta": float(a["spostamento_theta"].mean()),
            "guadagno_spostamento": float(a["spostamento_theta"].mean() /
                                          max(a["spostamento_time"].mean(), 1e-9) - 1.0),
            "iter_time": float(a["iter_time"].mean()),
            "iter_theta": float(a["iter_theta"].mean()),
            "velocita_inutilizzata_da_v_ref": float(1.0 - cfg.v_ref / cfg.vx_max),
        }
        # Il costo RELATIVO da solo inganna: J* varia di quasi due ordini di
        # grandezza fra i cicli, quindi lo stesso aumento assoluto si legge come
        # +1% o +40% a seconda del denominatore. Si registrano entrambi, piu' il
        # fondo scala di J*, cosi' il report puo' citare la cifra che significa
        # qualcosa invece di quella che fa impressione.
        dJ = a["J_con_terminale"] - a["J_senza_terminale"]
        out["vincolo_terminale"] = {
            "slack_max": float(a["slack_terminale"].max()),
            "sempre_ammissibile": bool(a["slack_terminale"].max() < 1e-6),
            "costo_relativo_min": float((dJ / np.maximum(np.abs(a["J_senza_terminale"]), 1e-9)).min()),
            "costo_relativo_max": float((dJ / np.maximum(np.abs(a["J_senza_terminale"]), 1e-9)).max()),
            "costo_assoluto_min": float(dJ.min()),
            "costo_assoluto_max": float(dJ.max()),
            "J_min": float(a["J_senza_terminale"].min()),
            "J_max": float(a["J_senza_terminale"].max()),
        }
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def to_markdown(res: dict) -> str:
    L = []
    m = res["meta"]
    L.append("# Risultati — generati automaticamente\n")
    L.append("> Non modificare a mano: rigenerare con `python3 metrics/make_results.py`.\n")
    L.append(f"- data: {m['data_utc']}")
    L.append(f"- commit: `{m['git_commit'][:10]}` sul branch `{m['git_branch']}`"
             + ("  **(albero di lavoro sporco: numeri non riproducibili da questo commit)**"
                if m["git_albero_sporco"] else ""))
    L.append(f"- profilo: `{m['profilo']}`")
    L.append(f"- CasADi {m['casadi']}, numpy {m['numpy']}, Python {m['python']}\n")
    p = m["parametri_chiave"]
    L.append(f"Parametri: N={p['N']}, dt={p['dt']}, v_ref={p['v_ref']}, "
             f"vx_max={p['vx_max']}, W_obs={p['W_obs_sigmoid']}, "
             f"integrator={p['integrator']}, path_mode={p['path_mode']}, "
             f"terminal={p['terminal_constraint']}\n")

    c1 = res.get("classe1")
    if c1:
        L.append("\n## Classe 1 — proprietà della formulazione\n")
        L.append("*Indipendenti dalla run: si calcolano una volta sola.*\n")
        i = c1["integratore"]
        L.append("\n### Ordine di troncamento (§2.1.3)\n")
        L.append("| regime | ordine Euler | ordine punto medio |")
        L.append("|---|---|---|")
        for nome, d in i["regimi"].items():
            L.append(f"| {nome} | {d['ordine_euler']:.2f} | {d['ordine_midpoint']:.2f} |")
        d = i["al_dt_deployato"]
        L.append(f"\nAl dt deployato ({d['dt']}) su 3 s: Euler {d['errore_euler_m']:.3e} m, "
                 f"punto medio {d['errore_midpoint_m']:.3e} m — guadagno {d['guadagno']:.0f}×.\n")

        dv = c1["derivate"]
        L.append("\n### Derivate: AD contro differenze finite (§5.2–5.3)\n")
        L.append("| metodo | valutazioni di f | accuratezza |")
        L.append("|---|---|---|")
        L.append(f"| differenze in avanti | {dv['valutazioni_fd_avanti']} | {dv['miglior_err_avanti']:.1e} |")
        L.append(f"| differenze centrate | {dv['valutazioni_fd_centrate']} | {dv['miglior_err_centrate']:.1e} |")
        L.append(f"| **AD inverso** | **{dv['ad_in_valutazioni_di_f']:.1f}** "
                 f"(intervallo {dv['ad_in_valutazioni_di_f_min']:.1f}–"
                 f"{dv['ad_in_valutazioni_di_f_max']:.1f}) | precisione macchina |")
        L.append("\n*Il costo dell'AD è un micro-benchmark su tempi di ~100 μs: si riporta "
                 "la mediana di più misure con il suo intervallo, perché una singola coppia "
                 "oscilla sensibilmente. Quello che conta, ed è stabile, è che stia fra 1 e 3 "
                 "come prevede il §5.3 — non la sua seconda cifra.*")
        if not dv.get("ad_ratio_attendibile", True):
            L.append("\n> **Misura non attendibile**: qualche ripetizione ha dato un rapporto "
                     "< 1, cioè un gradiente più veloce della funzione. Rieseguire a macchina "
                     "scarica prima di usare questo numero.")
        L.append(f"\nPassi ottimi misurati: avanti {dv['h_ottimo_avanti']:.2e} "
                 f"(teorico √eps = {dv['h_teorico_avanti']:.2e}), centrate "
                 f"{dv['h_ottimo_centrate']:.2e} (teorico eps^(1/3) = {dv['h_teorico_centrate']:.2e}).")
        L.append(f"Le differenze centrate userebbero il {dv['quota_budget_fd_centrate']*100:.0f}% "
                 f"del budget di ciclo ({dv['budget_ciclo_ms']:.0f} ms).\n")

        h = c1["hessiana"]
        L.append("\n### Hessiana esatta contro L-BFGS (§4.4.4)\n")
        L.append("| Hessiana | iterazioni | J* |")
        L.append("|---|---|---|")
        for k, v in h.items():
            L.append(f"| {k} | {v['iterazioni']} | {v['J']:.3f} |")

        pe = c1["penalita_esatta"]
        L.append("\n### Penalità esatta ℓ¹ (Thm 6.3.1)\n")
        L.append(f"d_safe = {pe['d_safe']}, max|μ\\*| = {pe['max_mu_vincolo_distanza']:.3e}\n")
        L.append("| ρ | slack ℓ¹ | slack ℓ² |")
        L.append("|---|---|---|")
        for r in pe["tabella"]:
            s1 = "0" if r["slack_l1"] < 1e-8 else f"{r['slack_l1']:.3e}"
            L.append(f"| {r['rho']:.0e} | {s1} | {r['slack_l2']:.3e} |")
        L.append(f"\nℓ¹ nullo da ρ = {pe['rho_slack_l1_nullo']:.0e}; "
                 f"pendenza ℓ² sulla coda = {pe['pendenza_l2_coda']:.2f} (attesa −1).\n")

        L.append("\n### Struttura dell'NLP\n")
        L.append("| N | variabili | vincoli | densità jac | densità hess |")
        L.append("|---|---|---|---|---|")
        for r in c1["nlp"]["per_N"]:
            L.append(f"| {r['N']} | {r['n_var']} | {r['n_con']} | "
                     f"{r['jac_density']*100:.2f}% | {r['hess_density']*100:.2f}% |")

    c2 = res.get("classe2")
    if c2:
        L.append("\n\n## Classe 2 — proprietà dell'istanza\n")
        L.append("*Variano ciclo per ciclo: il dato è il profilo, non un numero singolo.*\n")
        k = c2["kkt"]
        L.append("\n### KKT lungo la missione (§6.1)\n")
        L.append(f"LICQ sempre verificata: **{k['licq_sempre']}** · "
                 f"complementarità stretta sempre: **{k['complementarita_stretta_sempre']}** · "
                 f"SOC-C-2 sempre soddisfatta: **{k['soc_c2_sempre']}**\n")
        L.append(f"Dimensione del cono critico fra **{k['cono_critico_min']}** e "
                 f"**{k['cono_critico_max']}** a seconda del punto di lavoro "
                 f"(vale l'identità `dim(cono) = n_var − vincoli attivi`: è il "
                 f"complemento della saturazione, non una tendenza temporale).\n")
        L.append("| ciclo | t [s] | attivi | rango | LICQ | cono | λ_min proiettato |")
        L.append("|---|---|---|---|---|---|---|")
        for d in k["profilo"]:
            L.append(f"| {d['ciclo']} | {d['t']:.0f} | {d['n_attivi_totali']} | "
                     f"{d['rango_jacobiano_attivo']} | {'sì' if d['licq'] else 'NO'} | "
                     f"{d['dim_cono_critico']} | {d['hess_proj_lambda_min']:+.2e} |")
        b = c2["biforcazione"]["centred_pillar"]
        L.append(f"\n### Biforcazione (§4.4.5, Thm 4.4.6)\n")
        if b["soglia_sup"]:
            L.append(f"Soglia fra W_obs = {b['soglia_inf']:.0f} e {b['soglia_sup']:.0f}; "
                     f"il deployato è {res['meta']['parametri_chiave']['W_obs_sigmoid']:g} "
                     f"({'sotto' if b['deployato_sotto_soglia'] else 'sopra'} soglia).")
        bb = c2["biforcazione"]["bag_ciclo_piu_impegnativo"]
        L.append(f"Sul ciclo reale {bb['ciclo']}: biforca mai = **{bb['biforca_mai']}**.\n")

    c3 = res.get("classe3")
    if c3:
        L.append("\n\n## Classe 3 — prestazione in anello chiuso\n")
        L.append("*Dipendono da run e mondo: qui servono più missioni.*\n")
        e = c3["errore_predizione"]
        L.append(f"\n### Errore di predizione (§7.2.5) — bag `{e['bag']}`, "
                 f"{e['cicli_usati']} cicli\n")
        L.append(f"Offset a k=0: {e['offset_k0']:.4f} m (allineamento temporale, non modello).")
        riga = f"**Divergenza a fine orizzonte: {e['divergenza_fine_orizzonte']:.3f} m**"
        if "errore_euler_orizzonte" in e:
            riga += (f", cioè {e['divergenza_fine_orizzonte']/e['errore_euler_orizzonte']:.0f}× "
                     f"l'errore di Euler e "
                     f"{e['divergenza_fine_orizzonte']/e['errore_midpoint_orizzonte']:.0f}× "
                     f"quello del punto medio, sullo stesso orizzonte")
        L.append(riga + ".\n")
        if "path_following" in c3:
            pf = c3["path_following"]
            L.append("\n### Path following in θ (§7.2.4)\n")
            L.append("| grandezza | riferimento a tempo | ascissa θ |")
            L.append("|---|---|---|")
            L.append(f"| vx media [m/s] | {pf['vx_media_time']:.4f} | {pf['vx_media_theta']:.4f} |")
            L.append(f"| spostamento [m] | {pf['spostamento_time']:.4f} | {pf['spostamento_theta']:.4f} |")
            L.append(f"| iterazioni | {pf['iter_time']:.1f} | {pf['iter_theta']:.1f} |")
            L.append(f"\n**+{pf['guadagno_spostamento']*100:.0f}% di avanzamento**; "
                     f"v_ref lasciava inutilizzato il "
                     f"{pf['velocita_inutilizzata_da_v_ref']*100:.0f}% della velocità.\n")
            vt = c3["vincolo_terminale"]
            L.append("\n### Vincolo terminale (§7.2.5)\n")
            L.append(f"Slack massimo {vt['slack_max']:.3e} — sempre ammissibile: "
                     f"**{vt['sempre_ammissibile']}**. Costo del vincolo da "
                     f"{vt['costo_relativo_min']*100:+.1f}% a {vt['costo_relativo_max']*100:+.1f}%.\n")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default="metrics/bags/industrial_plant_fix")
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["classe1", "classe2", "classe3"])
    ap.add_argument("--quick", action="store_true",
                    help="meno punti di campionamento: per provare, non per il report")
    ap.add_argument("--out", default=os.path.join(_HERE, "out"))
    ap.add_argument("--report-dir", default=None,
                    help="albero Metrics/ del report da aggiornare "
                         "(default: report/Metrics)")
    ap.add_argument("--no-report", action="store_true",
                    help="non toccare l'albero di compilazione del report")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    voci = args.only or ["classe1", "classe2", "classe3"]
    res = {"meta": provenance(args.profile, cfg)}
    res["meta"]["modalita"] = "quick" if args.quick else "completa"
    res["meta"]["bag"] = args.bag

    t0 = time.perf_counter()
    if "classe1" in voci:
        print("[1/3] classe 1 — proprietà della formulazione…", flush=True)
        res["classe1"] = classe1(cfg, raw, args.quick)
    if "classe2" in voci:
        print("[2/3] classe 2 — proprietà dell'istanza…", flush=True)
        res["classe2"] = classe2(cfg, raw, args.bag, args.quick)
    if "classe3" in voci:
        print("[3/3] classe 3 — prestazione in anello chiuso…", flush=True)
        res["classe3"] = classe3(cfg, raw, args.bag, args.quick,
                                 res.get("classe2", {}).get("integratore_bag"))
    res["meta"]["durata_s"] = time.perf_counter() - t0

    os.makedirs(args.out, exist_ok=True)
    pj = os.path.join(args.out, "results.json")
    pm = os.path.join(args.out, "results.md")
    with open(pj, "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=float)
    with open(pm, "w") as fh:
        fh.write(to_markdown(res))
    salvati = [pj, pm]

    # LaTeX: stesse misure, forma citabile dal report. Se la generazione fallisce
    # NON si perde la campagna (json e md sono gia' su disco): si stampa l'errore
    # e si continua, perche' rifare i solve costa e rifare il .tex no.
    try:
        import results_tex
        salvati += results_tex.write_all(
            res, os.path.join(args.out, "tex"),
            results_tex.load_extra(pj))
    except Exception as exc:
        print(f"\nLaTeX NON generato: {exc}")
        print("  (i numeri sono comunque in results.json; "
              "riprovare con  python3 metrics/results_tex.py)")

    # Il report LaTeX legge macro, tabelle e figure da <build>/Metrics/. Senza
    # questo passo il .tex resta indietro di una campagna e non si vede: compila
    # lo stesso, con i numeri vecchi. Stesso trattamento del blocco sopra --- se
    # fallisce non si perde la campagna.
    if not args.no_report:
        try:
            import sync_report
            dest = args.report_dir or sync_report.DEST_DEFAULT
            n = len(sync_report.sync(os.path.join(args.out, "tex"), args.out, dest))
            print(f"\nreport aggiornato: {n} file in {dest}")
        except Exception as exc:
            print(f"\nreport NON aggiornato: {exc}")

    print("\nsalvati:")
    for q in salvati:
        print(f"  {q}")
    print(f"durata {res['meta']['durata_s']:.0f} s")
    if res["meta"]["git_albero_sporco"]:
        print("\nATTENZIONE: albero di lavoro sporco — questi numeri non sono")
        print("riproducibili dal commit indicato. Committare prima di usarli nel report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
