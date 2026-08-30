#!/usr/bin/env python3
"""
Vincoli robusti per constraint tightening — dispense §7.2.5.

Il vincolo di ostacolo e' imposto sulla traiettoria PREDETTA. Ma il §10.7
misura che la predizione diverge da quella vera: un vincolo soddisfatto nel
piano dell'MPC puo' essere violato nella realta'. Il rimedio del corso e'
irrigidire il vincolo di un margine che copra quella divergenza:

    ||p_k - o_j|| >= d_safe + beta(k) - s_jk

La particolarita' di questo progetto e' che beta(k) non va INDOVINATO: si legge
dal quantile dell'errore di predizione misurato sulle bag registrate. E' un
tubo ricavato dai dati, invece che da un'ipotesi sul disturbo.

Tre proprieta' che rendono la costruzione difendibile:
  - beta(0) e' quasi nullo, perche' a k=0 lo stato e' imposto come vincolo di
    uguaglianza: il vincolo non si irrigidisce dove non serve;
  - beta cresce monotonicamente, come l'incertezza;
  - il vincolo resta SOFT (lo slack c'e' gia'), quindi un tubo troppo largo
    fa crescere il costo ma non rende l'NLP inammissibile.

Uso:
    python3 viz/robust_constraints.py --bag viz/bags/industrial_plant_fix
    python3 viz/robust_constraints.py --quantile 0.99
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common          # noqa: E402
import prediction_error as PE  # noqa: E402
from a_star_mpc_planner.mpc_tracker import MPCTracker  # noqa: E402

T_MISSIONE = 30.0


def beta_da_bag(bagpath: str, cfg, quantile: float = 0.95) -> np.ndarray:
    """
    beta(k) dal quantile dell'errore di predizione misurato.

    Si sottrae l'offset a k=0: quello non e' errore di modello ma
    disallineamento fra l'istante di pubblicazione della predizione e quello di
    campionamento della posa (§10.7). Includerlo gonfierebbe il tubo di una
    costante che non ha nulla a che vedere con l'incertezza del modello.
    """
    import bag_source
    bag = bag_source.read_bag(bagpath)
    frs = bag_source.frames(bag)
    ts, ps = PE.pose_series(bag)
    N = cfg.N
    acc = [[] for _ in range(N + 1)]
    for f in frs:
        if not f.success or f.pred is None or len(f.pred) < N + 1:
            continue
        pred = np.atleast_2d(f.pred)
        for k in range(N + 1):
            vera = PE.pose_at(ts, ps, f.t + k * cfg.dt)
            if vera is not None:
                acc[k].append(float(np.linalg.norm(pred[k, :2] - vera[:2])))
    q = np.array([np.quantile(a, quantile) if a else np.nan for a in acc])
    if not np.isfinite(q).all():
        raise SystemExit("errore di predizione non stimabile su tutti i passi")
    beta = np.maximum(q - q[0], 0.0)
    # monotonia: l'incertezza non puo' diminuire andando avanti nel tempo.
    # Piccole inversioni sono rumore di campionamento, non informazione.
    return np.maximum.accumulate(beta)


def valuta_predetta(cfg, sc, beta, d_safe, x0=None, path=None, obs=None) -> dict:
    """
    Effetto del tightening sulla traiettoria PREDETTA — dove il vincolo agisce.

    NON si misura in anello chiuso, e la ragione va detta: nel simulatore di
    common.closed_loop il setpoint e' preso a distanza di lookahead lungo la
    traiettoria predetta e inseguito da un controllore proporzionale. Misurato:
    la clearance percorsa e' IDENTICA per obstacle_mode 'penalty' e 'l1', per
    ogni d_safe e ogni rho. L'anello chiuso e' quindi insensibile al
    trattamento degli ostacoli dell'MPC, e non e' un banco valido per questa
    misura.

    Si riporta anche lo SLACK: senza, non si distingue un vincolo rispettato da
    uno violato e pagato.
    """
    c = dataclasses.replace(
        cfg, obstacle_mode="l1", obs_d_safe=float(d_safe), obs_rho=1e5,
        robust_backoff=(None if beta is None else tuple(float(b) for b in beta)),
        max_iter=400)
    tr = MPCTracker(c)
    if x0 is None:
        x0 = np.array([sc.pose[0], sc.pose[1], sc.pose[2], 0.0, 0.0, 0.0])
        path = [(float(q[0]), float(q[1]), 0.0) for q in sc.reference()]
        obs = sc.obstacles
    r = tr.solve(np.asarray(x0, float), path, obstacle_points_2d=np.asarray(obs, float))
    # k >= 1: a k = 0 lo stato e' imposto dalla condizione iniziale e il
    # vincolo non e' applicato (§10.4), quindi includerlo maschererebbe
    # l'effetto dietro un minimo che nessun tightening puo' spostare.
    X = np.array(tr._opti.debug.value(tr._X))[:2, 1:].T
    O = np.atleast_2d(np.asarray(obs, float))
    S = np.array(tr._opti.debug.value(tr._S))
    return {
        "clearance": float(np.linalg.norm(X[:, None, :] - O[None, :, :], axis=2).min()),
        "slack": float(max(S.max(), 0.0)),
        "J": float(r.cost), "iter": int(r.iterations), "ms": float(r.solve_time_ms),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default="viz/bags/industrial_plant_fix")
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--scenari", nargs="*", default=["narrow_gap", "u_trap"])
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--d-safe", type=float, nargs="*", default=[0.4, 0.7, 1.0])
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    beta = beta_da_bag(args.bag, cfg, args.quantile)

    print(f"beta(k) dal quantile {args.quantile:.0%} dell'errore di predizione")
    print(f"bag: {os.path.basename(args.bag.rstrip('/'))} · N = {cfg.N} · "
          f"dt = {cfg.dt} · d_safe = {args.d_safe} m")
    print()
    print("| k | orizzonte [s] | beta(k) [m] |")
    print("|---|---|---|")
    for k in range(0, cfg.N + 1, max(1, cfg.N // 5)):
        print(f"| {k} | {k*cfg.dt:.2f} | {beta[k]:.4f} |")
    if cfg.N % max(1, cfg.N // 5):
        print(f"| {cfg.N} | {cfg.N*cfg.dt:.2f} | {beta[cfg.N]:.4f} |")
    print()
    print(f"beta(0) = {beta[0]:.4f} m (deve essere ~0: a k=0 lo stato e' imposto)")
    print(f"beta(N) = {beta[cfg.N]:.4f} m · monotona: "
          f"{bool(np.all(np.diff(beta) >= -1e-12))}")
    print()

    print("=" * 76)
    print("EFFETTO SULLA TRAIETTORIA PREDETTA")
    print("=" * 76)
    print("| scenario | d_safe | clearance senza | con beta | delta | slack senza/con | esito |")
    print("|---|---|---|---|---|---|---|")
    righe = []
    for nome in args.scenari:
        sc = common.SCENARIOS[nome]()
        for ds in args.d_safe:
            a = valuta_predetta(cfg, sc, None, ds)
            b_ = valuta_predetta(cfg, sc, beta, ds)
            d = b_["clearance"] - a["clearance"]
            if a["slack"] < 1e-6 and b_["slack"] < 1e-6 and abs(d) < 1e-4:
                esito = "vincolo inattivo"
            elif b_["slack"] > 1e-6:
                esito = "**inammissibile**"
            elif d > 1e-4:
                esito = "**tightening efficace**"
            else:
                esito = "nessun effetto"
            righe.append({"scenario": nome, "d_safe": ds, "senza": a, "con": b_,
                          "delta": d, "esito": esito})
            print(f"| {nome} | {ds:.2f} | {a['clearance']:.4f} | {b_['clearance']:.4f} | "
                  f"{d:+.4f} | {a['slack']:.3f}/{b_['slack']:.3f} | {esito} |", flush=True)

    print()
    print("Lettura (§7.2.5):")
    eff = [r for r in righe if r["esito"].startswith("**tightening")]
    ina = [r for r in righe if r["esito"] == "vincolo inattivo"]
    inf = [r for r in righe if r["esito"].startswith("**inammissibile")]
    if eff:
        best = max(eff, key=lambda r: r["delta"])
        print(f"  Il tightening FUNZIONA dove il vincolo morde e il robot ha spazio:")
        print(f"  {best['scenario']} a d_safe={best['d_safe']:.2f} guadagna "
              f"{best['delta']:+.3f} m di clearance predetta, con slack NULLO —")
        print("  cioe' il margine e' rispettato, non violato e pagato.")
    if ina:
        print(f"  In {len(ina)} casi il vincolo e' INATTIVO (d_safe + beta sotto la")
        print("  distanza gia' tenuta): nessun effetto, ed e' corretto cosi'.")
    if inf:
        print(f"  In {len(inf)} casi diventa INAMMISSIBILE (slack > 0): il tubo")
        print("  chiede piu' margine di quanto U_Sigma consenta di guadagnare in un")
        print("  orizzonte. Con vx >= 0 e vy_max = 0.02 il robot puo' solo avanzare")
        print("  lungo la propria direzione, non arretrare lateralmente (§10.4).")
        print("  La penalita' l1 mantiene il problema risolvibile: cede invece di")
        print("  rendere l'NLP inammissibile, che e' cio' per cui era stata scelta.")
    print()
    print("  LIMITE DELLA MISURA. L'effetto NON e' misurabile in anello chiuso in")
    print("  questo simulatore: la clearance percorsa risulta identica per")
    print("  obstacle_mode 'penalty' e 'l1', per ogni d_safe e ogni rho, perche' il")
    print("  setpoint e' preso a distanza di lookahead e inseguito da un")
    print("  controllore proporzionale. Il constraint tightening garantisce il")
    print("  margine NEL PIANO, ed e' li' che va verificato.")

    out = os.path.join(_HERE, "out", "robust_constraints.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"beta": beta.tolist(), "quantile": args.quantile,
                   "d_safe": args.d_safe, "righe": righe}, fh, indent=2, default=float)
    print(f"\nsalvato: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
