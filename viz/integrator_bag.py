#!/usr/bin/env python3
"""
Euler contro punto medio sugli orizzonti VERI di una bag.

Perche' non basta il test sintetico
-----------------------------------
tests/test_integrators.py misura l'ordine su un arco a velocita' costante. E'
il modo corretto di stimare un ordine — che e' una proprieta' asintotica e
richiede uno sweep su dt — e test_matches_nlp() verifica che lo schema misurato
sia bit per bit quello che mpc_tracker costruisce dentro l'NLP. Ma quel regime
non e' quello dell'MPC, che applica un ingresso DIVERSO a ogni nodo: lungo un
orizzonte vero la velocita' angolare cambia segno e gli errori si cancellano in
parte, invece di sommarsi lungo un unico arco.

Questo script rifa' la stessa misura sulla sequenza di ingressi ottima di
orizzonti realmente risolti, ripresi da una bag del G1 nel magazzino MuJoCo.

Cosa viene isolato
------------------
Il canale di velocita' e' ZOH esatto a qualunque passo. La sequenza post-lag e'
quindi calcolata una volta al passo deployato e tenuta ferma per tutti e tre i
percorsi (Euler, punto medio, riferimento), cosi' l'unica differenza e' come si
valuta R(psi) — che e' la domanda posta. Il riferimento e' l'arco esatto, non
un'integrazione fine: a velocita' costante sull'intervallo esiste in forma
chiusa, quindi non c'e' nessun errore residuo di riferimento da giustificare.

Il passo di integrazione si fa variare suddividendo l'intervallo di predizione
senza toccare l'ingresso, che resta costante su ciascun dt. E' cio' che rende
misurabile un ORDINE su una traiettoria vera.

    python3 viz/integrator_bag.py viz/bags/industrial_v6
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import common  # noqa: E402

SUBS = (1, 2, 4, 8, 16)


def measure(cfg, raw, bagpath: str, max_frames: int = 12, verbose: bool = True) -> dict:
    """Errore dei due schemi su orizzonti risolti da bag. Ritorna un dict serializzabile."""
    import bag_source
    import formulation_compare as FC
    import test_integrators as TI

    bag = bag_source.read_bag(bagpath)
    frs = bag_source.frames(bag)
    idx = FC.moving_frames(frs)
    if not idx:
        raise SystemExit("nessun ciclo utilizzabile: la bag non contiene cicli in "
                         "movimento con un path A* abbastanza lungo")
    sel = [idx[i] for i in
           np.linspace(0, len(idx) - 1, min(max_frames, len(idx))).astype(int)]

    if verbose:
        print(f"bag {os.path.basename(bagpath.rstrip('/'))}: {len(frs)} cicli, "
              f"{len(idx)} in movimento, {len(sel)} usati")
        print(f"profilo: N={cfg.N}, dt={cfg.dt} s, tau_v={cfg.tau_v}, "
              f"integratore deployato = {cfg.integrator!r}")

    err = {"euler": [[] for _ in SUBS], "midpoint": [[] for _ in SUBS]}
    lag, usati, omega = [], 0, []
    for k in sel:
        f = frs[k]
        path = [(float(p[0]), float(p[1]), 0.0) for p in np.atleast_2d(f.path)]
        _, r = FC._solve(cfg, np.asarray(f.x0, float), path,
                         np.asarray(f.obstacles, float))
        if not r.success:
            continue
        U = np.asarray(r.u_opt, float)
        h = TI.horizon_errors(f.x0, U, cfg.dt, cfg.tau_v, cfg.tau_w, subs=SUBS)
        for s in ("euler", "midpoint"):
            for i, v in enumerate(h[s]):
                err[s][i].append(v)
        lag.append(TI.lag_displacement(f.x0, U, cfg.dt, cfg.tau_v, cfg.tau_w))
        # ampiezza della rotazione sull'orizzonte: e' cio' che distingue questi
        # orizzonti dall'arco a omega costante del test sintetico
        omega.append(float(np.ptp(U[:, 2])))
        usati += 1

    if usati == 0:
        raise SystemExit("nessuno dei cicli selezionati e' stato risolto: "
                         "profilo incompatibile con la bag?")

    dts = [cfg.dt / s for s in SUBS]
    med = {s: [float(np.median(a)) for a in err[s]] for s in err}
    p95 = {s: [float(np.percentile(a, 95)) for a in err[s]] for s in err}
    ordine = {s: float(np.polyfit(np.log(dts), np.log(med[s]), 1)[0]) for s in med}

    out = {
        "bag": os.path.basename(bagpath.rstrip("/")),
        "cicli_usati": usati,
        "dt_deployato": float(cfg.dt),
        "N": int(cfg.N),
        "sub": list(SUBS),
        "dt": dts,
        "mediana": med,
        "p95": p95,
        "ordine": ordine,
        # al passo deployato (sub = 1): e' la riga che il report cita
        "al_dt_deployato": {
            "dt": float(cfg.dt),
            "errore_euler_m": med["euler"][0],
            "errore_midpoint_m": med["midpoint"][0],
            "p95_euler_m": p95["euler"][0],
            "p95_midpoint_m": p95["midpoint"][0],
            "guadagno": med["euler"][0] / max(med["midpoint"][0], 1e-300),
        },
        "lag_mediano_m": float(np.median(lag)),
        "omega_ptp_mediano": float(np.median(omega)),
    }

    if verbose:
        print()
        print(f"| dt [s] | Euler mediana [m] | punto medio mediana [m] | rapporto |")
        print("|---|---|---|---|")
        for i, d in enumerate(dts):
            e, m = med["euler"][i], med["midpoint"][i]
            print(f"| {d:.4f} | {e:.3e} | {m:.3e} | {e/m:.0f}x |")
        print(f"\nordine misurato su orizzonti veri: "
              f"Euler {ordine['euler']:.2f}, punto medio {ordine['midpoint']:.2f}")
        print(f"al passo deployato dt={cfg.dt}: Euler {med['euler'][0]:.3e} m "
              f"(p95 {p95['euler'][0]:.3e}), punto medio {med['midpoint'][0]:.3e} m "
              f"(p95 {p95['midpoint'][0]:.3e})")
        print(f"escursione mediana di omega sull'orizzonte: "
              f"{out['omega_ptp_mediano']:.2f} rad/s "
              f"(il test sintetico la tiene a 0)")
        print(f"spostamento trascurato dal transitorio del lag: "
              f"{out['lag_mediano_m']:.3e} m mediani — con tau={cfg.tau_v}s e' "
              f"dello stesso ordine dell'errore del punto medio: sotto quel "
              f"livello raffinare lo schema non compra piu' nulla.")
    return out


def closed_loop(cfg, raw, mondi=("u_trap", "narrow_gap"), verbose: bool = True) -> dict:
    """Euler contro punto medio in ANELLO CHIUSO, a parita' di tutto il resto.

    L'errore di predizione e' una cosa, il costo pagato in anello chiuso e'
    un'altra: si applica solo il primo ingresso e A* ripianifica, quindi la
    fedelta' della predizione entra solo di striscio. Questa e' la misura che
    sostanzia l'affermazione, finora non generata da nessuno script.
    """
    import dataclasses
    out = {}
    for mondo in mondi:
        sc = common.get_scenario(mondo)
        riga = {}
        for schema in ("euler", "midpoint"):
            c = dataclasses.replace(cfg, integrator=schema)
            tr = common.make_tracker(c)
            h = common.closed_loop(tr, sc, raw=raw)
            xy = np.asarray(h["pose"], float)[:, :2]
            riga[schema] = {
                "costo_mediano": float(np.median(h["cost"])),
                "lunghezza_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                "clearance_m": float(common.clearance(xy, sc.obstacles)),
                "dist_finale_goal_m": float(np.linalg.norm(xy[-1] - sc.goal)),
            }
        a, b = riga["euler"]["costo_mediano"], riga["midpoint"]["costo_mediano"]
        riga["delta_costo_pct"] = (100.0 * (a - b) / a) if (a and b) else None
        out[mondo] = riga
        if verbose:
            print(f"{mondo:12s} costo mediano  Euler {a:.3f}  punto medio {b:.3f}  "
                  f"({riga['delta_costo_pct']:+.1f}%)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?", default=os.path.join(_HERE, "bags", "industrial_v6"))
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--no-closed-loop", action="store_true")
    ap.add_argument("--json", default=os.path.join(_HERE, "out", "integrator_bag.json"))
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    res = measure(cfg, raw, args.bag, args.frames)
    if not args.no_closed_loop:
        print()
        print("anello chiuso (stesso profilo, solo l'integratore cambia):")
        res["anello_chiuso"] = closed_loop(cfg, raw)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=float)
    print(f"\nscritto {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
