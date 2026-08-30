#!/usr/bin/env python3
"""
Differenziazione automatica contro differenze finite — dispense §5.2 e §5.3.

Il corso da' tre numeri sul costo di un gradiente in n variabili:

  differenze in avanti  : n+1 valutazioni, accuratezza ~ sqrt(eps) ~ 1e-8
                          con passo ottimo h ~ sqrt(eps)
  differenze centrate   : 2n  valutazioni, accuratezza ~ eps^(2/3) ~ 1e-11
                          con passo ottimo h ~ eps^(1/3) ~ 7.6e-6
  AD in modo inverso    : < 3 valutazioni INDIPENDENTEMENTE da n,
                          accuratezza a precisione macchina

Questo script li verifica sull'obiettivo effettivamente minimizzato dall'MPC,
non su una funzione di prova, e poi misura quanto vale l'Hessiana esatta
confrontando IPOPT con Hessiana da AD contro L-BFGS.

Uso:
    python3 viz/ad_vs_fd.py
    python3 viz/ad_vs_fd.py --bag viz/bags/industrial_plant_fix
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import casadi as ca
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

EPS = np.finfo(float).eps


def build_point(cfg, sc, ref=None):
    """Costruisce l'NLP e restituisce (f, grad_f AD, x, p) in un punto sensato."""
    tracker = common.make_tracker(cfg)
    sc2 = sc if ref is None else common.Scenario(
        sc.name, sc.pose, sc.obstacles, sc.goal, ref, sc.extent)
    res = common.solve_at(tracker, sc.pose, sc2)
    o = tracker._opti
    f_fun = ca.Function("f", [o.x, o.p], [o.f])
    gf_fun = ca.Function("gf", [o.x, o.p], [ca.gradient(o.f, o.x)])
    x = np.array(o.debug.value(o.x)).ravel()
    p = np.array(o.debug.value(o.p)).ravel()
    return tracker, res, f_fun, gf_fun, x, p


def fd_forward(f, x, p, h):
    """Gradiente per differenze in avanti: n+1 valutazioni."""
    n = x.size
    g = np.empty(n)
    f0 = float(f(ca.DM(x), ca.DM(p)))
    for i in range(n):
        xp = x.copy(); xp[i] += h
        g[i] = (float(f(ca.DM(xp), ca.DM(p))) - f0) / h
    return g, n + 1


def fd_central(f, x, p, h):
    """Gradiente per differenze centrate: 2n valutazioni."""
    n = x.size
    g = np.empty(n)
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (float(f(ca.DM(xp), ca.DM(p))) - float(f(ca.DM(xm), ca.DM(p)))) / (2 * h)
    return g, 2 * n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="centred_pillar")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    ref = None
    if args.bag:
        import bag_source
        frs = bag_source.frames(bag_source.read_bag(args.bag))
        k = args.frame if args.frame is not None else bag_source.hardest_frame(frs)
        f_ = frs[k]
        sc = bag_source.to_scenario(f_, name=os.path.basename(args.bag.rstrip("/")))
        ref = np.atleast_2d(f_.path)[:, :2]
        etichetta = f"bag {sc.name} ciclo {k}"
    else:
        sc = common.SCENARIOS[args.scenario]()
        etichetta = f"scenario {sc.name}"

    tracker, res, f_fun, gf_fun, x, p = build_point(cfg, sc, ref)
    n = x.size
    print(f"{etichetta} · n = {n} variabili decisionali · N={cfg.N} dt={cfg.dt}")
    print()

    # ── Gradiente di riferimento: AD ────────────────────────────────────
    # Warm-up piu' minimo fra blocchi: con la media si misurava un gradiente AD
    # piu' veloce di una valutazione di f, cioe' un rapporto impossibile.
    xd, pd = ca.DM(x), ca.DM(p)
    g_ad = np.array(gf_fun(xd, pd)).ravel()
    t_ad = common.time_call(lambda: gf_fun(xd, pd), 200, 5)
    t_f = common.time_call(lambda: f_fun(xd, pd), 200, 5)

    print("=" * 74)
    print("COSTO  (§5.3)")
    print("=" * 74)
    print(f"una valutazione di f          : {t_f*1e6:9.1f} us")
    print(f"gradiente per AD              : {t_ad*1e6:9.1f} us  "
          f"= {t_ad/t_f:5.2f} valutazioni di f")
    print(f"  il corso prevede < 3 valutazioni indipendentemente da n: "
          f"{'CONFERMATO' if t_ad/t_f < 3 else 'NON confermato'}")
    print()

    # ── Accuratezza delle differenze finite al variare del passo ────────
    print("=" * 74)
    print("ACCURATEZZA vs PASSO  (§5.2)")
    print("=" * 74)
    scala = float(np.linalg.norm(g_ad))
    print(f"norma del gradiente di riferimento: {scala:.3e}")
    print()
    print("| h | errore relativo (avanti) | errore relativo (centrate) |")
    print("|---|---|---|")
    passi = [1e-2, 1e-4, np.cbrt(EPS), 1e-6, np.sqrt(EPS), 1e-10, 1e-12]
    best_f = (np.inf, None); best_c = (np.inf, None)
    for h in passi:
        gf_, _ = fd_forward(f_fun, x, p, h)
        gc_, _ = fd_central(f_fun, x, p, h)
        ef = float(np.linalg.norm(gf_ - g_ad) / max(scala, 1e-300))
        ec = float(np.linalg.norm(gc_ - g_ad) / max(scala, 1e-300))
        if ef < best_f[0]: best_f = (ef, h)
        if ec < best_c[0]: best_c = (ec, h)
        nota = ""
        if abs(h - np.sqrt(EPS)) < 1e-12: nota = "  <- sqrt(eps), ottimo per avanti"
        if abs(h - np.cbrt(EPS)) < 1e-12: nota = "  <- eps^(1/3), ottimo per centrate"
        print(f"| {h:.2e} | {ef:.3e} | {ec:.3e} |{nota}")
    print()
    print(f"miglior errore in avanti : {best_f[0]:.2e} a h = {best_f[1]:.2e}   "
          f"(il corso prevede ~1e-8)")
    print(f"miglior errore centrate  : {best_c[0]:.2e} a h = {best_c[1]:.2e}   "
          f"(il corso prevede ~1e-11)")
    print()

    # ── Costo in valutazioni ────────────────────────────────────────────
    _, nev_f = fd_forward(f_fun, x, p, np.sqrt(EPS))
    _, nev_c = fd_central(f_fun, x, p, np.cbrt(EPS))
    print("=" * 74)
    print("BILANCIO")
    print("=" * 74)
    print(f"| metodo | valutazioni di f | tempo stimato | accuratezza |")
    print(f"|---|---|---|---|")
    print(f"| differenze in avanti | {nev_f} | {nev_f*t_f*1e3:7.2f} ms | {best_f[0]:.1e} |")
    print(f"| differenze centrate  | {nev_c} | {nev_c*t_f*1e3:7.2f} ms | {best_c[0]:.1e} |")
    print(f"| AD (modo inverso)    | {t_ad/t_f:.1f} | {t_ad*1e3:7.2f} ms | "
          f"precisione macchina |")
    print()
    # Il budget per ciclo lo fissa il RATE DI CONTROLLO (mpc_rate_hz), non il
    # passo di discretizzazione dt: sono due parametri distinti.
    rate = float(raw.get("mpc_rate_hz", 1.0 / cfg.dt))
    budget_ms = 1000.0 / rate
    print(f"A {rate:g} Hz nominali il budget per ciclo e' {budget_ms:.0f} ms: "
          f"le differenze centrate da sole ne userebbero "
          f"{nev_c*t_f*1e3/budget_ms*100:.0f}%.")
    print("E il rapporto peggiora LINEARMENTE con n, quindi con l'orizzonte:")
    print("e' l'argomento che rende discutibile allungare N (roadmap §1.3).")
    print()

    # ── Hessiana esatta contro L-BFGS ───────────────────────────────────
    print("=" * 74)
    print("HESSIANA ESATTA (AD) contro L-BFGS  (§4.4.4)")
    print("=" * 74)
    print("| Hessiana | iterazioni | tempo solve | J* | esito |")
    print("|---|---|---|---|---|")
    for hess in ("exact", "limited-memory"):
        c = dataclasses.replace(cfg, hessian=hess, max_iter=500)
        tr = common.make_tracker(c)
        sc2 = sc if ref is None else common.Scenario(
            sc.name, sc.pose, sc.obstacles, sc.goal, ref, sc.extent)
        r = common.solve_at(tr, sc.pose, sc2)
        print(f"| {hess:14s} | {r.iterations:3d} | {r.solve_time_ms:7.1f} ms | "
              f"{r.cost:.3f} | {r.status} |")
    print()
    print("Lettura: l'Hessiana esatta costa piu' lavoro per iterazione ma ne")
    print("serve meno; L-BFGS evita le derivate seconde e paga in iterazioni.")
    print("E' il compromesso Newton / quasi-Newton del §4.4.4, misurato senza")
    print("scrivere un solutore.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
