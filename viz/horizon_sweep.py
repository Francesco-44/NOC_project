#!/usr/bin/env python3
"""
Scelta dell'orizzonte: sweep su N e dt — dispense §7.2.5.

L'orizzonte di predizione e' N passi da dt secondi. I due parametri non sono
intercambiabili, perche' agiscono su tre grandezze diverse:

    orizzonte temporale   T = N*dt          -> quanto lontano l'MPC vede
    numero di variabili   ~ N               -> quanto costa risolvere
    errore di troncamento ~ dt^p            -> quanto e' fedele la predizione
                                               (p = 1 con Euler, 2 col punto medio)

Allungare l'orizzonte alzando N costa calcolo; alzando dt costa accuratezza.
Lo sweep bidimensionale serve a vedere dove sta il compromesso, invece di
ereditare N = 15 e dt = 0.2 da una taratura mai giustificata.

METODO. Ogni combinazione (N, dt) e' valutata in ANELLO CHIUSO su scenari con
ostacoli. La durata della missione e' tenuta costante IN SECONDI, non in passi:
in questo simulatore dt e' anche il periodo di controllo e il passo
dell'impianto, quindi confrontare a parita' di passi darebbe a dt piccoli una
missione piu' corta e falserebbe tutto.

Uso:
    python3 viz/horizon_sweep.py                 # griglia completa (~10 min)
    python3 viz/horizon_sweep.py --quick
    python3 viz/horizon_sweep.py --scenari narrow_gap u_trap corridor
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402

# Durata della missione simulata [s]. Costante fra tutte le combinazioni.
T_MISSIONE = 30.0
# Soglia di "goal raggiunto" usata da common.closed_loop
R_GOAL = 0.3


def valuta(cfg, raw, sc, N, dt) -> dict:
    """Una missione in anello chiuso con un dato (N, dt)."""
    c = dataclasses.replace(cfg, N=int(N), dt=float(dt))
    tr = common.make_tracker(c)
    steps = max(5, int(round(T_MISSIONE / dt)))
    t0 = time.perf_counter()
    h = common.closed_loop(tr, sc, steps=steps, raw=raw)
    wall = time.perf_counter() - t0

    P = np.asarray(h["pose"], dtype=float)
    ms = np.asarray(h["solve_ms"], dtype=float)
    ok = np.asarray(h["success"], dtype=float)
    d_fin = float(np.linalg.norm(P[-1, :2] - sc.goal))
    # closed_loop registra la posa PRIMA di muovere e poi esce: l'ultima posa
    # salvata e' sempre un passo prima del goal, quindi d_fin risulta appena
    # sopra la soglia anche quando il goal e' stato raggiunto. Il segnale
    # affidabile e' l'uscita anticipata dal ciclo.
    raggiunto = bool(len(P) < steps)
    # lunghezza percorsa e efficienza rispetto alla distanza in linea d'aria
    lung = float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum())
    diretta = float(np.linalg.norm(sc.goal - sc.pose[:2]))
    return {
        "N": int(N), "dt": float(dt),
        "T_orizzonte": float(N * dt),
        "n_var": int(6 * (N + 1) + 3 * N),
        "passi": int(len(P)),
        "t_missione_s": float(len(P) * dt),
        "goal_raggiunto": raggiunto,
        "distanza_finale": d_fin,
        "passi_al_goal": int(len(P)) if raggiunto else None,
        "tempo_al_goal_s": float(len(P) * dt) if raggiunto else None,
        "clearance_min": float(common.clearance(P[:, :2], sc.obstacles)),
        "lunghezza_percorso": lung,
        "efficienza": float(diretta / lung) if lung > 1e-9 else 0.0,
        "solve_ms_mediana": float(np.median(ms)),
        "solve_ms_p95": float(np.percentile(ms, 95)),
        "solve_ms_max": float(ms.max()),
        "tasso_successo": float(ok.mean()),
        "wall_s": wall,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--scenari", nargs="*", default=["narrow_gap", "u_trap"])
    ap.add_argument("--N", type=int, nargs="*", default=None)
    ap.add_argument("--dt", type=float, nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    Ns = args.N or ([8, 15, 30] if args.quick else [5, 10, 15, 25, 40])
    dts = args.dt or ([0.2, 0.4] if args.quick else [0.1, 0.2, 0.3, 0.4])
    rate = float(raw.get("mpc_rate_hz", 1.0 / cfg.dt))
    budget = 1000.0 / rate

    print(f"griglia {len(Ns)}x{len(dts)} su {len(args.scenari)} scenari · "
          f"missione {T_MISSIONE:.0f} s · budget di ciclo {budget:.0f} ms")
    print(f"deployato: N={cfg.N} dt={cfg.dt} (orizzonte {cfg.N*cfg.dt:.1f} s)")
    print()

    righe = []
    t0 = time.perf_counter()
    for nome in args.scenari:
        sc = common.SCENARIOS[nome]()
        for N in Ns:
            for dt in dts:
                r = valuta(cfg, raw, sc, N, dt)
                r["scenario"] = nome
                righe.append(r)
                print(f"  {nome:12s} N={N:3d} dt={dt:.2f}  "
                      f"T={r['T_orizzonte']:4.1f}s  "
                      f"goal={'si' if r['goal_raggiunto'] else 'NO'}  "
                      f"clear={r['clearance_min']:.3f}  "
                      f"p95={r['solve_ms_p95']:6.1f}ms", flush=True)
    print(f"\ndurata totale {time.perf_counter()-t0:.0f} s")

    # ── aggregazione sugli scenari ──────────────────────────────────────
    print()
    print("=" * 78)
    print("MEDIA SUGLI SCENARI")
    print("=" * 78)
    print("| N | dt | T [s] | var | goal | t al goal [s] | clearance min | "
          "lung. percorso | solve p95 [ms] | entro budget |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    agg = {}
    for N in Ns:
        for dt in dts:
            sel = [r for r in righe if r["N"] == N and r["dt"] == dt]
            if not sel:
                continue
            a = {
                "N": N, "dt": dt, "T": N * dt, "n_var": sel[0]["n_var"],
                "goal": float(np.mean([r["goal_raggiunto"] for r in sel])),
                "t_goal": (float(np.mean([r["tempo_al_goal_s"] for r in sel]))
                           if all(r["goal_raggiunto"] for r in sel) else None),
                "clearance": float(np.mean([r["clearance_min"] for r in sel])),
                "lung": float(np.mean([r["lunghezza_percorso"] for r in sel])),
                "p95": float(np.mean([r["solve_ms_p95"] for r in sel])),
                "succ": float(np.mean([r["tasso_successo"] for r in sel])),
            }
            a["entro_budget"] = bool(a["p95"] <= budget)
            agg[(N, dt)] = a
            tg = f"{a['t_goal']:.1f}" if a["t_goal"] is not None else "—"
            print(f"| {N} | {dt:.2f} | {a['T']:.1f} | {a['n_var']} | "
                  f"{a['goal']*100:.0f}% | {tg} | {a['clearance']:.3f} | "
                  f"{a['lung']:.2f} | {a['p95']:.1f} | "
                  f"{'si' if a['entro_budget'] else 'NO'} |")

    # ── lettura ─────────────────────────────────────────────────────────
    print()
    print("Lettura:")
    fuori = [a for a in agg.values() if not a["entro_budget"]]
    nogoal = [a for a in agg.values() if a["goal"] < 0.99]
    fatt = [a for a in agg.values() if a["entro_budget"] and a["goal"] > 0.99]

    if nogoal:
        print(f"  {len(nogoal)} combinazioni non raggiungono il goal su tutti gli "
              f"scenari: " + ", ".join(f"N={a['N']}/dt={a['dt']:g}" for a in nogoal))
    if fuori:
        print(f"  {len(fuori)} su {len(agg)} sforano il budget di {budget:.0f} ms sul "
              f"p95: " + ", ".join(f"N={a['N']}/dt={a['dt']:g}" for a in fuori))
        print("  (il vincolo real-time lo viola la CODA, non il caso tipico: la")
        print("   mediana nasconderebbe il problema)")

    # Gli obiettivi sono in CONFLITTO — arrivare presto, stare lontano dagli
    # ostacoli, calcolare in fretta — quindi non esiste un "migliore": esiste un
    # insieme non dominato. Sceglierne uno per il solo massimo di clearance
    # premierebbe configurazioni che impiegano il doppio del tempo.
    def domina(a, b):
        """a domina b: non peggiore su tutto e migliore su almeno un criterio."""
        crit = [(a["t_goal"], b["t_goal"], -1),      # meno e' meglio
                (a["clearance"], b["clearance"], +1),  # piu' e' meglio
                (a["p95"], b["p95"], -1)]
        if any(x is None or y is None for x, y, _ in crit):
            return False
        ge = all((x - y) * sgn >= 0 for x, y, sgn in crit)
        gt = any((x - y) * sgn > 0 for x, y, sgn in crit)
        return ge and gt

    nd = [a for a in fatt if not any(domina(b, a) for b in fatt if b is not a)]
    print()
    if nd:
        print(f"  Insieme NON DOMINATO su (tempo al goal, clearance, p95), fra le "
              f"{len(fatt)} combinazioni ammissibili:")
        print("  | N | dt | T [s] | t al goal [s] | clearance | p95 [ms] |")
        print("  |---|---|---|---|---|---|")
        for a in sorted(nd, key=lambda z: z["t_goal"]):
            print(f"  | {a['N']} | {a['dt']:g} | {a['T']:.1f} | {a['t_goal']:.1f} | "
                  f"{a['clearance']:.3f} | {a['p95']:.1f} |")
    dep = agg.get((cfg.N, cfg.dt))
    if dep:
        stato = "NON DOMINATA" if dep in nd else "dominata"
        print()
        print(f"  La configurazione deployata (N={cfg.N}, dt={cfg.dt:g}) e' **{stato}**: "
              f"t={dep['t_goal'] if dep['t_goal'] is not None else float('nan'):.1f} s, "
              f"clearance {dep['clearance']:.3f} m, p95 {dep['p95']:.1f} ms.")

    # Il fenomeno piu' istruttivo: orizzonti lunghi PEGGIORANO.
    lunghi = [a for a in agg.values() if a["T"] >= 6.0 and a["t_goal"] is not None]
    corti = [a for a in agg.values() if a["T"] < 6.0 and a["t_goal"] is not None]
    if lunghi and corti:
        print()
        print(f"  Orizzonte < 6 s: tempo al goal {np.mean([a['t_goal'] for a in corti]):.1f} s, "
              f"clearance {np.mean([a['clearance'] for a in corti]):.3f} m")
        print(f"  Orizzonte >= 6 s: tempo al goal {np.mean([a['t_goal'] for a in lunghi]):.1f} s, "
              f"clearance {np.mean([a['clearance'] for a in lunghi]):.3f} m")
        print("  Allungare l'orizzonte oltre ~5 s PEGGIORA entrambe le metriche: il")
        print("  riferimento si estende su un percorso che A* ripianifichera' comunque,")
        print("  e l'MPC si impegna a inseguire un obiettivo che cambiera'.")

    out_dir = os.path.join(_HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "horizon_sweep.json"), "w") as fh:
        json.dump({"righe": righe, "budget_ms": budget,
                   "deployato": {"N": cfg.N, "dt": cfg.dt}}, fh, indent=2, default=float)

    # ── figura ──────────────────────────────────────────────────────────
    common.ensure_mpl3d()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.full((len(Ns), len(dts)), np.nan)
    C = np.full_like(M, np.nan)
    for i, N in enumerate(Ns):
        for j, dt in enumerate(dts):
            a = agg.get((N, dt))
            if a:
                M[i, j] = a["clearance"]
                C[i, j] = a["p95"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, D, tit, cmap in ((axes[0], M, "clearance minima [m]\n(più alto è meglio)", "viridis"),
                             (axes[1], C, "solve p95 [ms]\n(più basso è meglio)", "magma_r")):
        im = ax.imshow(D, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(dts))); ax.set_xticklabels([f"{d:g}" for d in dts])
        ax.set_yticks(range(len(Ns))); ax.set_yticklabels(Ns)
        ax.set_xlabel("$\\Delta t$ [s]"); ax.set_ylabel("$N$")
        ax.set_title(tit, fontsize=10)
        for i in range(len(Ns)):
            for j in range(len(dts)):
                if np.isfinite(D[i, j]):
                    ax.text(j, i, f"{D[i,j]:.2f}", ha="center", va="center",
                            fontsize=8, color="w")
        # marca il punto deployato
        if cfg.N in Ns and cfg.dt in dts:
            ax.plot(dts.index(cfg.dt), Ns.index(cfg.N), "s", ms=18, mfc="none",
                    mec="#d62728", mew=2.5)
        fig.colorbar(im, ax=ax, shrink=.85)

    ax = axes[2]
    for a in agg.values():
        col = "#2ca02c" if a["entro_budget"] else "#d62728"
        ax.scatter(a["p95"], a["clearance"], s=26 + 2.2 * a["n_var"] / 10,
                   c=col, alpha=.75, edgecolors="k", linewidths=.4)
        ax.annotate(f"{a['N']}/{a['dt']:g}", (a["p95"], a["clearance"]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.axvline(budget, ls="--", c="k", label=f"cycle budget {budget:.0f} ms")
    ax.set_xscale("log")
    ax.set_xlabel("solve time, 95th pct. [ms]"); ax.set_ylabel("minimum clearance [m]")
    ax.set_title("performance against cost\n(green = within budget)", fontsize=10)
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    fig.suptitle("Horizon selection: $N \\times \\Delta t$ --- "
                 f"red box = deployed configuration", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, "horizon_sweep.png")
    common.save_figure(fig, out, 130)
    print(f"\nsalvati:\n  {out}\n  {os.path.join(out_dir,'horizon_sweep.json')}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
