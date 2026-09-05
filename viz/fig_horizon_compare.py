#!/usr/bin/env python3
"""
FIGURA §5.3 — la configurazione deployata contro quella che fallisce.

Il testo della sezione afferma tre cose. Questa figura le mostra tutte e tre,
senza le tabelle:

  (a),(b)  che cosa fa il robot, nei due mondi, con N=15/dt=0.35 (deployato) e
           con N=5/dt=0.10 (il punto piu' economico della campagna). In U-trap
           il secondo ATTRAVERSA la parete di fondo: clearance 0.000 m.
  (c)      la soglia: la clearance minima contro l'orizzonte temporale N*dt,
           tutte le 50 corse della campagna. Sotto 2.5 s si sfiora o si sbatte,
           sopra 3 s no, e quale coppia (N, dt) realizzi l'orizzonte non conta.
  (d)      il prezzo: il tempo di solve contro N, che e' l'unico parametro che
           lo governa.

I pannelli (a) e (b) sono RICALCOLATI con i moduli di produzione (common.
closed_loop, lo stesso tracker dell'NLP deployato); (c) e (d) leggono
viz/out/horizon_sweep.json, cioe' la campagna che generava le tabelle.

    python3 viz/fig_horizon_compare.py --no-show
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

import common  # noqa: E402

DEP = (15, 0.35)          # configurazione deployata
CHEAP = (5, 0.10)         # il punto piu' economico, e non deployabile
T_MISSIONE = 30.0         # come in horizon_sweep.py: missione a durata fissa
C_DEP, C_CHEAP = "#1f6fb4", "#d62728"


def corsa(cfg, raw, sc, N, dt):
    """Una missione in anello chiuso, con i moduli deployati."""
    c = dataclasses.replace(cfg, N=int(N), dt=float(dt))
    tr = common.make_tracker(c)
    h = common.closed_loop(tr, sc, steps=max(5, int(round(T_MISSIONE / dt))), raw=raw)
    P = np.asarray(h["pose"], float)[:, :2]
    ms = np.asarray(h["solve_ms"], float)
    col = common.check_collisions(P, sc.obstacles)
    return {
        "P": P, "t": len(P) * dt,
        "len": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()),
        "clear": float(common.clearance(P, sc.obstacles)),
        "p95": float(np.percentile(ms, 95)),
        "hit": bool(col["attraversamento"] or col["contatto"]),
    }


def punto_critico(P, obs):
    """Il punto della traiettoria in cui la clearance e' minima."""
    d = np.linalg.norm(P[:, None, :] - np.asarray(obs)[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return P[i], np.asarray(obs)[j]


def pannello_mondo(ax, sc, rd, rc, titolo, loc="lower left"):
    ax.scatter(sc.obstacles[:, 0], sc.obstacles[:, 1], s=9, c="#555555",
               marker="s", zorder=3, label="obstacles")
    for r, col, lab in ((rd, C_DEP, f"$N=15$, $\\Delta t=0.35$ s"),
                        (rc, C_CHEAP, f"$N=5$, $\\Delta t=0.10$ s")):
        ax.plot(r["P"][:, 0], r["P"][:, 1], "-", color=col, lw=2.2, zorder=5,
                label=lab + ("  — collision" if r["hit"] else ""))
        p, q = punto_critico(r["P"], sc.obstacles)
        ax.plot([p[0], q[0]], [p[1], q[1]], ":", color=col, lw=1.2, zorder=6)
        ax.annotate(f"{r['clear']:.3f} m", 0.5 * (p + q), xytext=(8, -11),
                    textcoords="offset points", fontsize=8, color=col, zorder=7,
                    bbox=dict(fc="w", ec="none", alpha=.75, pad=1.0))
    ax.plot(*sc.pose[:2], "o", ms=7, mfc="w", mec="k", mew=1.4, zorder=8)
    ax.plot(*sc.goal, "*", ms=15, mfc="#f0c000", mec="k", mew=.9, zorder=8)
    ax.set_title(titolo, fontsize=10)
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$y$ [m]")
    ax.set_aspect("equal"); ax.grid(alpha=.25)
    # spazio in basso perche' la legenda non copra gli ostacoli
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.18 * (hi - lo), hi)
    ax.legend(fontsize=7.5, loc=loc, framealpha=.92)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "fig_horizon_compare"))
    ap.add_argument("--sweep", default=os.path.join(_HERE, "out", "horizon_sweep.json"))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile()
    import matplotlib
    if args.no_show or not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))

    # --- (a), (b): le due traiettorie, nei due mondi ------------------------
    for ax, nome, tit, loc in ((axes[0, 0], "narrow_gap", "(a) narrow gap", "lower right"),
                               (axes[0, 1], "u_trap", "(b) U-trap", "lower left")):
        sc = common.get_scenario(nome)
        rd, rc = corsa(cfg, raw, sc, *DEP), corsa(cfg, raw, sc, *CHEAP)
        pannello_mondo(ax, sc, rd, rc, tit, loc)
        for tag, r in (("deployed", rd), ("cheap   ", rc)):
            print(f"{nome:11s} {tag}  t={r['t']:5.2f} s  len={r['len']:5.2f} m  "
                  f"clear={r['clear']:.3f} m  p95={r['p95']:5.1f} ms  hit={r['hit']}")

    # --- (c), (d): la campagna ---------------------------------------------
    R = json.load(open(args.sweep))["righe"]
    mk = {"narrow_gap": "o", "u_trap": "^"}
    ax = axes[1, 0]
    ax.axvspan(2.5, 3.0, color="#f0c000", alpha=.20, zorder=1)
    ax.annotate("threshold", (2.75, 0.02), fontsize=8, ha="center", color="#8a6d00")
    for r in R:
        dep = (r["N"], r["dt"]) == DEP
        cheap = (r["N"], r["dt"]) == CHEAP
        ax.scatter(r["T_orizzonte"], r["clearance_min"], marker=mk[r["scenario"]],
                   s=(70 if dep or cheap else 26),
                   c=(C_DEP if dep else C_CHEAP if cheap else "#9aa5ad"),
                   edgecolors="k" if dep or cheap else "none",
                   linewidths=.8, zorder=5 if dep or cheap else 3)
    for lab, m in mk.items():
        ax.scatter([], [], marker=m, s=26, c="#9aa5ad",
                   label={"narrow_gap": "narrow gap", "u_trap": "U-trap"}[lab])
    ax.set_title("(c) worst-case clearance against look-ahead $N\\Delta t$", fontsize=10)
    ax.set_xlabel("$N\\Delta t$ [s]"); ax.set_ylabel("min. clearance [m]")
    ax.grid(alpha=.25); ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1, 1]
    for r in R:
        dep = (r["N"], r["dt"]) == DEP
        cheap = (r["N"], r["dt"]) == CHEAP
        ax.scatter(r["N"], r["solve_ms_p95"], marker=mk[r["scenario"]],
                   s=(70 if dep or cheap else 26),
                   c=(C_DEP if dep else C_CHEAP if cheap else "#9aa5ad"),
                   edgecolors="k" if dep or cheap else "none",
                   linewidths=.8, zorder=5 if dep or cheap else 3)
    ax.set_title("(d) tail solve time against $N$", fontsize=10)
    ax.set_xlabel("$N$"); ax.set_ylabel("solve time, 95th pct. [ms]")
    ax.grid(alpha=.25)

    fig.tight_layout()
    paths = common.save_figure(fig, args.out + ".png", 150)
    print("salvato:", ", ".join(paths))
    if not (args.no_show or not os.environ.get("DISPLAY")):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
