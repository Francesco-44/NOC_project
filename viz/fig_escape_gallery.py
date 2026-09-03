#!/usr/bin/env python3
"""
FIGURA §5.5 — la galleria delle fughe.

Un pannello per mondo: la geometria vera, la traccia della baseline (proiezione
lungo il raggio) e quella della regola deployata (argmin della geodetica). I
mondi sono in due gruppi.

TRAPPOLE       concavita' in cui la baseline entra e non esce: vicolo cieco,
               ferro di cavallo, corridoio a L, e le tre varianti di muro lungo
               in cui il varco e' da un lato solo.
CONTROLLI      scene che SEMBRANO trappole e non lo sono: il corridoio aperto in
               fondo, lo zigzag, la stanza con una porta sola, il magazzino. Qui
               la geodetica non deve fare niente di speciale: deve tirare dritto.
               Sono i falsi positivi, e senza di loro il risultato sulle
               trappole non dimostra nulla, perche' una regola che rifiuta OGNI
               concavita' le supererebbe tutte e fallirebbe queste.

DUE SCELTE DI DISEGNO, entrambe per errori visti sulla prima versione.

1. La baseline e' una BANDA larga e trasparente sotto la linea sottile della
   geodetica, non una tratteggiata accanto. Serve a due cose opposte: dove le
   due regole coincidono (corridoio aperto, porta, magazzino) l'alone rosso
   attorno alla linea blu e' l'unico modo di far vedere che coincidono, perche'
   due tratti sovrapposti si nascondono a vicenda; e dove la baseline rimbalza
   (vicolo cieco, ferro di cavallo) il gomitolo da 80 m diventa una macchia
   invece di un intrico di tratti che si leggono come tante croci.

2. Un solo marcatore di esito per piano, a croce piena con bordo bianco, in modo
   che sia distinguibile dalle intersezioni della traccia.

    python3 viz/fig_escape_gallery.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402

TRAPPOLE = ["dead_end", "horseshoe", "l_corridor",
            "long_wall", "long_wall_south", "long_wall_false_north"]
CONTROLLI = ["open_corridor", "zigzag", "door_room", "industrial"]

TITOLI = {
    "dead_end":              "dead end",
    "horseshoe":             "horseshoe",
    "l_corridor":            "L-corridor",
    "long_wall":             "long wall, gap north",
    "long_wall_south":       "long wall, gap south",
    "long_wall_false_north": "long wall, false gap",
    "open_corridor":         "open corridor",
    "zigzag":                "zigzag",
    "door_room":             "single door",
    "industrial":            "warehouse",
}

C_BASE, C_GEO = "#d62728", "#1f77b4"


def _riga(righe, mondo, piano):
    for r in righe:
        if r["mondo"] == mondo and r["piano"] == piano:
            return r
    return None


def _esito(r):
    """Etichetta compatta di esito, per il sottotitolo del pannello."""
    if r is None:
        return "--"
    if not r["goal"]:
        return "trapped"
    if r["attraversamento"]:
        return "through a wall"
    return f"{r['lung_m']:.0f} m"


def _barra_scala(ax, L):
    """Barra di scala: le dieci arene non hanno la stessa estensione, quindi
    senza questa i pannelli non sono confrontabili fra loro."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xa = x0 + 0.06 * (x1 - x0)
    ya = y0 + 0.07 * (y1 - y0)
    # alone bianco: in tre pannelli la barra cade sopra una traccia
    ax.plot([xa, xa + L], [ya, ya], "-", c="w", lw=4.0,
            solid_capstyle="butt", zorder=8)
    ax.plot([xa, xa + L], [ya, ya], "-", c="#222222", lw=1.4,
            solid_capstyle="butt", zorder=9)
    ax.text(xa + L / 2, ya + 0.02 * (y1 - y0), f"{L:g} m", ha="center",
            va="bottom", fontsize=6, color="#222222", zorder=9,
            bbox=dict(fc="w", ec="none", pad=.8))


def main() -> int:
    src = os.path.join(_HERE, "out", "escape_all.json")
    righe = json.load(open(src))

    mondi = TRAPPOLE + CONTROLLI
    ncol = 5
    nrow = int(np.ceil(len(mondi) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.6, 6.0))
    axes = np.atleast_2d(axes).ravel()

    for ax, m in zip(axes, mondi):
        sc = common.world_scenario(m)
        ax.scatter(sc.obstacles[:, 0], sc.obstacles[:, 1], s=1.4, c="#909090",
                   zorder=1, linewidths=0)

        for piano, col, lw, alpha, z in (("baseline", C_BASE, 3.4, .32, 2),
                                         ("geodetica", C_GEO, 1.25, .95, 4)):
            r = _riga(righe, m, piano)
            if r is None or "traccia" not in r:
                continue
            P = np.asarray(r["traccia"], dtype=float)
            ax.plot(P[:, 0], P[:, 1], "-", lw=lw, c=col, alpha=alpha, zorder=z,
                    solid_capstyle="round", solid_joinstyle="round")
            if not r["goal"]:
                ax.plot(P[-1, 0], P[-1, 1], "X", ms=8, c=col, mec="w", mew=1.1,
                        zorder=7)
            elif r["attraversamento"]:
                ax.plot(P[-1, 0], P[-1, 1], "P", ms=8, c=col, mec="w", mew=1.1,
                        zorder=7)

        rg = _riga(righe, m, "geodetica")
        ax.plot(*rg["spawn"][:2], "o", ms=4.5, c="#222222", zorder=6)
        ax.plot(*rg["goal_xy"], "*", ms=10, c="#ff7f0e", mec="k", mew=.3,
                zorder=6)

        trappola = m in TRAPPOLE
        ax.set_title(TITOLI[m], fontsize=9,
                     color="#111111" if trappola else "#555555", pad=13)
        sub = (f"{_esito(_riga(righe, m, 'baseline'))}"
               f"  →  {_esito(rg)}")
        ax.text(.5, 1.012, sub, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7, color="#666666")

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#222222" if trappola else "#c4c4c4")
            s.set_linewidth(1.3 if trappola else .8)
        span = max(np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()))
        _barra_scala(ax, 5.0 if span < 26 else 10.0)

    for ax in axes[len(mondi):]:
        ax.axis("off")

    h = [plt.Line2D([], [], c=C_BASE, lw=3.4, alpha=.32),
         plt.Line2D([], [], c=C_GEO, lw=1.25),
         plt.Line2D([], [], c="#222222", marker="o", ls="", ms=4.5),
         plt.Line2D([], [], c="#ff7f0e", marker="*", ls="", ms=10, mec="k",
                    mew=.3),
         plt.Line2D([], [], c="#555555", marker="X", ls="", ms=8, mec="w",
                    mew=1.1)]
    fig.legend(h, ["baseline: ray projection", "deployed: geodesic arg min",
                   "start", "goal", "still trapped when the budget ran out"],
               loc="lower center", ncol=5, fontsize=8.5, frameon=False,
               bbox_to_anchor=(.5, -.012))
    fig.tight_layout(rect=(0, .05, 1, 1))

    dest = os.path.join(_HERE, "out", "fig_escape_gallery.pdf")
    fig.savefig(dest, bbox_inches="tight")
    print("scritto:", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
