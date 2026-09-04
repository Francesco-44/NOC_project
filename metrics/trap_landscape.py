#!/usr/bin/env python3
"""
PANNELLO — il minimo locale dentro le trappole, e come sparisce.

Disegna, per un mondo di g1_sim, DUE paesaggi di navigazione sul piano:

  SINISTRA   d_eucl(p) = ||p - goal||          mascherata sullo spazio libero
  DESTRA     d_geo(p)  = distanza GEODETICA dal goal, aggirando gli ostacoli

Sono i due modi di rispondere alla domanda "quanto sono lontano dal goal", e la
differenza fra loro e' tutta la storia di questi mondi.

PERCHE' PROPRIO QUESTE DUE. Il pianificatore non minimizza un potenziale: A*
sceglie un BERSAGLIO e ci pianifica dentro. Ma il bersaglio lo sceglie per
distanza dal goal, quindi il robot in pratica DISCENDE quel campo, vincolato a
restare nello spazio libero. Un minimo locale in senso vincolato — una cella
libera nessuno dei cui vicini liberi ha valore minore — e' allora una posizione
da cui ogni mossa ammissibile ALLONTANA dal goal. E' esattamente il fondo del
vicolo cieco, l'interno del ferro di cavallo, il punto medio davanti al muro.

Il campo geodetico, per costruzione, NON PUO' averne: e' l'uscita di un
Dijkstra, quindi ogni cella libera ha per forza un vicino a valore strettamente
minore lungo la catena che la collega al goal. Non e' un fatto empirico da
verificare mondo per mondo, e' una proprieta' dell'algoritmo — ed e' la ragione
per cui sostituirlo alla distanza euclidea elimina la classe di fallimento
invece di attenuarla.

I MONDI DI CONTROLLO servono a mostrare il rovescio: in open_corridor o zigzag
il campo euclideo NON ha minimi locali interni, il livello scende in modo
monotono fino al goal, e infatti li' si passa. La differenza fra trappola e non
trappola e' visibile prima ancora di far muovere il robot.

Uso:
    python3 metrics/trap_landscape.py --mondi horseshoe dead_end long_wall open_corridor
    python3 metrics/trap_landscape.py --mondi horseshoe --traiettoria
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402

# Oltre questo scarto fra geodetica ed euclidea un minimo locale e' una vera
# trappola: sotto, e' un semplice appoggio a un muro senza giro da fare.
SOGLIA_TRAPPOLA = 2.0

from a_star_mpc_planner.geodesic_field import GeodesicField, block_radius  # noqa: E402


def campi(sc, raw, reso=0.15):
    """(X, Y, d_eucl, d_geo, occ) sullo stesso reticolo."""
    rb = block_radius(float(raw["grid_std"]), float(raw["obstacle_threshold"]))
    gf = GeodesicField(sc.obstacles, sc.goal, sc.pose[:2], reso=reso, r_block=rb)
    xs = gf.minx + np.arange(gf.nx) * gf.reso
    ys = gf.miny + np.arange(gf.ny) * gf.reso
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    eucl = np.hypot(X - sc.goal[0], Y - sc.goal[1])
    eucl = np.where(gf.occ, np.nan, eucl)
    geo = np.where(gf.occ | ~np.isfinite(gf.D), np.nan, gf.D)
    return X, Y, eucl, geo, gf


def minimi_locali(F, goal_ij, tol=1e-9):
    """Celle libere il cui valore e' <= a quello di OGNI vicino libero (8-vicini).

    Il goal e' escluso: e' il minimo globale, non una trappola. Si usa <= e non
    < perche' su un reticolo un fondo piatto e' comunque un minimo: da li'
    nessuna mossa migliora.
    """
    nx, ny = F.shape
    # Vettorizzato: il minimo sugli 8 vicini si ottiene con np.fmin su 8 copie
    # traslate (fmin ignora i NaN, cioe' le celle occupate, che e' proprio il
    # comportamento voluto — un vicino dentro un muro non e' una via d'uscita).
    P = np.full((nx + 2, ny + 2), np.nan)
    P[1:-1, 1:-1] = F
    vic = np.full_like(F, np.nan)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            vic = np.fmin(vic, P[1 + di:nx + 1 + di, 1 + dj:ny + 1 + dj])
    mask = np.isfinite(F) & np.isfinite(vic) & (F <= vic + tol)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    if goal_ij is not None:
        gi, gj = goal_ij
        # si esclude un intorno del goal: e' il minimo GLOBALE, non una trappola
        r = 3
        mask[max(0, gi - r):gi + r + 1, max(0, gj - r):gj + r + 1] = False
    return list(zip(*np.nonzero(mask)))


def raggruppa(pts, reso, raggio=1.0):
    """Un minimo locale occupa piu' celle: si tengono i centri dei gruppi."""
    centri = []
    for p in pts:
        for c in centri:
            if np.hypot((p[0] - c[0]) * reso, (p[1] - c[1]) * reso) < raggio:
                break
        else:
            centri.append(p)
    return centri


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mondi", nargs="*",
                    default=["horseshoe", "dead_end", "long_wall", "open_corridor"])
    ap.add_argument("--reso", type=float, default=0.15)
    ap.add_argument("--traiettoria", action="store_true",
                    help="sovrappone la traiettoria in anello chiuso (lento)")
    ap.add_argument("--hw", type=float, default=10.0)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    common.ensure_mpl3d()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, raw = common.load_profile(overrides=[f"grid_half_width={args.hw}"])
    salvati = []
    for nome in args.mondi:
        sc = common.world_scenario(nome)
        X, Y, eucl, geo, gf = campi(sc, raw, args.reso)
        gij = gf._idx(sc.goal[0], sc.goal[1])

        me = raggruppa(minimi_locali(eucl, gij), gf.reso)
        mg = raggruppa(minimi_locali(geo, gij), gf.reso)

        # PROFONDITA' DELLA TRAPPOLA = d_geo - d_eucl nel punto di minimo.
        # Separa una trappola vera da un minimo innocuo: contro un muro
        # perimetrale il campo euclideo ha comunque minimi locali (ci si appoggia
        # e ogni mossa allontana), ma li' la geodetica vale quanto l'euclidea,
        # quindi la profondita' e' ~0 e non c'e' nessun giro da fare. Dentro un
        # vicolo cieco invece la geodetica esplode: e' il cammino in piu' che il
        # robot dovrebbe percorrere, cioe' il costo REALE di esserci finito.
        prof = []
        for (i, j) in me:
            wx, wy = gf.minx + i * gf.reso, gf.miny + j * gf.reso
            de, dg = float(eucl[i, j]), float(geo[i, j])
            prof.append((dg - de, wx, wy, de, dg))
        prof.sort(reverse=True)
        trappole = [t for t in prof if t[0] > SOGLIA_TRAPPOLA]

        traj = None
        if args.traiettoria:
            import escape_test
            r = escape_test.corri(sc, cfg, raw, "geodetica", t_max=260.0, traccia=True)
            traj = np.asarray(r.get("traccia", []), dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
        for ax, F, tit, mins in (
            (axes[0], eucl, r"$\|p-goal\|$  (euclidea, quella usata finora)", me),
            (axes[1], geo, r"$d_{geo}(p)$  (geodetica, sulla mappa nota)", mg),
        ):
            im = ax.pcolormesh(X, Y, F, shading="auto", cmap="viridis")
            liv = np.nanpercentile(F, np.linspace(2, 98, 14))
            ax.contour(X, Y, F, levels=np.unique(liv), colors="w",
                       linewidths=.45, alpha=.55)
            ax.plot(sc.obstacles[:, 0], sc.obstacles[:, 1], ".", ms=1.1,
                    c="#111", alpha=.85)
            ax.plot(*sc.pose[:2], "o", ms=9, mfc="#2b7bd6", mec="k", label="spawn")
            ax.plot(*sc.goal, "*", ms=18, mfc="gold", mec="k", label="goal")
            for (i, j) in mins:
                wx, wy = gf.minx + i * gf.reso, gf.miny + j * gf.reso
                p_ = float(geo[i, j]) - float(eucl[i, j])
                trappola = p_ > SOGLIA_TRAPPOLA
                ax.plot(wx, wy, "X", ms=15 if trappola else 8,
                        mfc="#d62728" if trappola else "#bbbbbb",
                        mec="k", mew=1.2, zorder=5)
                if trappola:
                    ax.annotate(f"+{p_:.0f} m", (wx, wy), fontsize=9,
                                fontweight="bold", color="#d62728",
                                xytext=(8, 8), textcoords="offset points")
            if traj is not None and len(traj):
                ax.plot(traj[:, 0], traj[:, 1], "-", lw=2.0, c="#ff7f0e",
                        alpha=.95, label="traiettoria")
            ax.set_title(f"{tit}\nminimi locali: {len(mins)}  "
                         f"(trappole: {len([1 for (i,j) in mins if float(geo[i,j])-float(eucl[i,j]) > SOGLIA_TRAPPOLA])})",
                         fontsize=10)
            ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            fig.colorbar(im, ax=ax, shrink=.85, label="distanza dal goal [m]")
        axes[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"{nome} — la trappola e' un minimo locale della metrica "
                     f"euclidea, non della geodetica", fontsize=12)
        fig.tight_layout()
        out = os.path.join(_HERE, "out", f"paesaggio_{nome}.png")
        salvati += common.save_figure(fig, out, 130)
        verdetto = "TRAPPOLA" if trappole else "PASSANTE"
        print(f"{nome:16s} min. euclidei {len(me):2d} (di cui trappole "
              f"{len(trappole)})  min. geodetici {len(mg):2d}   [{verdetto}]")
        for (d, wx, wy, de, dg) in trappole:
            print(f"                  ({wx:6.2f},{wy:6.2f})  eucl {de:5.2f} m -> "
                  f"geo {dg:6.2f} m   profondita' +{d:.1f} m")
        if not args.no_show:
            continue
    print("\nsalvati:\n  " + "\n  ".join(salvati))
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
