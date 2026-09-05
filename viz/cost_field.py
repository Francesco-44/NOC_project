#!/usr/bin/env python3
"""
PANNELLO 1 — il paesaggio di navigazione c(x, y).

Disegna, sul piano del mondo, il costo di STARE in un punto:

    c(p) = costo di inseguimento del riferimento  +  barriera degli ostacoli

con la quota z della superficie che e' il costo, esattamente come le figure del
corso (Fig. 1.1, 4.9, 4.16). Sopra ci vanno:

  * il pallino della posizione reale del robot, che si sposta nel tempo;
  * la traiettoria predetta dall'MPC sull'orizzonte, disegnata SULLA superficie;
  * il confine dell'insieme raggiungibile in un orizzonte.

Che cosa e' e che cosa NON e'
-----------------------------
    J(U) = sum_k c(p_k) + termini sull'ingresso

`c` e' il costo di STARE in un punto; `J` e' il costo di UNA TRAIETTORIA, cioe'
la somma di `c` lungo di essa. L'MPC minimizza J, non c. Conseguenza visibile:
il pallino NON segue la massima pendenza, e puo' SALIRE localmente se questo
abbassa la somma sull'orizzonte. E' esattamente la differenza fra MPC e campo di
potenziale artificiale, ed e' il motivo per cui l'MPC esce da una trappola a U
dove un APF si incastra.

Il terzo termine che verrebbe naturale aggiungere — il costo della manovra
necessaria per arrivare in p — NON viene sommato: cresce con il quadrato della
distanza, domina gli altri due e sposta il minimo globale addosso al robot.
Viene invece usato per cio' che significa davvero: definisce l'insieme
RAGGIUNGIBILE, cioe' U_Sigma, disegnato come regione.

Uso
---
    python3 viz/cost_field.py                        # scenario u_trap, figura statica
    python3 viz/cost_field.py --scenario corridor
    python3 viz/cost_field.py --animate              # GIF dell'anello chiuso
    python3 viz/cost_field.py --profile src/a_star_mpc_planner/config/planner_params_g1.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(_REPO, "viz", "out")


def build_field(sc, cfg, res=0.04, reference="path", ref_xy=None):
    x0, x1, y0, y1 = sc.extent
    xs = np.arange(x0, x1 + res, res)
    ys = np.arange(y0, y1 + res, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    ref = sc.reference() if ref_xy is None else ref_xy
    track = (common.tracking_cost(P, ref, cfg) if reference == "path"
             else common.goal_cost(P, sc.goal, cfg))
    C = (track + common.obstacle_cost(P, sc.obstacles, cfg)).reshape(X.shape)
    return xs, ys, X, Y, C


def local_minima(C, xs, ys, size=21, pct=60):
    from scipy.ndimage import minimum_filter
    loc = (C == minimum_filter(C, size=size)) & (C < np.percentile(C, pct))
    return [(xs[i], ys[j], C[i, j]) for i, j in np.argwhere(loc)]


def _surface_z(C, log):
    return np.log10(C - C.min() + 1.0) if log else C


# ---------------------------------------------------------------------------
# Tavolozza — una sola, condivisa da tutti i pannelli
# ---------------------------------------------------------------------------
# Criterio: il CAMPO sta sul fondo e deve restare PALLIDO, perche' sopra ci
# vanno sette oggetti diversi che devono leggersi tutti. L'informazione di
# livello non si perde: la porta il tratteggio delle isolinee, non il colore.
# I colori delle sovrapposizioni sono presi dalla scala di Okabe-Ito, che
# resta distinguibile con qualunque forma di daltonismo e in bianco e nero.
C_FIELD_LO = 0.00      # estremi del troncamento di Blues per il riempimento
C_FIELD_HI = 0.62
C_ISO      = "#2B4C72"  # isolinee
C_LIDAR    = "#101010"  # ritorni LiDAR
C_EXEC     = "#D62728"  # traiettoria eseguita (e pallino del robot)
C_HORIZON  = "#000000"  # orizzonte MPC (con alone bianco)
C_REF      = "#009E73"  # riferimento A* del ciclo mostrato
C_REF_LAST = "#CC79A7"  # riferimento A* dell'ultimo ciclo
C_REACH    = "#E69F00"  # insieme raggiungibile (arancio: il campo e' blu)
C_MIN      = "#7A3FA8"  # minimi locali spuri
C_GOAL     = "#FFC20A"  # goal (e minimo che ci coincide)
C_JSTAR    = "#1F3B73"  # J* nel pannello (c)
C_SOLVE    = "#D55E00"  # tempo di soluzione nel pannello (c)


def _field_cmap():
    """Blues troncata: bianco -> blu medio, mai vicino al nero.

    Con la mappa intera (o con viridis, che era la scelta precedente) la valle
    del riferimento diventa scura quanto i punti LiDAR che le stanno sopra, e
    la cresta degli ostacoli diventa chiara quanto la linea bianca
    dell'orizzonte: entrambi gli oggetti spariscono nel fondo proprio dove
    contano. Troncando in alto il fondo resta sempre piu' chiaro di qualunque
    sovrapposizione.
    """
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    try:                                  # matplotlib >= 3.6
        base = mpl.colormaps["Blues"]
    except AttributeError:                # 3.5 di sistema
        base = mpl.cm.get_cmap("Blues")
    return LinearSegmentedColormap.from_list(
        "campo", base(np.linspace(C_FIELD_LO, C_FIELD_HI, 256)))


def _halo(lw=3.0, fg="white"):
    """Alone chiaro: rende leggibile una linea scura anche sulle creste."""
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground=fg)]


def extent_covering(ext, *point_sets, margin=1.2):
    """Allarga `ext` finche' contiene tutti i punti dati, piu' un margine.

    Serve perche' con una bag il campo viene costruito attorno a UN ciclo,
    mentre la traiettoria eseguita copre l'intera missione: senza allargare,
    meta' del percorso finisce fuori dal campo e il pannello (b) mostra una
    banda bianca al posto del costo.
    """
    xs, ys = [ext[0], ext[1]], [ext[2], ext[3]]
    for pts in point_sets:
        if pts is None:
            continue
        P = np.asarray(pts, dtype=float).reshape(-1, 2)
        if not len(P):
            continue
        xs += [P[:, 0].min() - margin, P[:, 0].max() + margin]
        ys += [P[:, 1].min() - margin, P[:, 1].max() + margin]
    return (min(xs), max(xs), min(ys), max(ys))


def figure(sc, cfg, hist, xs, ys, X, Y, C, log=True, out=None, show=True,
           focus=0):
    """
    I tre pannelli della figura 'paesaggio'.

    `focus` e' l'indice del ciclo a cui si riferiscono il pallino del robot,
    l'orizzonte MPC e l'insieme raggiungibile. Con una bag e' il ciclo su cui
    e' centrato il campo: disegnare il robot al ciclo 0 sopra un campo
    costruito su un altro ciclo (com'era prima) fa apparire l'insieme
    raggiungibile in un angolo, scollegato da tutto il resto.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    common.ensure_mpl3d()

    cmap = _field_cmap()
    Z = _surface_z(C, log)
    zlab = ("$\\log_{10}(c - c_{\\min} + 1)$" if log else "$c(x,y)$")
    poses = hist["pose"]
    k = int(np.clip(focus, 0, len(poses) - 1))
    pose_k = poses[k]
    pred = np.asarray(hist["pred"][k], dtype=float)
    reach = common.reachable_mask(np.stack([X.ravel(), Y.ravel()], 1),
                                  pose_k, cfg).reshape(X.shape)
    traj = poses[:, :2]
    ciclo = "first cycle" if k == 0 else f"cycle {k}"

    fig = plt.figure(figsize=(14.4, 8.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.30],
                          height_ratios=[1.06, 0.88],
                          left=0.05, right=0.905, top=0.905, bottom=0.075,
                          wspace=0.15, hspace=0.34)

    def zof(p):
        i = np.clip(np.searchsorted(xs, p[0]) - 1, 0, len(xs) - 1)
        j = np.clip(np.searchsorted(ys, p[1]) - 1, 0, len(ys) - 1)
        return Z[i, j]

    # ---- (a) superficie 3-D ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True,
                    alpha=0.97, rcount=110, ccount=110, rasterized=True)
    ax.plot(traj[:, 0], traj[:, 1], [zof(p) for p in traj],
            color=C_EXEC, lw=2.6, zorder=10, path_effects=_halo(4.0),
            label="executed trajectory")
    ax.plot(pred[:, 0], pred[:, 1], [zof(p) for p in pred],
            color=C_HORIZON, lw=2.0, ls="--", zorder=11,
            path_effects=_halo(3.4), label=f"MPC horizon ({ciclo})")
    ax.scatter([pose_k[0]], [pose_k[1]], [zof(pose_k[:2])], color=C_EXEC, s=55,
               edgecolors="white", linewidths=0.8, depthshade=False,
               zorder=12, label="robot")
    ax.scatter([sc.goal[0]], [sc.goal[1]], [zof(sc.goal)], marker="*", s=190,
               c=C_GOAL, edgecolors="k", linewidths=0.6, depthshade=False,
               zorder=12, label="goal")
    ax.set_xlabel("x [m]", labelpad=2)
    ax.set_ylabel("y [m]", labelpad=2)
    ax.set_zlabel(zlab, labelpad=10)
    ax.tick_params(labelsize=8, pad=1)
    ax.set_title(f"(a) cost surface $c(x,y)$ — {sc.name}", fontsize=11, pad=-6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.97), fontsize=8,
              frameon=False, handlelength=1.8, labelspacing=0.3)
    ax.view_init(elev=48, azim=-125)
    # gli assi 3-D sprecano un terzo del riquadro in margini: senza zoom il
    # pannello (a) resta grande la meta' degli altri due
    try:
        ax.set_box_aspect((1.0, 1.0, 0.62), zoom=1.30)
    except TypeError:                     # matplotlib 3.5
        ax.set_box_aspect((1.0, 1.0, 0.62))

    # ---- (b) curve di livello dall'alto ------------------------------------
    ax2 = fig.add_subplot(gs[:, 1])
    lv = np.linspace(float(Z.min()), float(Z.max()), 25)
    cs = ax2.contourf(X, Y, Z, levels=lv, cmap=cmap, extend="neither")
    # NB: il riempimento resta vettoriale. contourf ignora rasterized (lo
    # dice con una UserWarning), e rasterizzarlo per zorder costerebbe la
    # nitidezza in stampa senza far risparmiare granche': con 25 livelli il
    # PDF sta comunque sotto il mezzo megabyte.
    ax2.contour(X, Y, Z, levels=lv[::2], colors=C_ISO, linewidths=0.5,
                alpha=0.6)

    # insieme raggiungibile: regione, non solo bordo
    ax2.contourf(X, Y, reach.astype(float), levels=[0.5, 1.5],
                 colors=[C_REACH], alpha=0.14, zorder=2)
    ax2.contour(X, Y, reach.astype(float), levels=[0.5], colors=C_REACH,
                linewidths=2.2, zorder=3)

    ax2.scatter(sc.obstacles[:, 0], sc.obstacles[:, 1], s=5, c=C_LIDAR,
                zorder=4, label="LiDAR returns (obstacles)")

    refs = hist.get("ref")
    if refs is not None and refs[k] is not None:
        ax2.plot(refs[k][:, 0], refs[k][:, 1], color=C_REF, lw=2.0, ls="--",
                 zorder=5, path_effects=_halo(3.4),
                 label=f"A$^\\star$ reference ({ciclo})")
        last = next((r for r in refs[::-1] if r is not None), None)
        if last is not None and not np.array_equal(last, refs[k]):
            ax2.plot(last[:, 0], last[:, 1], color=C_REF_LAST, lw=2.0, ls=":",
                     zorder=5, path_effects=_halo(3.4),
                     label="A$^\\star$ reference (last cycle)")
    ax2.plot(traj[:, 0], traj[:, 1], color=C_EXEC, lw=2.8, zorder=6,
             path_effects=_halo(4.2), label="executed path")
    ax2.plot(pred[:, 0], pred[:, 1], ls="--", color=C_HORIZON, lw=2.2,
             zorder=7, path_effects=_halo(3.6),
             label=f"MPC horizon ({ciclo})")
    ax2.scatter([sc.goal[0]], [sc.goal[1]], marker="*", s=300, c=C_GOAL,
                edgecolors="k", linewidths=0.8, zorder=9, label="goal")
    ax2.scatter([pose_k[0]], [pose_k[1]], s=110, c=C_EXEC, edgecolors="k",
                linewidths=0.8, zorder=9, label=f"robot ({ciclo})")

    n_spurii = 0
    for (mx, my, mc) in local_minima(C, xs, ys):
        is_goal = np.hypot(mx - sc.goal[0], my - sc.goal[1]) < 0.45
        n_spurii += (not is_goal)
        ax2.scatter([mx], [my], marker="v", s=110,
                    c=(C_GOAL if is_goal else C_MIN), edgecolors="k",
                    linewidths=0.8, zorder=8)

    cax = ax2.inset_axes([1.02, 0.0, 0.022, 1.0])
    cb = fig.colorbar(cs, cax=cax)
    cb.set_label(zlab + "\ncolour scale shared by (a) and (b)", fontsize=9)
    from matplotlib.ticker import MaxNLocator
    cb.locator = MaxNLocator(nbins=7)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=8)
    cb.solids.set_rasterized(True)

    ax2.set_aspect("equal")
    ax2.set_xlim(xs[0], xs[-1])
    ax2.set_ylim(ys[0], ys[-1])
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.tick_params(labelsize=9)
    ax2.set_title("(b) level sets of $c$, local minima and reachable set",
                  fontsize=11, pad=8)

    h, l = ax2.get_legend_handles_labels()
    h += [Line2D([], [], ls="", marker="v", ms=9, mfc=C_MIN, mec="k",
                 label="local minimum (spurious)"),
          Line2D([], [], ls="", marker="v", ms=9, mfc=C_GOAL, mec="k",
                 label="local minimum at goal"),
          Patch(facecolor=C_REACH, alpha=0.30, edgecolor=C_REACH, lw=1.5,
                label=f"reachable set in {cfg.N*cfg.dt:.1f} s")]
    l += [x.get_label() for x in h[len(l):]]
    ax2.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.085),
               ncol=3, fontsize=9, frameon=False, columnspacing=1.4,
               handlelength=2.2)

    # ---- (c) il costo VERO lungo il tempo ---------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    t = np.arange(len(hist["cost"])) * cfg.dt
    axb = ax3.twinx()
    axb.fill_between(t, 0.0, hist["solve_ms"], color=C_SOLVE, alpha=0.18,
                     lw=0, zorder=1)
    axb.plot(t, hist["solve_ms"], color=C_SOLVE, lw=1.0, alpha=0.9, zorder=2)
    axb.axhline(cfg.dt * 1000, color=C_SOLVE, ls=":", lw=1.2)
    axb.annotate(f"budget {cfg.dt*1000:.0f} ms", (t[-1], cfg.dt * 1000),
                 xytext=(-4, 3), textcoords="offset points", ha="right",
                 fontsize=8, color=C_SOLVE)
    axb.set_ylabel("solve time [ms]", color=C_SOLVE, fontsize=10)
    axb.tick_params(axis="y", colors=C_SOLVE, labelsize=9)
    axb.set_ylim(0, max(cfg.dt * 1000, float(np.max(hist["solve_ms"]))) * 1.35)
    axb.set_zorder(1)

    ax3.plot(t, hist["cost"], color=C_JSTAR, lw=1.6, zorder=3)
    ax3.axvline(k * cfg.dt, color=C_EXEC, ls="--", lw=1.2, alpha=0.8, zorder=4)
    ax3.annotate(f"field shown here ({ciclo})", (k * cfg.dt, 1.0),
                 xycoords=("data", "axes fraction"), xytext=(3, -10),
                 textcoords="offset points", fontsize=8, color=C_EXEC,
                 va="top",
                 ha="left" if k * cfg.dt < 0.6 * t[-1] else "right")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("$J^\\star$ per cycle", color=C_JSTAR, fontsize=10)
    ax3.tick_params(axis="y", colors=C_JSTAR, labelsize=9)
    ax3.tick_params(axis="x", labelsize=9)
    ax3.set_xlim(t[0], t[-1] if len(t) > 1 else t[0] + 1)
    ax3.set_title("(c) optimal cost $J^\\star$ and solve time per control cycle",
                  fontsize=11, pad=6)
    ax3.grid(alpha=0.25, zorder=0)
    ax3.set_facecolor("none")
    ax3.set_zorder(2)

    fig.suptitle("Navigation cost landscape "
                 f"($N={cfg.N}$, $\\Delta t={cfg.dt:g}$ s, "
                 f"$W_\\mathrm{{obs}}={cfg.W_obs_sigmoid:g}$, "
                 f"$r_\\mathrm{{obs}}={cfg.obs_r:g}$ m)",
                 fontsize=13, y=0.975)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        common.save_figure(fig, out, 160)
        print(f"salvato: {out}")
    if show:
        plt.show()
    return fig


def animate(sc, cfg, hist, xs, ys, X, Y, C, log=True, out=None, fps=8, stride=1):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    common.ensure_mpl3d()

    Z = _surface_z(C, log)
    poses = hist["pose"]

    def zof(p):
        i = np.clip(np.searchsorted(xs, p[0]) - 1, 0, len(xs) - 1)
        j = np.clip(np.searchsorted(ys, p[1]) - 1, 0, len(ys) - 1)
        return Z[i, j]

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=_field_cmap(), alpha=0.9, linewidth=0,
                    rcount=70, ccount=70, rasterized=True)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_zlabel("log10(c - c_min + 1)" if log else "c")
    ax.view_init(elev=52, azim=-125)
    dot, = ax.plot([], [], [], "o", color=C_EXEC, ms=9)
    ln, = ax.plot([], [], [], color=C_EXEC, lw=2.0)
    pr, = ax.plot([], [], [], "--", color=C_HORIZON, lw=1.6)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(X, Y, Z, levels=40, cmap=_field_cmap())
    ax2.scatter(sc.obstacles[:, 0], sc.obstacles[:, 1], s=5, c=C_LIDAR)
    ax2.scatter([sc.goal[0]], [sc.goal[1]], marker="*", s=170, c=C_GOAL,
                edgecolors="k", zorder=5)
    ax2.set_aspect("equal"); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    dot2, = ax2.plot([], [], "o", color=C_EXEC, ms=8, zorder=6)
    ln2, = ax2.plot([], [], color=C_EXEC, lw=2.0)
    pr2, = ax2.plot([], [], "--", color=C_HORIZON, lw=1.6)
    reach_art = [None]
    ttl = ax2.set_title("")

    P = np.stack([X.ravel(), Y.ravel()], 1)

    def upd(k):
        p = poses[k]; pred = hist["pred"][k]
        tr = poses[:k + 1, :2]
        zt = [zof(q) for q in tr]
        ln.set_data(tr[:, 0], tr[:, 1]); ln.set_3d_properties(zt)
        dot.set_data([p[0]], [p[1]]); dot.set_3d_properties([zof(p[:2])])
        pr.set_data(pred[:, 0], pred[:, 1])
        pr.set_3d_properties([zof(q) for q in pred])
        ln2.set_data(tr[:, 0], tr[:, 1]); dot2.set_data([p[0]], [p[1]])
        pr2.set_data(pred[:, 0], pred[:, 1])
        if reach_art[0] is not None:
            # matplotlib >= 3.10: ContourSet e' un Artist e .collections non
            # esiste piu'; si rimuove direttamente. Il ramo vecchio resta per
            # compatibilita' con la 3.5 di sistema.
            try:
                reach_art[0].remove()
            except (AttributeError, NotImplementedError):
                for c_ in getattr(reach_art[0], "collections", []):
                    c_.remove()
        m = common.reachable_mask(P, p, cfg).reshape(X.shape).astype(float)
        reach_art[0] = ax2.contour(X, Y, m, levels=[0.5], colors=C_REACH,
                                   linewidths=1.8)
        ttl.set_text(f"t = {k*cfg.dt:.1f} s    J* = {hist['cost'][k]:.0f}")
        return ln, dot, pr, ln2, dot2, pr2

    frames = range(0, len(poses), max(1, stride))
    anim = FuncAnimation(fig, upd, frames=frames, interval=1000 / fps, blit=False)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        anim.save(out, writer=PillowWriter(fps=fps))
        print(f"salvato: {out}")
    return anim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="u_trap",
                    choices=sorted(common.SCENARIOS))
    ap.add_argument("--bag", default=None,
                    help="rosbag di un run vero: sostituisce lo scenario sintetico "
                         "e usa la traiettoria REALMENTE percorsa dal G1")
    ap.add_argument("--frame", type=int, default=None,
                    help="con --bag: indice del ciclo su cui centrare il campo "
                         "(default: quello a costo massimo, il piu' interessante)")
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--reference", default="path", choices=["path", "goal"],
                    help="path: errore rispetto al riferimento A* (fedele all'MPC); "
                         "goal: attrazione verso il goal")
    ap.add_argument("--res", type=float, default=0.04, help="risoluzione griglia [m]")
    ap.add_argument("--steps", type=int, default=250,
                    help="massimo; si esce prima al raggiungimento del goal")
    ap.add_argument("--replan-every", type=int, default=5,
                    help="ogni quanti cicli si rilancia A* (orizzonte mobile)")
    ap.add_argument("--linear", action="store_true", help="quota lineare invece di log")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--anim-stride", type=int, default=3,
                    help="salva un frame ogni N cicli (il rendering 3-D e' lento)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="CHIAVE=VALORE",
                    help="sovrascrive un parametro del profilo, ripetibile")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if args.no_show or not os.environ.get("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    common.ensure_mpl3d()

    cfg, raw = common.load_profile(args.profile, args.overrides)

    if args.bag:
        import bag_source
        frs = bag_source.frames(bag_source.read_bag(args.bag))
        k = (args.frame if args.frame is not None
             else bag_source.hardest_frame(frs))
        k = int(np.clip(k, 0, len(frs) - 1))
        sc = bag_source.to_scenario(frs[k], name=os.path.basename(args.bag.rstrip("/")))
        ref_xy = frs[k].path
        print(f"bag: {len(frs)} cicli · campo centrato sul ciclo {k} "
              f"(t={frs[k].t:.1f} s, J*={frs[k].cost:.0f})")
        hist_bag = {
            "pose": np.array([f.x0[:3] for f in frs], dtype=float),
            "cost": np.array([f.cost for f in frs], dtype=float),
            "solve_ms": np.array([f.solve_ms for f in frs], dtype=float),
            "success": np.array([float(f.success) for f in frs]),
            "pred": np.array([(f.pred if f.pred is not None else f.x0[:2][None, :])
                              for f in frs], dtype=object),
            "ref": np.array([f.path for f in frs], dtype=object),
        }
        # il campo e' costruito attorno al ciclo k, ma la traiettoria copre
        # l'intera missione: senza allargare l'estensione meta' del percorso
        # cadrebbe fuori dal campo (banda bianca nel pannello (b))
        sc = dataclasses.replace(
            sc, extent=extent_covering(sc.extent, hist_bag["pose"][:, :2],
                                       *[r for r in hist_bag["ref"]
                                         if r is not None]))
        focus = k
    else:
        focus = 0
        hist_bag = None
        sc = common.get_scenario(args.scenario)
        ref_xy = common.plan_astar(sc.pose, sc.goal, sc.obstacles, raw)
        if ref_xy is None:
            print("A* non ha trovato un percorso: uso la retta verso il goal")
    xs, ys, X, Y, C = build_field(sc, cfg, args.res, args.reference, ref_xy)

    print(f"scenario '{sc.name}' · profilo N={cfg.N} dt={cfg.dt} "
          f"W_obs={cfg.W_obs_sigmoid:g} obs_r={cfg.obs_r:g}")
    print(f"griglia {X.shape[0]}x{X.shape[1]} a {args.res} m · "
          f"c in [{C.min():.1f}, {C.max():.1f}]")
    mins = local_minima(C, xs, ys)
    print(f"minimi locali di c(x,y): {len(mins)}")
    ref_for_tag = ref_xy if (ref_xy is not None and args.reference == "path") else None
    for (mx, my, mc) in mins:
        m = np.array([mx, my])
        if np.linalg.norm(m - sc.goal) < 0.45:
            tag = "GOAL"
        elif ref_for_tag is not None and \
                np.linalg.norm(ref_for_tag - m, axis=1).min() < 3 * args.res:
            # con reference=path l'intero riferimento e' una valle a costo ~0:
            # un minimo che ci sta sopra e' atteso, non una trappola
            tag = "sulla valle del riferimento A* (atteso)"
        else:
            tag = "*** minimo locale spurio (trappola) ***"
        print(f"   ({mx:+.2f}, {my:+.2f})  c={mc:9.1f}   {tag}")

    P = np.stack([X.ravel(), Y.ravel()], 1)
    tmin = common.reach_time(P, sc.pose, cfg)
    print(f"insieme raggiungibile in {cfg.N*cfg.dt:.1f} s: "
          f"{(tmin <= cfg.N*cfg.dt).sum()} celle, estensione max "
          f"{np.linalg.norm(P[tmin <= cfg.N*cfg.dt] - sc.pose[:2], axis=1).max():.2f} m")

    if hist_bag is not None:
        hist = hist_bag                      # traiettoria REALE, niente simulazione
    else:
        tracker = common.make_tracker(cfg)
        hist = common.closed_loop(tracker, sc, steps=args.steps, raw=raw,
                                  replan_every=args.replan_every)
    reached = np.linalg.norm(hist["pose"][-1, :2] - sc.goal) < 0.35
    print(f"\nanello chiuso: {len(hist['pose'])} cicli · "
          f"goal {'RAGGIUNTO' if reached else 'NON raggiunto'} "
          f"(distanza finale {np.linalg.norm(hist['pose'][-1,:2]-sc.goal):.2f} m)")
    print(f"J*: min {hist['cost'].min():.0f}  max {hist['cost'].max():.0f} · "
          f"solve medio {hist['solve_ms'].mean():.1f} ms "
          f"(budget {cfg.dt*1000:.0f} ms) · successi "
          f"{100*hist['success'].mean():.0f}%")
    cl = common.clearance(hist["pose"][:, :2], sc.obstacles)
    body = 0.35   # raggio di ingombro del G1
    if cl < 0.10:
        verdict = "*** ATTRAVERSA gli ostacoli ***"
    elif cl < body:
        verdict = f"COLLISIONE: sotto il raggio di ingombro {body:g} m"
    elif cl < cfg.obs_r + 0.05:
        verdict = f"al limite: la barriera tiene a obs_r={cfg.obs_r:g} m"
    else:
        verdict = "OK"
    print(f"clearance minima percorsa: {cl:.3f} m   ->   {verdict}")

    # biforcazioni: quante volte A* ha cambiato classe di omotopia
    side = np.array([0 if r is None else (1 if r[:, 1].max() > abs(r[:, 1].min()) else -1)
                     for r in hist["ref"]])
    nz = side[side != 0]
    flips = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
    if len(nz):
        print(f"riferimento A*: parte {'sopra' if nz[0] > 0 else 'sotto'} · "
              f"cambi di lato (biforcazioni): {flips}")

    tag = f"{sc.name}_{os.path.basename(args.profile).replace('.yaml','')}"
    if args.bag:
        tag = f"bag_{tag}"
    if args.animate:
        animate(sc, cfg, hist, xs, ys, X, Y, C, log=not args.linear,
                out=os.path.join(OUT, f"pannello1_{tag}.gif"),
                stride=args.anim_stride)
    figure(sc, cfg, hist, xs, ys, X, Y, C, log=not args.linear,
           out=os.path.join(OUT, f"pannello1_{tag}.png"),
           show=not (args.no_show or not os.environ.get("DISPLAY")),
           focus=focus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
