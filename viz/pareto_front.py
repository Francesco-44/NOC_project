#!/usr/bin/env python3
"""
Fronte di Pareto sui tre obiettivi del path following — dispense §7.4.

Il corso prescrive una procedura precisa per il multi-obiettivo a posteriori:

  (I)   normalizzare gli obiettivi allo stesso ordine di grandezza
  (II)  risolvere ripetutamente campionando i pesi sul simplesso
        A = {alpha >= 0, sum alpha_i = 1}, includendo i VERTICI
  (III) post-processare: punti non dominati, punto Utopico, scelta come punto
        piu' vicino all'Utopico in norma 2

E avverte che la somma pesata recupera il fronte completo solo se questo e'
CONVESSO — cosa che va verificata, non assunta.

Qui i tre obiettivi sono quelli che la eq. (7.5) introduce naturalmente:

  alpha_1  accuratezza geometrica   (pesi Q sul tracking)
  alpha_2  sforzo di controllo      (pesi R sull'ingresso)
  alpha_3  avanzamento sul percorso (peso su (1 - theta)^2)

I pesi vengono scalati per 3 cosi' che il baricentro (1/3, 1/3, 1/3) riproduca
esattamente la taratura di partenza.

ATTENZIONE METODOLOGICA. Le METRICHE con cui si valutano le soluzioni usano
pesi FISSI, non quelli campionati: altrimenti ogni punto del simplesso sarebbe
giudicato con un metro diverso e il confronto non avrebbe senso.

Uso:
    python3 viz/pareto_front.py
    python3 viz/pareto_front.py --risoluzione 5 --scenari narrow_gap corridor
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402

T_MISSIONE = 30.0
NOMI = ("accuratezza", "sforzo", "tempo")


def simplesso(n: int):
    """Griglia sul simplesso a 3 componenti, VERTICI INCLUSI (punto II)."""
    pts = []
    for i, j in itertools.product(range(n + 1), repeat=2):
        k = n - i - j
        if k < 0:
            continue
        pts.append((i / n, j / n, k / n))
    return pts


def valuta(cfg, raw, sc, alpha) -> dict:
    """Una missione con i pesi alpha; le metriche usano pesi FISSI."""
    a1, a2, a3 = alpha
    c = dataclasses.replace(
        cfg,
        path_mode='theta',
        Q_x=cfg.Q_x * 3 * a1, Q_y=cfg.Q_y * 3 * a1, Q_yaw=cfg.Q_yaw * 3 * a1,
        R_vx=cfg.R_vx * 3 * a2, R_vy=cfg.R_vy * 3 * a2, R_omega=cfg.R_omega * 3 * a2,
        theta_progress_weight=cfg.theta_progress_weight * 3 * a3,
        max_iter=200,
    )
    tr = common.make_tracker(c)
    steps = max(5, int(round(T_MISSIONE / cfg.dt)))
    h = common.closed_loop(tr, sc, steps=steps, raw=raw)
    P = np.asarray(h["pose"], dtype=float)
    raggiunto = bool(len(P) < steps)

    # --- metriche a pesi FISSI ------------------------------------------
    # accuratezza: distanza media dal riferimento geometrico (il path), non dal
    # riferimento temporale — e' la grandezza che il path following dovrebbe
    # migliorare, e non dipende da come e' parametrizzato il tempo.
    ref = sc.reference()
    d = np.linalg.norm(P[:, None, :2] - ref[None, :, :2], axis=2).min(axis=1)
    acc = float(d.mean())
    # sforzo: velocita' comandate ricostruite dal moto, con i pesi NOMINALI
    dP = np.diff(P[:, :2], axis=0) / cfg.dt
    dW = np.diff(np.unwrap(P[:, 2])) / cfg.dt
    sforzo = float((cfg.R_vx * (dP ** 2).sum(1) + cfg.R_omega * dW ** 2).mean())
    tempo = float(len(P) * cfg.dt)
    return {
        "alpha": list(alpha), "goal": raggiunto,
        "accuratezza": acc, "sforzo": sforzo, "tempo": tempo,
        "clearance": float(common.clearance(P[:, :2], sc.obstacles)),
    }


def non_dominati(F: np.ndarray) -> np.ndarray:
    """Maschera dei punti non dominati (tutti gli obiettivi da MINIMIZZARE)."""
    m = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not m[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            m[i] = False
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--scenari", nargs="*", default=["narrow_gap"])
    ap.add_argument("--risoluzione", type=int, default=4,
                    help="passi per lato del simplesso (4 -> 15 punti)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    pts = simplesso(args.risoluzione)
    print(f"simplesso a {len(pts)} punti (vertici inclusi) su "
          f"{len(args.scenari)} scenari · path_mode = theta")
    print(f"baricentro (1/3,1/3,1/3) = taratura di partenza "
          f"(Q={cfg.Q_x:g}, R={cfg.R_vx:g}, w_theta={cfg.theta_progress_weight:g})")
    print()

    righe = []
    t0 = time.perf_counter()
    for nome in args.scenari:
        sc = common.SCENARIOS[nome]()
        for al in pts:
            r = valuta(cfg, raw, sc, al)
            r["scenario"] = nome
            righe.append(r)
            print(f"  α=({al[0]:.2f},{al[1]:.2f},{al[2]:.2f}) "
                  f"acc={r['accuratezza']:.4f} sforzo={r['sforzo']:.4f} "
                  f"t={r['tempo']:5.1f} goal={'si' if r['goal'] else 'NO'}", flush=True)
    print(f"\ndurata {time.perf_counter()-t0:.0f} s")

    # aggregazione sugli scenari, solo missioni riuscite
    agg = {}
    for al in pts:
        sel = [r for r in righe if tuple(r["alpha"]) == tuple(al)]
        if not sel or not all(r["goal"] for r in sel):
            continue
        agg[tuple(al)] = {k: float(np.mean([r[k] for r in sel]))
                          for k in ("accuratezza", "sforzo", "tempo", "clearance")}
    if len(agg) < 3:
        raise SystemExit("troppe poche missioni riuscite per costruire un fronte")

    A = np.array(list(agg.keys()))
    F = np.array([[v["accuratezza"], v["sforzo"], v["tempo"]] for v in agg.values()])

    # (I) normalizzazione: senza, il tempo (~10) schiaccerebbe l'accuratezza (~0.1)
    lo, hi = F.min(0), F.max(0)
    Fn = (F - lo) / np.where(hi - lo < 1e-12, 1.0, hi - lo)

    # (III) non dominati, Utopico, scelta
    nd = non_dominati(Fn)
    utop = Fn.min(0)                      # punto Utopico: migliore su ogni obiettivo
    dist = np.linalg.norm(Fn - utop, axis=1)
    best = int(np.argmin(np.where(nd, dist, np.inf)))

    print()
    print("=" * 78)
    print("FRONTE DI PARETO  (§7.4)")
    print("=" * 78)
    print(f"missioni riuscite: {len(agg)}/{len(pts)} · non dominate: {int(nd.sum())}")
    print(f"punto Utopico (normalizzato): {np.round(utop,3)} — per costruzione non")
    print("e' realizzabile: e' il migliore su OGNI obiettivo preso separatamente.")
    print()
    print("| α (acc, sforzo, tempo) | accuratezza [m] | sforzo | tempo [s] | "
          "clearance [m] | dist. da Utopico |")
    print("|---|---|---|---|---|---|")
    ordine = np.argsort(dist)
    for i in ordine:
        if not nd[i]:
            continue
        v = list(agg.values())[i]
        mark = "  ← **scelto**" if i == best else ""
        print(f"| ({A[i,0]:.2f}, {A[i,1]:.2f}, {A[i,2]:.2f}) | {v['accuratezza']:.4f} | "
              f"{v['sforzo']:.4f} | {v['tempo']:.1f} | {v['clearance']:.3f} | "
              f"{dist[i]:.3f}{mark} |")

    print()
    ab = A[best]
    print(f"Scelta: α = ({ab[0]:.2f}, {ab[1]:.2f}, {ab[2]:.2f}), il punto non dominato")
    print("piu' vicino all'Utopico in norma 2 (procedura del §7.4).")
    # confronto col baricentro, cioe' la taratura di partenza
    j = int(np.argmin(np.linalg.norm(A - 1.0 / 3.0, axis=1)))
    print(f"Per confronto, il baricentro α≈(0.33,0.33,0.33) — la taratura attuale — "
          f"dista {dist[j]:.3f} ed e' {'non dominato' if nd[j] else 'DOMINATO'}.")

    # convessita' del fronte: si verifica se i punti non dominati stanno sul
    # guscio convesso inferiore. Se non lo sono, la somma pesata NON puo'
    # raggiungerli, e il corso avverte esattamente di questo.
    P2 = Fn[nd][:, [0, 2]]                     # coppia accuratezza-tempo
    conv = True
    if len(P2) >= 3:
        o = np.argsort(P2[:, 0]); Q = P2[o]
        for a, b, c in zip(Q, Q[1:], Q[2:]):
            # cross product: se cambia segno la frontiera non e' convessa
            if np.cross(b - a, c - b) > 1e-9:
                conv = False
                break
    # Un fronte esiste sempre; la domanda e' se sia INFORMATIVO. Se gli
    # obiettivi variano di pochi punti percentuali non sono in vero conflitto,
    # e "non dominato" smette di essere una distinzione utile.
    spread = (F.max(0) - F.min(0)) / np.maximum(np.abs(F.mean(0)), 1e-12)
    print()
    print("Escursione relativa degli obiettivi sul simplesso:")
    for nm, sp in zip(NOMI, spread):
        print(f"  {nm:12s} {sp*100:5.1f}%")
    if spread.max() < 0.15:
        print()
        print("  Il fronte e' SOTTILE: nessun obiettivo varia piu' del "
              f"{spread.max()*100:.0f}% al variare dei pesi.")
        print("  I tre obiettivi non sono in vero conflitto in questa")
        print("  configurazione, per due ragioni identificabili:")
        print("   1. in modo theta il robot satura vx_max quasi sempre (§10.8),")
        print("      quindi il tempo di percorrenza e' fissato dalla cinematica")
        print("      e non dai pesi;")
        print("   2. l'anello chiuso insegue un setpoint a distanza di lookahead")
        print("      con un controllore proporzionale, che smorza le differenze")
        print("      fini fra le soluzioni dell'MPC.")
        print("  Conclusione onesta: la taratura non e' il collo di bottiglia.")
        print("  Un fronte informativo richiederebbe obiettivi che confliggano")
        print("  davvero — per esempio clearance contro tempo con vx libera.")

    print()
    print(f"Fronte (accuratezza vs tempo) convesso: **{conv}**.")
    if not conv:
        print("  La somma pesata NON puo' raggiungere le porzioni non convesse:")
        print("  i punti mancanti richiederebbero la strategia a vincoli (eq. 7.8).")

    out_dir = os.path.join(_HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pareto_front.json"), "w") as fh:
        json.dump({"punti": [{"alpha": list(k), **v} for k, v in agg.items()],
                   "non_dominati": nd.tolist(), "scelto": A[best].tolist(),
                   "utopico_normalizzato": utop.tolist(),
                   "fronte_convesso": bool(conv),
                   "escursione_relativa": dict(zip(NOMI, spread.tolist())),
                   "fronte_informativo": bool(spread.max() >= 0.15)},
                  fh, indent=2, default=float)

    # ── figure: curva di Pareto (Fig. 7.9) + spider chart (Fig. 7.10) ────
    common.ensure_mpl3d()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 4.4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(F[~nd, 0], F[~nd, 2], s=28, c="#bbbbbb", label="dominated")
    ax1.scatter(F[nd, 0], F[nd, 2], s=52, c="#1f77b4", label="non-dominated front")
    ax1.scatter(F[best, 0], F[best, 2], s=150, marker="*", c="#d62728",
                label="selected (nearest to utopia)", zorder=5)
    ax1.scatter(lo[0], lo[2], s=110, marker="P", c="#2ca02c",
                label="utopia point", zorder=5)
    ax1.set_xlabel("accuracy: mean distance from path [m]")
    ax1.set_ylabel("time to goal [s]")
    ax1.set_title("Pareto curve", fontsize=10)
    ax1.grid(alpha=.3); ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(F[~nd, 1], F[~nd, 2], s=28, c="#bbbbbb")
    ax2.scatter(F[nd, 1], F[nd, 2], s=52, c="#1f77b4")
    ax2.scatter(F[best, 1], F[best, 2], s=150, marker="*", c="#d62728", zorder=5)
    ax2.set_xlabel("control effort"); ax2.set_ylabel("time to goal [s]")
    ax2.set_title("effort against time", fontsize=10)
    ax2.grid(alpha=.3)

    ax3 = fig.add_subplot(1, 3, 3, projection="polar")
    ang = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    ang += ang[:1]
    idx_nd = np.nonzero(nd)[0]
    scelti = list(idx_nd[np.argsort(dist[idx_nd])][:3])
    for i in scelti:
        # nello spider chart 1 = migliore, cosi' "piu' grande e' meglio"
        v = (1.0 - Fn[i]).tolist(); v += v[:1]
        lbl = f"α=({A[i,0]:.2f},{A[i,1]:.2f},{A[i,2]:.2f})"
        ax3.plot(ang, v, lw=2, label=lbl + (" ←" if i == best else ""))
        ax3.fill(ang, v, alpha=.12)
    ax3.set_xticks(ang[:-1]); ax3.set_xticklabels(NOMI, fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.set_title("Spider chart\n1 = best", fontsize=10)
    ax3.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.35, 1.15))

    fig.suptitle("Multi-objective scalarisation of the three cost blocks", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, "pareto_front.png")
    common.save_figure(fig, out, 130)
    print(f"\nsalvati:\n  {out}\n  {os.path.join(out_dir,'pareto_front.json')}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
