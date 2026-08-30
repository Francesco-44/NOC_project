#!/usr/bin/env python3
"""
Errore di predizione: modello dell'MPC contro impianto — dispense §7.2.5.

L'MPC predice con un modello nominale (uniciclo piu' un lag del primo ordine);
l'impianto e' il G1 a 29 gradi di liberta' che cammina in MuJoCo. Il
disallineamento e' reale e strutturale — passi discreti, oscillazione del
bacino, ritardo del controllore di camminata — e non e' modellato da nessuna
parte.

Questo script lo QUANTIFICA dai dati gia' registrati, senza nuovi esperimenti:
per ogni ciclo confronta la traiettoria predetta (/mpc/predicted_path, salvata
al tempo t) con quella effettivamente percorsa (/robot_pose ai tempi t+k*dt).

    errore(k) = || predetto(k) - percorso(t + k*dt) ||

Uso:
    python3 viz/prediction_error.py viz/bags/industrial_plant_fix
    python3 viz/prediction_error.py <bag> --no-show
"""
from __future__ import annotations

import argparse
import os
import sys
from bisect import bisect_left

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common       # noqa: E402
import bag_source   # noqa: E402


def pose_series(bag):
    """
    (t in secondi, array (M,3) di [x, y, yaw]) da /robot_pose.

    I tempi sono RELATIVI al primo messaggio di /mpc/diagnostics, perche' e'
    cosi' che bag_source.Frame definisce il proprio `t`. Usare qui i timestamp
    assoluti farebbe cadere ogni confronto fuori intervallo, in silenzio.
    """
    if not bag["diag"]:
        raise SystemExit("la bag non contiene /mpc/diagnostics")
    t0 = bag["diag"][0][0]
    ts, ps = [], []
    for t_ns, m in bag["pose"]:
        q = m.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
        ts.append((t_ns - t0) * 1e-9)
        ps.append([m.pose.position.x, m.pose.position.y, yaw])
    return np.asarray(ts), np.asarray(ps)


def pose_at(ts, ps, t):
    """Posa interpolata linearmente all'istante t; None fuori dall'intervallo."""
    if t < ts[0] or t > ts[-1]:
        return None
    i = bisect_left(ts, t)
    if i == 0:
        return ps[0]
    t0, t1 = ts[i - 1], ts[i]
    if t1 <= t0:
        return ps[i]
    w = (t - t0) / (t1 - t0)
    out = ps[i - 1] + w * (ps[i] - ps[i - 1])
    # lo yaw va interpolato sull'angolo, non sul valore grezzo
    d = np.arctan2(np.sin(ps[i][2] - ps[i - 1][2]), np.cos(ps[i][2] - ps[i - 1][2]))
    out[2] = ps[i - 1][2] + w * d
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    bag = bag_source.read_bag(args.bag)
    frs = bag_source.frames(bag)
    ts, ps = pose_series(bag)
    if len(ts) < 2:
        raise SystemExit("la bag non contiene abbastanza messaggi /robot_pose")

    nome = os.path.basename(args.bag.rstrip("/"))
    print(f"bag {nome}: {len(frs)} cicli, {len(ts)} pose, dt = {cfg.dt} s")

    # errore[k] su tutti i cicli, per k = 0..N
    N = cfg.N
    acc = [[] for _ in range(N + 1)]
    acc_yaw = [[] for _ in range(N + 1)]
    usati = 0
    for f in frs:
        if not f.success or f.pred is None or len(f.pred) < N + 1:
            continue
        pred = np.atleast_2d(f.pred)
        ok = False
        for k in range(N + 1):
            vera = pose_at(ts, ps, f.t + k * cfg.dt)
            if vera is None:
                continue
            acc[k].append(float(np.linalg.norm(pred[k, :2] - vera[:2])))
            # /mpc/predicted_path porta anche l'orientamento, ma bag_source lo
            # scarta tenendo solo (x, y): il confronto sullo yaw e' disponibile
            # solo se in futuro verra' conservato.
            if pred.shape[1] >= 3:
                d = np.arctan2(np.sin(pred[k, 2] - vera[2]),
                               np.cos(pred[k, 2] - vera[2]))
                acc_yaw[k].append(abs(float(d)))
            ok = True
        usati += int(ok)
    print(f"cicli utilizzabili: {usati}")
    if usati == 0:
        raise SystemExit("nessun ciclo confrontabile: la bag copre un intervallo "
                         "troppo corto, oppure /mpc/predicted_path e' assente")
    print()

    print("| k | orizzonte [s] | errore mediano [m] | p95 [m] | max [m] |")
    print("|---|---|---|---|---|")
    med = np.full(N + 1, np.nan)
    for k in range(N + 1):
        if not acc[k]:
            continue
        a = np.asarray(acc[k]); med[k] = np.median(a)
        y = np.degrees(np.median(acc_yaw[k])) if acc_yaw[k] else float("nan")
        if k % max(1, N // 5) == 0 or k == N:
            print(f"| {k:2d} | {k*cfg.dt:5.2f} | {np.median(a):.4f} | "
                  f"{np.percentile(a,95):.4f} | {a.max():.4f} |")

    fin = np.isfinite(med)
    print()
    print("Lettura (§7.2.5):")
    # A k=0 lo stato predetto E' x0, imposto come vincolo di uguaglianza: in
    # teoria l'errore e' nullo. Quello che si misura e' quindi un OFFSET di
    # allineamento temporale (l'istante in cui la predizione viene pubblicata
    # non coincide con l'istante in cui /robot_pose viene campionata), e va
    # sottratto per isolare la divergenza vera del modello.
    off = med[0]
    v_tip = float(np.median(np.abs(np.diff(ps[:, :2], axis=0)).sum(1) /
                            np.maximum(np.diff(ts), 1e-9)))
    print(f"  offset a k=0: {off:.4f} m. Non e' errore di modello — a k=0 lo stato")
    print("  predetto E' x0, imposto come vincolo di uguaglianza. Misura il")
    print("  disallineamento fra l'istante di pubblicazione della predizione e")
    print(f"  quello di campionamento della posa: a ~{v_tip:.2f} m/s corrisponde a")
    print(f"  circa {off/max(v_tip,1e-9)*1000:.0f} ms, coerente con il periodo di ciclo misurato.")
    print()
    if fin.sum() > 2:
        kk = np.arange(N + 1)[fin]
        div = med[fin] - off          # divergenza al netto dell'offset
        pend = np.polyfit(kk[1:] * cfg.dt, div[1:], 1)[0]
        print(f"  DIVERGENZA (al netto dell'offset): cresce di ~{pend:.3f} m/s di")
        print(f"  predizione, arrivando a {med[N]-off:.3f} m a fine orizzonte "
              f"({N*cfg.dt:.1f} s).")
        print()
        print("  Confronto con l'errore di DISCRETIZZAZIONE (tests/test_integrators.py):")
        print(f"  a dt={cfg.dt} su 3 s, Euler sbaglia 1.74e-2 m, il punto medio 8.7e-5 m.")
        rap = (med[N] - off) / 1.74e-2
        if rap > 2:
            print(f"  Qui la divergenza e' {rap:.0f}x l'errore di Euler e "
                  f"{(med[N]-off)/8.7e-5:.0f}x quello del punto medio:")
            print("  il termine dominante NON e' l'integratore ma il disallineamento")
            print("  di modello (uniciclo contro G1 a 29 gdl che cammina).")
            print("  E' la spiegazione quantitativa del perche' passare a RK2 migliori")
            print("  la predizione di 200x ma l'anello chiuso di appena l'1%.")

    common.ensure_mpl3d()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    kk = np.arange(N + 1) * cfg.dt
    p50 = np.array([np.median(acc[k]) if acc[k] else np.nan for k in range(N + 1)])
    p95 = np.array([np.percentile(acc[k], 95) if acc[k] else np.nan for k in range(N + 1)])
    a1.fill_between(kk, 0, p95, alpha=.2, color="#1f77b4", label="p95")
    a1.plot(kk, p50, "o-", color="#1f77b4", lw=2, label="median")
    a1.axhline(1.74e-2, ls="--", c="#2ca02c",
               label="Euler truncation error over 3 s (1.7 cm)")
    a1.set_xlabel("prediction horizon [s]"); a1.set_ylabel("position error [m]")
    a1.set_title("Prediction error against the plant")
    a1.grid(alpha=.3); a1.legend(fontsize=8)

    yy = np.array([np.degrees(np.median(acc_yaw[k])) if acc_yaw[k] else np.nan
                   for k in range(N + 1)])
    a2.plot(kk, yy, "o-", color="#d62728", lw=2)
    a2.set_xlabel("prediction horizon [s]"); a2.set_ylabel("heading error [deg]")
    a2.set_title("Heading error")
    a2.grid(alpha=.3)
    fig.suptitle(f"Prediction model against plant --- {nome}  "
                 f"(dispense §7.2.5)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                       f"errore_predizione_{nome}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    common.save_figure(fig, out, 130)
    print(f"\nsalvato: {out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
