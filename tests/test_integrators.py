#!/usr/bin/env python3
"""
Ordine di troncamento dello schema di integrazione del canale di posizione.

Riferimento: dispense §2.1.3, eq. (2.9) Euler in avanti ed eq. (2.10) regola del
punto medio (RK2). Errore globale atteso: O(dt) per Euler, O(dt^2) per il punto
medio. Il test misura l'ordine con un fit log-log e verifica che sia quello.

La soluzione esatta e' disponibile in forma chiusa: con (vx, vy, omega) costanti
nel riferimento solidale, la posa evolve lungo un arco di circonferenza.

    python3 tests/test_integrators.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "a_star_mpc_planner"))


def exact_step(pose, v, dt):
    """
    Posa esatta dopo dt con velocita' costanti nel corpo.

    pose = (px, py, yaw), v = (vx, vy, omega).
    Per omega != 0 lo spostamento nel mondo e'
        d = (1/w) R(yaw) [[sin a, cos a - 1], [1 - cos a, sin a]] [vx, vy]^T
    con a = w*dt; per w -> 0 degenera nella traslazione rigida R(yaw) v dt.
    """
    px, py, yaw = pose
    vx, vy, w = v
    if abs(w) < 1e-12:
        c, s = math.cos(yaw), math.sin(yaw)
        return (px + (vx * c - vy * s) * dt,
                py + (vx * s + vy * c) * dt,
                yaw)
    a = w * dt
    M = np.array([[math.sin(a),      math.cos(a) - 1.0],
                  [1.0 - math.cos(a), math.sin(a)]]) / w
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    d = R @ (M @ np.array([vx, vy]))
    return (px + d[0], py + d[1], yaw + a)


def euler_step(pose, v, dt):
    px, py, yaw = pose
    vx, vy, w = v
    c, s = math.cos(yaw), math.sin(yaw)          # yaw a INIZIO intervallo
    return (px + (vx * c - vy * s) * dt,
            py + (vx * s + vy * c) * dt,
            yaw + w * dt)


def midpoint_step(pose, v, dt):
    px, py, yaw = pose
    vx, vy, w = v
    yaw_eval = yaw + 0.5 * w * dt                # yaw a META' intervallo
    c, s = math.cos(yaw_eval), math.sin(yaw_eval)
    return (px + (vx * c - vy * s) * dt,
            py + (vx * s + vy * c) * dt,
            yaw + w * dt)


def integrate(step, pose0, v, dt, T):
    """
    Integra per un tempo T con passo dt e restituisce la posa finale.

    dt DEVE dividere T: con un numero non intero di passi si finirebbe a un
    istante diverso da quello del riferimento esatto, e l'errore misurato
    sarebbe dominato dal disallineamento dell'orizzonte invece che dall'ordine
    dello schema (con dt=0.4 e T=3.0 i due schemi davano lo stesso errore).
    """
    n_exact = T / dt
    n = int(round(n_exact))
    if abs(n_exact - n) > 1e-9:
        raise ValueError(f"dt={dt} non divide T={T}: {n_exact} passi")
    pose = pose0
    for _ in range(n):
        pose = step(pose, v, dt)
    return pose


def fit_order(dts, errs):
    """Pendenza della retta log(err) vs log(dt): e' l'ordine globale."""
    return float(np.polyfit(np.log(np.asarray(dts)), np.log(np.asarray(errs)), 1)[0])


def global_error_table(v, T=3.0, dts=(0.2, 0.1, 0.05, 0.025, 0.0125)):
    """Errore di posizione a orizzonte fisso, per i due schemi."""
    pose0 = (0.0, 0.0, 0.0)
    ref = exact_step(pose0, v, T)
    rows = []
    for dt in dts:
        pe = integrate(euler_step, pose0, v, dt, T)
        pm = integrate(midpoint_step, pose0, v, dt, T)
        rows.append((
            dt,
            math.hypot(pe[0] - ref[0], pe[1] - ref[1]),
            math.hypot(pm[0] - ref[0], pm[1] - ref[1]),
        ))
    return rows


# ---------------------------------------------------------------------------
# Sequenze di ingressi REALI (orizzonti risolti da una bag)
# ---------------------------------------------------------------------------
# Le funzioni sopra misurano l'ordine su un arco a velocita' costante: e' cio'
# che serve per stimare un ordine (che e' una proprieta' asintotica e richiede
# uno sweep su dt) ma NON e' il regime dell'MPC, che applica un ingresso diverso
# a ogni nodo. Quelle che seguono ripetono la misura sulla sequenza di ingressi
# ottima di un orizzonte vero, letta da viz/integrator_bag.py.
#
# COSA VIENE ISOLATO. Il canale di velocita' e' ZOH esatto a qualunque passo, e
# sul profilo deployato (tau = 1 ms << dt) satura: v_{k+1} = u_k. La sequenza di
# velocita' post-lag e' quindi calcolata UNA volta al passo deployato e tenuta
# costante su ciascun intervallo, per tutti gli schemi e per il riferimento.
# Cosi' l'unica differenza fra le tre traiettorie e' COME si valuta R(psi), che
# e' esattamente la domanda posta. Includere il transitorio del lag nel
# riferimento misurerebbe un'altra cosa (vedi lag_displacement).


def lag_coeffs(dt, tau_v, tau_w):
    """1 - exp(-dt/tau) per i due canali: gli stessi di mpc_tracker._passo."""
    return (1.0 - math.exp(-dt / max(tau_v, 1e-9)),
            1.0 - math.exp(-dt / max(tau_w, 1e-9)))


def velocity_sequence(x0, U, dt, tau_v, tau_w):
    """Velocita' post-lag lungo l'orizzonte. ZOH esatto: nessuna approssimazione."""
    lv, lw = lag_coeffs(dt, tau_v, tau_w)
    vx, vy, wz = float(x0[3]), float(x0[4]), float(x0[5])
    out = []
    for u in U:
        vx = (1.0 - lv) * vx + lv * float(u[0])
        vy = (1.0 - lw) * vy + lw * float(u[1])
        wz = (1.0 - lw) * wz + lw * float(u[2])
        out.append((vx, vy, wz))
    return out


def pose_track(pose0, V, dt, step, sub=1):
    """Posa lungo l'orizzonte, integrata con `step` a passo dt/sub.

    `sub` suddivide l'intervallo di predizione SENZA toccare l'ingresso, che
    resta costante su ciascun dt: e' cosi' che si fa variare il passo di
    integrazione a sequenza di comandi fissata, e quindi si misura un ordine su
    una traiettoria vera.
    """
    h = dt / sub
    pose = (float(pose0[0]), float(pose0[1]), float(pose0[2]))
    out = [pose]
    for v in V:
        for _ in range(sub):
            pose = step(pose, v, h)
        out.append(pose)
    return out


def pose_track_exact(pose0, V, dt):
    """Riferimento: arco esatto su ciascun intervallo a velocita' costante."""
    pose = (float(pose0[0]), float(pose0[1]), float(pose0[2]))
    out = [pose]
    for v in V:
        pose = exact_step(pose, v, dt)
        out.append(pose)
    return out


def horizon_errors(x0, U, dt, tau_v, tau_w, subs=(1, 2, 4, 8, 16)):
    """Errore di posizione lungo un orizzonte vero, per i due schemi.

    Ritorna {'sub': [...], 'dt': [...], 'euler': [...], 'midpoint': [...]} con
    l'errore FINALE (a fine orizzonte) contro l'arco esatto, uno per passo.
    """
    V = velocity_sequence(x0, U, dt, tau_v, tau_w)
    ref = pose_track_exact(x0[:3], V, dt)
    rx, ry = ref[-1][0], ref[-1][1]
    out = {"sub": list(subs), "dt": [dt / s for s in subs],
           "euler": [], "midpoint": []}
    for s in subs:
        for nome, step in (("euler", euler_step), ("midpoint", midpoint_step)):
            tr = pose_track(x0[:3], V, dt, step, sub=s)
            out[nome].append(math.hypot(tr[-1][0] - rx, tr[-1][1] - ry))
    return out


def lag_displacement(x0, U, dt, tau_v, tau_w):
    """Spostamento trascurato tenendo v costante a v_{k+1} sull'intervallo.

    Il modello dell'MPC risolve v' = (u - v)/tau, quindi dentro l'intervallo la
    velocita' SALE verso u invece di essere gia' arrivata. L'integrale della
    differenza vale (u - v_k) * tau * (1 - e^{-dt/tau}) per intervallo. Sul
    profilo deployato tau e' 1 ms e il termine e' sub-millimetrico, ma e' dello
    stesso ordine dell'errore del punto medio: serve a dire dove sta il
    pavimento sotto cui raffinare lo schema non compra piu' nulla.
    """
    lv, lw = lag_coeffs(dt, tau_v, tau_w)
    vx, vy, wz = float(x0[3]), float(x0[4]), float(x0[5])
    tot = 0.0
    for u in U:
        dvx, dvy = float(u[0]) - vx, float(u[1]) - vy
        tot += math.hypot(dvx * tau_v * lv, dvy * tau_w * lw)
        vx = (1.0 - lv) * vx + lv * float(u[0])
        vy = (1.0 - lw) * vy + lw * float(u[1])
        wz = (1.0 - lw) * wz + lw * float(u[2])
    return tot


def test_orders():
    """Euler ~ ordine 1, punto medio ~ ordine 2, su piu' regimi di rotazione."""
    # (vx, vy, omega) — i limiti deployati sul G1 sono vx<=0.3, vy<=0.02, |w|<=0.3
    regimi = {
        "G1 nominale (vx=0.2, w=0.3)":  (0.20, 0.00, 0.30),
        "G1 con deriva laterale":       (0.20, 0.02, 0.30),
        "rotazione rapida (w=1.0)":     (0.30, 0.00, 1.00),
    }
    ok = True
    for nome, v in regimi.items():
        rows = global_error_table(v)
        dts = [r[0] for r in rows]
        o_e = fit_order(dts, [r[1] for r in rows])
        o_m = fit_order(dts, [r[2] for r in rows])
        print(f"\n{nome}")
        print("| dt [s] | errore Euler [m] | errore punto medio [m] | rapporto |")
        print("|---|---|---|---|")
        for dt, ee, em in rows:
            print(f"| {dt:.3f} | {ee:.3e} | {em:.3e} | {ee/em:.0f}x |")
        print(f"ordine stimato:  Euler {o_e:.2f}   punto medio {o_m:.2f}")
        if not (0.85 <= o_e <= 1.15):
            print(f"  FALLITO: ordine di Euler {o_e:.2f} fuori da [0.85, 1.15]")
            ok = False
        if not (1.85 <= o_m <= 2.15):
            print(f"  FALLITO: ordine del punto medio {o_m:.2f} fuori da [1.85, 2.15]")
            ok = False
    return ok


def test_matches_nlp():
    """
    Lo schema del test coincide con quello dentro l'NLP.

    Non basta che la teoria torni: serve che mpc_tracker integri davvero cosi'.
    Si confronta un passo di dinamica estratto dall'NLP con lo step di riferimento,
    per entrambi gli schemi.
    """
    import casadi as ca
    from a_star_mpc_planner.mpc_tracker import MPCConfig, MPCTracker

    dt = 0.2
    # tau << dt  =>  lag = 1  =>  v_next = u  : cosi' il confronto isola px/py
    base = dict(N=2, dt=dt, tau_v=1e-3, tau_w=1e-3, vx_max=1.0, vy_max=1.0,
                omega_max=2.0, W_obs_sigmoid=0.0, max_obs_constraints=1)
    pose0 = (0.3, -0.2, 0.4)
    u = (0.25, 0.03, 0.6)
    ok = True

    for schema, ref_step in (("euler", euler_step), ("midpoint", midpoint_step)):
        tr = MPCTracker(MPCConfig(integrator=schema, **base))
        tr._build_nlp()
        opti = tr._opti
        # _X[:,1] e' una variabile, non un'espressione: la mappa f(X0,U0) si
        # ricostruisce dai vincoli di uguaglianza. I primi 6 sono, in ordine,
        # X[i,1] - f_i(X[:,0], U[:,0]) per i = 0..5.
        f_expr = tr._X[:, 1] - opti.g[:6]
        # Uno slice di matrice non e' "purely symbolic": ca.Function lo rifiuta.
        # Si valuta quindi assegnando i valori iniziali e leggendo l'espressione.
        opti.set_initial(tr._X[:, 0], [pose0[0], pose0[1], pose0[2], 0.0, 0.0, 0.0])
        opti.set_initial(tr._U[:, 0], list(u))
        got = np.array(opti.debug.value(f_expr, opti.initial())).ravel()[:3]
        want = np.array(ref_step(pose0, u, dt))
        err = float(np.max(np.abs(got - want)))
        print(f"\nNLP '{schema}': scarto dallo schema di riferimento = {err:.3e}")
        if err > 1e-12:
            print(f"  FALLITO: l'NLP non integra come '{schema}'")
            print(f"    NLP  {got}")
            print(f"    atteso {want}")
            ok = False
    return ok


def main():
    print("=" * 72)
    print("Ordine di troncamento — dispense §2.1.3, eq. (2.9) vs (2.10)")
    print("=" * 72)
    ok = test_orders()
    print()
    print("=" * 72)
    print("Coerenza con la dinamica costruita da MPCTracker")
    print("=" * 72)
    ok = test_matches_nlp() and ok
    print()
    print("ESITO:", "OK" if ok else "FALLITO")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
