#!/usr/bin/env python3
"""
Analisi KKT dell'NLP risolto dall'MPC — dispense §6.1.

Verifica su un ciclo di controllo reale (estratto da una bag) le condizioni che
il corso enuncia in astratto:

  §6.1.1  LICQ (Def. 6.1.5)  — i gradienti dei vincoli attivi sono indipendenti
  §6.1.2  KKT (eq. 6.8)      — stazionarieta' della lagrangiana e complementarita'
  §6.1.3  SOC-C-2 (Thm 6.1.6)— Hessiana della lagrangiana definita positiva sul
                               cono critico  =>  certificato di minimo LOCALE

Uso:
    python3 viz/kkt_analysis.py --bag viz/bags/industrial_plant_fix
    python3 viz/kkt_analysis.py --scenario centred_pillar
    python3 viz/kkt_analysis.py --bag <bag> --frame 300 --set mpc_W_obs_sigmoid=600
"""
from __future__ import annotations

import argparse
import os
import sys

import casadi as ca
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# Un moltiplicatore sotto questa soglia si considera nullo: IPOPT e' un metodo di
# punto interno, i mu dei vincoli inattivi tendono a zero ma non ci arrivano.
TOL_MU = 1e-6
# Un vincolo si considera attivo se dista da un suo limite meno di questa soglia.
# IPOPT si ferma sulla central path: all'ottimo i vincoli attivi distano ~1e-8
# dal bordo, non zero.
TOL_ACT = 1e-6


def solve_and_extract(tracker, x0, path, obstacles):
    """Risolve e restituisce (opti, soluzione) con i duali disponibili."""
    res = tracker.solve(np.asarray(x0, float), path, obstacle_points_2d=obstacles)
    if not res.success:
        print("ATTENZIONE: il solve non e' riuscito; l'analisi usa opti.debug")
    return tracker._opti, res


def classify_constraints(opti):
    """
    Separa uguaglianze e disuguaglianze leggendo i LIMITI, non l'espressione.

    Opti canonizza ogni vincolo come  lbg <= g(x) <= ubg,  e quando il membro
    destro e' un parametro lo assorbe nei limiti: `X[:,0] == p_x0` diventa
    g = X[:,0] con lbg = ubg = p_x0. Valutare |g| su quelle righe restituisce
    quindi LO STATO, non il residuo. Il residuo e' sempre g - lbg.

    Restituisce (g, lbg, ubg, lam, is_eq).
    """
    d = opti.debug
    g = np.array(d.value(opti.g)).ravel()
    lbg = np.array(d.value(opti.lbg)).ravel()
    ubg = np.array(d.value(opti.ubg)).ravel()
    lam = np.array(d.value(opti.lam_g)).ravel()
    return g, lbg, ubg, lam, (lbg == ubg)


def active_mask(g, lbg, ubg, is_eq, tol=TOL_ACT):
    """
    True dove una disuguaglianza tocca uno dei suoi limiti finiti.

    Due righe distinte possono condividere la stessa espressione g con limiti
    diversi (`U[0,k] >= 0` e `U[0,k] <= vx_max` sono entrambe la riga U[0,k]):
    e' per questo che l'attivazione va decisa sul limite, non sul valore.
    """
    at_lo = np.isfinite(lbg) & (np.abs(g - lbg) < tol)
    at_hi = np.isfinite(ubg) & (np.abs(ubg - g) < tol)
    return (~is_eq) & (at_lo | at_hi), at_lo, at_hi


def analyze(cfg, x0, path, obs) -> dict:
    """
    Tutte le grandezze KKT di UN ciclo, in forma strutturata.

    Separata da main() apposta: il generatore di risultati
    (viz/make_results.py) deve poter riusare esattamente questo calcolo, non
    una sua copia — due implementazioni della stessa misura divergono al primo
    ritocco, ed e' cosi' che i numeri di un report smettono di corrispondere al
    codice.
    """
    tracker = common.make_tracker(cfg)
    res = tracker.solve(np.asarray(x0, float), path, obstacle_points_2d=obs)
    opti = tracker._opti
    N = cfg.N

    g, lbg, ubg, lam, is_eq = classify_constraints(opti)
    active_all, at_lo, at_hi = active_mask(g, lbg, ubg, is_eq)
    n_eq = int(is_eq.sum())
    n_ineq = len(g) - n_eq
    idx_ineq = np.nonzero(~is_eq)[0]
    act_i = active_all[idx_ineq]
    mu_i = lam[idx_ineq]
    strong = act_i & (np.abs(mu_i) > TOL_MU)
    weak = act_i & (np.abs(mu_i) <= TOL_MU)

    # Jacobiano dei vincoli attivi -> LICQ e cono critico
    idx_att = list(np.nonzero(is_eq | active_all)[0])
    J = ca.Function("J", [opti.x, opti.p], [ca.jacobian(opti.g, opti.x)])
    xv, pv = opti.debug.value(opti.x), opti.debug.value(opti.p)
    A = np.array(J(xv, pv))[idx_att, :]
    rank = int(np.linalg.matrix_rank(A))

    # stazionarieta' e Hessiana della lagrangiana
    lam_s = ca.MX.sym("lam", opti.g.shape[0])
    L = opti.f + ca.dot(lam_s, opti.g)
    gL = ca.Function("gL", [opti.x, opti.p, lam_s], [ca.gradient(L, opti.x)])
    r_stat = float(np.abs(np.array(gL(xv, pv, lam)).ravel()).max())
    gf = ca.Function("gf", [opti.x, opti.p], [ca.gradient(opti.f, opti.x)])
    r_gradf = float(np.abs(np.array(gf(xv, pv)).ravel()).max())

    H = ca.Function("H", [opti.x, opti.p, lam_s], [ca.hessian(L, opti.x)[0]])
    Hv = np.array(H(xv, pv, lam)); Hv = 0.5 * (Hv + Hv.T)
    _, sv, Vt = np.linalg.svd(A)
    tol = max(A.shape) * (sv.max() if sv.size else 0.0) * np.finfo(float).eps
    ns = Vt[np.sum(sv > tol):].T
    ev_min = ev_max = float("nan")
    if ns.shape[1]:
        ev = np.linalg.eigvalsh(0.5 * (ns.T @ Hv @ ns + (ns.T @ Hv @ ns).T))
        ev_min, ev_max = float(ev.min()), float(ev.max())

    # ripartizione dei box attivi: per ogni k l'ordine e'
    # [vx>=0, vx<=vx_max, |vy|<=vy_max, |w|<=omega_max]
    etichette = ["vx>=0", "vx<=vx_max", "|vy|<=vy_max", "|w|<=omega_max"]
    ripart = {}
    for j, e in enumerate(etichette):
        sel = np.arange(len(idx_ineq)) % 4 == j
        ripart[e] = int(act_i[sel].sum())

    return {
        "success": bool(res.success),
        "J": float(res.cost),
        "iterazioni": int(res.iterations),
        "n_var": int(opti.x.shape[0]),
        "n_con": int(len(g)),
        "n_eq": n_eq,
        "n_ineq": n_ineq,
        "residuo_eq": float(np.abs(g[is_eq] - lbg[is_eq]).max()),
        "n_attivi_ineq": int(act_i.sum()),
        "n_fortemente_attivi": int(strong.sum()),
        "n_debolmente_attivi": int(weak.sum()),
        "complementarita_stretta": bool(weak.sum() == 0),
        "n_attivi_totali": len(idx_att),
        "rango_jacobiano_attivo": rank,
        "licq": bool(rank == len(idx_att)),
        "dim_cono_critico": int(ns.shape[1]),
        "hess_proj_lambda_min": ev_min,
        "hess_proj_lambda_max": ev_max,
        "soc_c2": bool(np.isfinite(ev_min) and ev_min > 0),
        "grad_L_inf": r_stat,
        "grad_f_inf": r_gradf,
        "max_lambda_eq": float(np.abs(lam[is_eq]).max()),
        "max_mu_ineq": float(np.abs(mu_i[strong]).max()) if strong.any() else 0.0,
        "box_attivi": ripart,
        "N": int(N),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--scenario", default="centred_pillar")
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override di un parametro del YAML, es. mpc_W_obs_sigmoid=600")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, args.overrides)
    tracker = common.make_tracker(cfg)

    if args.bag:
        import bag_source
        frs = bag_source.frames(bag_source.read_bag(args.bag))
        k = args.frame if args.frame is not None else bag_source.hardest_frame(frs)
        f = frs[k]
        x0 = f.x0
        path = [(float(p[0]), float(p[1]), 0.0) for p in f.path]
        obs = np.asarray(f.obstacles, float)
        print(f"bag: ciclo {k}/{len(frs)}  t={f.t:.1f} s  J*={f.cost:.0f}  "
              f"iterazioni={f.iterations}")
    else:
        sc = common.SCENARIOS[args.scenario]()
        x0 = np.array([sc.pose[0], sc.pose[1], sc.pose[2], 0.0, 0.0, 0.0])
        ref = sc.reference()
        path = [(float(p[0]), float(p[1]), 0.0) for p in ref]
        obs = sc.obstacles
        print(f"scenario sintetico '{sc.name}'")

    print(f"profilo N={cfg.N} dt={cfg.dt} W_obs={cfg.W_obs_sigmoid} "
          f"integrator={cfg.integrator}")
    print()

    opti, res = solve_and_extract(tracker, x0, path, obs)
    N = cfg.N

    g, lbg, ubg, lam, is_eq = classify_constraints(opti)
    active_all, at_lo, at_hi = active_mask(g, lbg, ubg, is_eq)
    n_tot = len(g)
    n_eq = int(is_eq.sum())
    n_ineq = n_tot - n_eq
    resid_eq = np.abs(g[is_eq] - lbg[is_eq])

    print("=" * 74)
    print("STRUTTURA  (dispense §6.1.1)")
    print("=" * 74)
    print(f"variabili decisionali n = {int(opti.x.shape[0])}")
    print(f"vincoli m = {n_tot}   ({n_eq} uguaglianze, {n_ineq} disuguaglianze)")
    print(f"residuo massimo sulle uguaglianze |g - lbg|: {resid_eq.max():.3e}")

    # ── Active set fra le disuguaglianze ────────────────────────────────
    idx_ineq = np.nonzero(~is_eq)[0]
    act_i = active_all[idx_ineq]
    mu_i = lam[idx_ineq]
    strong = act_i & (np.abs(mu_i) > TOL_MU)
    weak = act_i & (np.abs(mu_i) <= TOL_MU)

    print()
    print("=" * 74)
    print("ACTIVE SET e COMPLEMENTARITA'  (§6.1.2, eq. 6.8)")
    print("=" * 74)
    print(f"disuguaglianze attive        : {int(act_i.sum())} / {n_ineq}")
    print(f"  fortemente attive (mu > 0) : {int(strong.sum())}")
    print(f"  debolmente attive (mu = 0) : {int(weak.sum())}")
    if weak.any():
        print("  -> complementarita' NON stretta: il cono critico (§6.1.3) non")
        print("     degenera in un sottospazio, e la verifica di SOC-C-2 sul solo")
        print("     nucleo dei vincoli attivi e' NECESSARIA ma non sufficiente.")
    else:
        print("  -> complementarita' stretta: il cono critico coincide con il")
        print("     nucleo del Jacobiano attivo, e SOC-C-2 si verifica esattamente.")

    # I box sono aggiunti in ordine, per ogni k: U0>=0, U0<=vx, |U1|<=vy, |U2|<=w
    etichette = ["vx >= 0", "vx <= vx_max", "|vy| <= vy_max", "|w| <= omega_max"]
    print()
    print("  ripartizione dei vincoli attivi lungo l'orizzonte:")
    for j, e in enumerate(etichette):
        sel = np.arange(len(idx_ineq)) % 4 == j
        n_a = int(act_i[sel].sum())
        n_lo = int((at_lo[idx_ineq] & act_i)[sel].sum())
        n_hi = int((at_hi[idx_ineq] & act_i)[sel].sum())
        print(f"    {e:20} {n_a:3d} / {N}   (al minimo {n_lo}, al massimo {n_hi})")

    # ── LICQ ────────────────────────────────────────────────────────────
    print()
    print("=" * 74)
    print("LICQ  (Def. 6.1.5)")
    print("=" * 74)
    idx_att = list(np.nonzero(is_eq | active_all)[0])
    J = ca.Function("J", [opti.x, opti.p], [ca.jacobian(opti.g, opti.x)])
    Jv = np.array(J(opti.debug.value(opti.x), opti.debug.value(opti.p)))
    A = Jv[idx_att, :]
    rank = np.linalg.matrix_rank(A)
    print(f"vincoli attivi (uguaglianze + disuguaglianze attive): {len(idx_att)}")
    print(f"rango del Jacobiano attivo: {rank}")
    if rank == len(idx_att):
        print("  -> LICQ VERIFICATA: i gradienti attivi sono indipendenti,")
        print("     quindi i moltiplicatori KKT esistono e sono UNICI.")
    else:
        print(f"  -> LICQ VIOLATA: {len(idx_att) - rank} dipendenze lineari.")

    # ── Stazionarieta' della lagrangiana ────────────────────────────────
    print()
    print("=" * 74)
    print("STAZIONARIETA'  (§6.1.2, eq. 6.8a)")
    print("=" * 74)
    lam_s = ca.MX.sym("lam", opti.g.shape[0])
    L = opti.f + ca.dot(lam_s, opti.g)
    gL = ca.Function("gL", [opti.x, opti.p, lam_s], [ca.gradient(L, opti.x)])
    r = np.array(gL(opti.debug.value(opti.x), opti.debug.value(opti.p), lam)).ravel()
    print(f"|| grad_x L(x*, lambda*) ||_inf = {np.abs(r).max():.3e}")
    print(f"|| grad_x f(x*) ||_inf          = "
          f"{np.abs(np.array(ca.Function('gf',[opti.x,opti.p],[ca.gradient(opti.f,opti.x)])(opti.debug.value(opti.x), opti.debug.value(opti.p))).ravel()).max():.3e}")
    print("  (il residuo va confrontato con la tolleranza di IPOPT, non con zero:")
    print("   un metodo di punto interno si ferma sulla central path)")

    # ── Moltiplicatori ──────────────────────────────────────────────────
    print()
    print("=" * 74)
    print("MOLTIPLICATORI")
    print("=" * 74)
    lam_eq = lam[is_eq]
    print(f"uguaglianze  : max|lambda| = {np.abs(lam_eq).max():.3e}   "
          f"mediana = {np.median(np.abs(lam_eq)):.3e}")
    if strong.any():
        print(f"disuguaglianze: max|mu| = {np.abs(mu_i[strong]).max():.3e}   "
              f"mediana = {np.median(np.abs(mu_i[strong])):.3e}")
        print()
        print("  Questi mu sono il dato che serve alla penalita' esatta l1")
        print("  (Thm 6.3.1): rho > max|mu*| rende lo slack esattamente nullo.")
        print(f"  soglia suggerita: rho > {np.abs(mu_i[strong]).max():.3e}")
    else:
        print("disuguaglianze: nessun vincolo fortemente attivo")

    # ── SOC-C-2: Hessiana proiettata sul cono critico ───────────────────
    print()
    print("=" * 74)
    print("SOC-C-2  (Thm 6.1.6) — Hessiana proiettata sul cono critico")
    print("=" * 74)
    H = ca.Function("H", [opti.x, opti.p, lam_s], [ca.hessian(L, opti.x)[0]])
    Hv = np.array(H(opti.debug.value(opti.x), opti.debug.value(opti.p), lam))
    Hv = 0.5 * (Hv + Hv.T)
    # Base del nucleo di A: le direzioni critiche (con complementarita' stretta
    # il cono coincide con ker A).
    _, s_val, Vt = np.linalg.svd(A)
    tol = max(A.shape) * (s_val.max() if s_val.size else 0.0) * np.finfo(float).eps
    ns = Vt[np.sum(s_val > tol):].T          # (n, n - rank)
    print(f"dimensione del cono critico: {ns.shape[1]}")
    if ns.shape[1] == 0:
        print("  cono banale: la soluzione e' determinata dai soli vincoli attivi")
    else:
        Hp = ns.T @ Hv @ ns
        ev = np.linalg.eigvalsh(0.5 * (Hp + Hp.T))
        print(f"autovalore minimo: {ev.min():.6e}")
        print(f"autovalore massimo: {ev.max():.6e}")
        if ev.min() > 0:
            print("  -> SOC-C-2 SODDISFATTA: x* e' un minimo locale STRETTO.")
            print("     E' il massimo certificabile: il problema non e' convesso,")
            print("     quindi l'ottimalita' globale non e' dimostrabile (§4.3.3).")
        else:
            print("  -> SOC-C-2 NON soddisfatta: direzione a curvatura non positiva.")
        if ev.min() > 0:
            cond = ev.max() / ev.min()
            c_rate = (ev.max() - ev.min()) / (ev.max() + ev.min())
            print(f"numero di condizionamento sul cono: {cond:.3e}")
            print(f"  costante di contrazione c = (l_max-l_min)/(l_max+l_min) "
                  f"= {c_rate:.6f}")
            print("  (§4.4.3: piu' c e' vicino a 1, piu' lenta la convergenza lineare)")
        else:
            print("  condizionamento non definito: l'Hessiana proiettata non e'")
            print("  definita positiva, quindi non e' un operatore invertibile sul cono.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
