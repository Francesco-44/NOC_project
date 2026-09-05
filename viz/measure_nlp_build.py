#!/usr/bin/env python3
"""
Costo di costruzione dell'NLP: istruzioni scalari e primo solve a freddo.

Perche' e' separato da make_results.py: e' l'unica misura del report che
dipende dalla MACCHINA e non dai dati. results.json e' riproducibile, questi
due tempi no, quindi non entrano nel file dei risultati; il numero di
istruzioni invece e' deterministico e serve solo a documentare l'effetto di
`expand: true`.

I valori vanno riportati a mano in viz/results_tex.py (sec_report_macros),
dove alimentano \\resNops, \\resColdDep e \\resColdMax.

Uso:
    python3 viz/measure_nlp_build.py
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import casadi as ca  # noqa: E402

import common  # noqa: E402

# N deployato e N piu' lungo della campagna orizzonte (horizon_sweep_ext.json).
N_DEPLOY, N_MAX = 15, 40


def misura(N: int) -> tuple[float, int, int, int]:
    cfg, _ = common.load_profile()
    cfg = dataclasses.replace(cfg, N=N)
    sc = common.narrow_gap()
    t0 = time.perf_counter()
    tr = common.make_tracker(cfg)        # grafo MX -> SX, sparsita', derivate
    common.solve_at(tr, sc.pose, sc)     # primo solve: paga la costruzione
    cold = time.perf_counter() - t0
    o = tr._opti
    fg = ca.Function("nlp", [o.x, o.p], [o.f, o.g]).expand()
    return cold, int(fg.n_instructions()), int(o.x.shape[0]), int(o.g.shape[0])


def main() -> int:
    for N in (N_DEPLOY, N_MAX):
        cold, nops, n, m = misura(N)
        etichetta = "deployata" if N == N_DEPLOY else "orizzonte massimo"
        print(f"N={N:3d} ({etichetta}): primo solve a freddo {cold:.3f} s, "
              f"{nops} istruzioni scalari, n_var={n}, n_g={m}")
    print("\nriportare in viz/results_tex.py, sec_report_macros:")
    print("  resNops = istruzioni a N=%d, resColdDep = tempo a N=%d, "
          "resColdMax = tempo a N=%d" % (N_DEPLOY, N_DEPLOY, N_MAX))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
