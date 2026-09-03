# Risultati — generati automaticamente

> Non modificare a mano: rigenerare con `python3 viz/make_results.py`.

- data: 2026-09-02T22:49:41+00:00
- commit: `9466c8a615` sul branch `main`  **(albero di lavoro sporco: numeri non riproducibili da questo commit)**
- profilo: `src/a_star_mpc_planner/config/planner_params_g1.yaml`
- CasADi 3.7.2, numpy 1.26.4, Python 3.10.12

Parametri: N=15, dt=0.35, v_ref=0.27, vx_max=0.4, W_obs=120.0, integrator=midpoint, path_mode=time, terminal=none


## Classe 1 — proprietà della formulazione

*Indipendenti dalla run: si calcolano una volta sola.*


### Ordine di troncamento (§2.1.3)

| regime | ordine Euler | ordine punto medio |
|---|---|---|
| nominale (vx=0.2, w=0.3) | 1.00 | 2.00 |
| con deriva laterale | 1.00 | 2.00 |
| rotazione rapida (w=1.0) | 1.00 | 2.00 |

Al dt deployato (0.35) su 3 s: Euler 2.855e-02 m, punto medio 2.498e-04 m — guadagno 114×.


### Derivate: AD contro differenze finite (§5.2–5.3)

| metodo | valutazioni di f | accuratezza |
|---|---|---|
| differenze in avanti | 142 | 1.3e-07 |
| differenze centrate | 282 | 9.8e-11 |
| **AD inverso** | **1.5** (intervallo 1.1–2.1) | precisione macchina |

*Il costo dell'AD è un micro-benchmark su tempi di ~100 μs: si riporta la mediana di più misure con il suo intervallo, perché una singola coppia oscilla sensibilmente. Quello che conta, ed è stabile, è che stia fra 1 e 3 come prevede il §5.3 — non la sua seconda cifra.*

Passi ottimi misurati: avanti 1.49e-08 (teorico √eps = 1.49e-08), centrate 6.06e-06 (teorico eps^(1/3) = 6.06e-06).
Le differenze centrate userebbero il 30% del budget di ciclo (125 ms).


### Hessiana esatta contro L-BFGS (§4.4.4)

| Hessiana | iterazioni | J* |
|---|---|---|
| exact | 35 | 3645.974 |
| limited-memory | 89 | 3645.974 |

### Penalità esatta ℓ¹ (Thm 6.3.1)

d_safe = 1.1, max|μ\*| = 2.677e+04

| ρ | slack ℓ¹ | slack ℓ² |
|---|---|---|
| 1e+03 | 5.602e-01 | 6.529e-01 |
| 1e+04 | 1.438e-01 | 3.189e-01 |
| 1e+05 | 0 | 9.539e-02 |
| 1e+06 | 0 | 1.326e-02 |
| 1e+07 | 0 | 1.337e-03 |
| 1e+08 | 0 | 1.338e-04 |

ℓ¹ nullo da ρ = 1e+05; pendenza ℓ² sulla coda = -1.00 (attesa −1).


### Struttura dell'NLP

| N | variabili | vincoli | densità jac | densità hess |
|---|---|---|---|---|
| 10 | 96 | 106 | 2.52% | 6.51% |
| 15 | 141 | 156 | 1.73% | 4.45% |
| 25 | 231 | 256 | 1.07% | 2.73% |
| 50 | 456 | 506 | 0.54% | 1.39% |


## Classe 2 — proprietà dell'istanza

*Variano ciclo per ciclo: il dato è il profilo, non un numero singolo.*


### KKT lungo la missione (§6.1)

LICQ sempre verificata: **True** · complementarità stretta sempre: **True** · SOC-C-2 sempre soddisfatta: **True**

Dimensione del cono critico fra **38** e **44** a seconda del punto di lavoro (vale l'identità `dim(cono) = n_var − vincoli attivi`: è il complemento della saturazione, non una tendenza temporale).

| ciclo | t [s] | attivi | rango | LICQ | cono | λ_min proiettato |
|---|---|---|---|---|---|---|
| 0 | 0 | 103 | 103 | sì | 38 | +1.33e+00 |
| 86 | 46 | 101 | 101 | sì | 40 | +1.26e+00 |
| 173 | 85 | 98 | 98 | sì | 43 | +1.25e+00 |
| 260 | 120 | 101 | 101 | sì | 40 | +1.31e+00 |
| 347 | 153 | 97 | 97 | sì | 44 | +1.24e+00 |
| 433 | 191 | 100 | 100 | sì | 41 | +1.26e+00 |
| 520 | 245 | 100 | 100 | sì | 41 | +1.25e+00 |
| 607 | 315 | 101 | 101 | sì | 40 | +1.28e+00 |
| 694 | 358 | 98 | 98 | sì | 43 | +1.23e+00 |

### Biforcazione (§4.4.5, Thm 4.4.6)

Soglia fra W_obs = 60 e 120; il deployato è 120 (sopra soglia).
Sul ciclo reale 436: biforca mai = **False**.



## Classe 3 — prestazione in anello chiuso

*Dipendono da run e mondo: qui servono più missioni.*


### Errore di predizione (§7.2.5) — bag `industrial_v6`, 695 cicli

Offset a k=0: 0.0561 m (allineamento temporale, non modello).
**Divergenza a fine orizzonte: 0.797 m**, cioè 38× l'errore di Euler e 2180× quello del punto medio, sullo stesso orizzonte.


### Path following in θ (§7.2.4)

| grandezza | riferimento a tempo | ascissa θ |
|---|---|---|
| vx media [m/s] | 0.2690 | 0.3786 |
| spostamento [m] | 1.3510 | 2.0376 |
| iterazioni | 9.3 | 25.2 |

**+51% di avanzamento**; v_ref lasciava inutilizzato il 32% della velocità.


### Vincolo terminale (§7.2.5)

Slack massimo 0.000e+00 — sempre ammissibile: **True**. Costo del vincolo da +1.2% a +40.4%.

