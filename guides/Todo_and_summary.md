# Riepilogo cose ancora da implementare e stato attuale progetto di NOC Optimal trajectory

> **⚠ Documento storico — si riferisce al VECCHIO stack.** È stato scritto quando il progetto
> voleva restare agnostico rispetto al robot e copriva sia il Go2 (quadrupede, Gazebo + CHAMP)
> sia il drone, e ne descrive architetture e pacchetti che **non esistono più nel repository**.
> La trattazione attuale è solo sull'Unitree G1 con impianto MuJoCo: per lo stato aggiornato
> si guardino [`roadmap_teorica_noc.md`](roadmap_teorica_noc.md) (teoria e voci ancora aperte)
> e [`Recap.md`](Recap.md) (runbook end-to-end). Questo file è conservato come traccia di come
> si è arrivati alle scelte attuali, non come riferimento operativo.

Analisi degli argomenti teorici del corso e scelta di quali parti approfondire nel report e nella presentazione,
considerazioni riguardo all'unificazione architetturale dei due stack di drone robot Unitree (Go2 e/o G1), 
recap di alcuni test e metriche già estratti allo stato attuale del progetto.

---

## Parte 1 — Mappatura degli argomenti del corso sul progetto

Punto di partenza: le dispense di *Numerical Optimization for Control* (062047,
M.Sc. Automation and Control Engineering, PoliMi, Lorenzo Fagiano, v1.1) fornite
dall'utente. L'obiettivo era identificare **quali sezioni del corso potessero
giustificare, con teoria e metriche, le scelte implementative già presenti nel
progetto** — trasformando il report da "abbiamo scelto X" a "abbiamo scelto X, qui
la teoria che lo motiva, qui il numero che lo dimostra".

### To do list

1. **Fix the perception stack to avoid the problems of oscillations and inflation map**

2. **Describe the problem formulation, NLP structure and CasADi parameters (multiple-shooting, single-shooting) and extrapolate some metrics about the MPC. Describe the constraints that are used, soft and hard, type of penalty used in the cost function; add considerations on terminal constraint and recursive feasibility.**

3. **Test the planner, extract metrics about path length, time to reach the goal, % of success; solve the issue with non-convexity**

4. **Add considerations about discretization method and derivative computations, with some metrics and comparing different approaches**

5. **Add considerations about single-objective or multi-objective implementation of the problem**


### Metodo di lavoro proposto

Non un elenco esaustivo di citazioni, ma 4-6 blocchi argomentati in profondità,
ciascuno con: ipotesi teorica → scelta implementativa nel codice → metrica
misurata. I blocchi identificati come prioritari:

1. **Struttura dell'NLP** (§7.2.1–7.2.2 del corso): multiple shooting vs single
   shooting condensato. Il progetto aveva già entrambe le formulazioni in punti
   diversi (drone: condensato in `trajectory_opt/`, multiple shooting in
   `mpc_tracker.py`), quindi era un A/B test già scritto da strumentare.

2. **Discretizzazione** (§2.1.3, Euler vs RK2): il doppio integratore del drone
   con tenuta a gradino (ZOH) ammette una discretizzazione **esatta**, non è
   un'approssimazione di Euler — punto di forza da dichiarare esplicitamente. Il
   modello cinematico SE(2) del Go2 no, ed è lì che RK2 (mid-point rule) guadagna
   un ordine di accuratezza al costo di un'addizione.

3. **Regolarità delle funzioni di costo e derivate** (§5.2–5.3, §4.2.5, §4.4.4):
   perché CasADi/AD invece delle differenze finite (costo per gradiente < 3×
   valutazione singola, indipendente da *n*, contro *n+1* o *2n* valutazioni a
   precisione inferiore); e perché la penalità ostacolo `max(0,·)²` è C¹ ma non
   C², mentre B-spline e barriera logistica sono C^∞ — rilevante perché IPOPT è
   un metodo Newton-type che usa l'Hessiana.

4. **Vincoli soft e penalità esatta** (§6.3.3, Teorema 6.3.1): con penalità *L¹*
   e peso ρ superiore al moltiplicatore di Lagrange del vincolo hard
   corrispondente, la violazione residua è **esattamente zero**; con penalità *L²*
   decade come 1/ρ ma non si annulla mai. Il progetto aveva già entrambe le
   versioni in `new_mpc.py` (slack quadratico sugli half-space, slack lineare sul
   vincolo terminale) senza che fosse dichiarato il motivo della scelta.

5. **Tuning multi-obiettivo** (§7.4): sostituire la tabella di pesi tarata a mano
   con la procedura a-posteriori del corso — punto Utopico, curva di Pareto,
   spider chart — invece di giustificazioni aneddotiche nei commenti YAML.

### Approfondimenti di secondo livello

- **Non-convessità e biforcazioni** (§4.3.3, §4.4.5, Teorema 4.4.6): la legge MPC
  come funzione dello stato corrente può biforcare quando due minimi locali si
  scambiano di rango (es. "passa a destra" vs "passa a sinistra" di un ostacolo),
  spiegando il chattering osservato empiricamente.
- **Ingredienti terminali e recursive feasibility** (§7.2.5): il progetto non
  aveva alcun vincolo terminale, solo un costo terminale pesato.
- **LICQ e selezione dei punti LiDAR** (§6.1.1): la selezione per settore
  angolare nel codice esistente preserva empiricamente l'indipendenza lineare dei
  gradienti dei vincoli attivi, senza che fosse mai stato dichiarato il perché.

### Cosa è stato scartato esplicitamente (in quanto non troppo inerente con le cose trattate nel progetto)

System identification (cap. 3.1), Moving Horizon Estimation (§3.2/§7.3), path
following parametrico in θ (§7.2.4) — non pertinenti al flusso attuale del
progetto, da citare al più come sviluppi futuri.

---

## Parte 2 — Implementazione di un possibile stack ibrido per drone + Go2: `trajopt_core`

### Struttura

```
trajopt_core/
├── trajopt_core/
│   ├── mapping/gaussian_grid_map.py     griglia gaussiana condivisa
│   ├── planning/a_star_planner.py       A* a orizzonte mobile condiviso
│   ├── models/
│   │   ├── base.py                      MotionModel — l'astrazione
│   │   ├── double_integrator_z.py       aereo: NX=7, NU=4, grado relativo 2
│   │   └── kinematic_se2.py             Go2: NX=3, NU=3, grado relativo 1
│   ├── mpc/
│   │   ├── ocp.py                       TrajectoryOCP parametrico, build-once
│   │   ├── reference.py                 riferimento + condizionamento del path
│   │   ├── obstacles.py                 4 strategie ostacolo intercambiabili
│   │   ├── lookahead.py                 estrazione del setpoint
│   │   └── config.py                    OCPConfig, SolverOptions
│   ├── bench/                           simulatore di missione + recorder statistiche
│   └── config_io.py                     un solo schema YAML per entrambi i robot
├── config/
│   ├── planner_params.yaml              profilo base (aereo)
│   └── legged_overrides.yaml            delta per il Go2 (~25 righe)
├── examples/
│   ├── cross_platform_demo.py
│   └── bench_formulations.py
├── docs/benchmarks.md                   tabelle generate dai benchmark
└── tests/                               56 test, incluso il golden-equivalence
```

`trajopt_ros/` (pacchetto separato, sottile):

```
trajopt_ros/
├── trajopt_ros/
│   ├── common.py                        conversioni ROS <-> core
│   ├── a_star_node.py                   nodo generico, nomi topic relativi
│   └── mpc_node.py                      nodo generico, nomi topic relativi
├── launch/planner.launch.py             PROFILES: mappatura topic per piattaforma
└── test/test_nodes_smoke.py             4 test end-to-end
```

### L'astrazione `MotionModel`

Interfaccia con quattro metodi obbligatori che corrispondono esattamente a
`(f_Σ, U_Σ, X_Σ, C)`: `step()`, `input_bounds()`, `add_state_constraints()`,
`PLANAR_IDX`. Tutto il resto di `trajopt_core.mpc` è scritto una volta sola e
condiviso.

### Verifica dell'equivalenza (non solo dichiarata)

`tests/generate_golden.py` esegue uno scenario deterministico attraverso le
**tre** implementazioni — `new_mujoco` (repo drone), `a_star_mpc_planner` (repo
Go2, su `/mnt/c/Users/franc/Desktop/Tesi/Go2_navigation`), e `trajopt_core` — e
verifica che producano **esattamente** la stessa griglia e lo stesso path A*
(19 waypoint, bit-identici). Il dedup non è un'affermazione, è un test.

### Quattro strategie di ostacolo intercambiabili

Implementate come classi con la stessa interfaccia, selezionabili da YAML:
`GaussianGridCost` (B-spline, C²), `HalfSpaceQuadratic` (hinge quadratica, C¹),
`SigmoidBarrier` (barriera logistica, C^∞), `SlackedHalfSpace` (vincolo SCA con
slack L¹ o L²). Questo trasforma gli esperimenti del capitolo teorico in switch
di configurazione anziché fork di codice.

### Configurazione: un profilo base + delta

`load_planner(base, overrides)` fa un deep-merge: il profilo Go2 non duplica il
file base, dichiara solo ~25 righe di differenze (modello, limiti di velocità,
peso dello yaw, raggio di sicurezza). Questo rende la domanda "quanto dello
stack è specifico della piattaforma?" verificabile leggendo un file corto invece
di diffare due configurazioni quasi identiche.

---

## Parte 3 — Risultati dei benchmark: quattro ipotesi smentite

Il valore principale per il report: quattro risultati che hanno **contraddetto
le premesse** con cui erano stati progettati gli esperimenti — più forti di una
conferma, perché dimostrano comprensione del meccanismo sottostante.

### 1. "Build-once" non è gratis

Rendere parametrico il costo B-spline della griglia (knot in frame locale,
coefficienti come parametro CasADi) mantiene il grafo NLP fisso ed è
bit-identico alla versione non parametrica — ma costa **~14 secondi/solve**
contro 35-50 ms della versione ricostruita ogni ciclo, a parità di iterazioni.
Con coefficienti simbolici CasADi non può sfruttare la struttura tensoriale
dello spline. Lezione: eliminare la ricostruzione paga solo se la
parametrizzazione lascia intatto il costo di valutazione — vero per i termini
puntuali, falso per lo spline. Default cambiato di conseguenza.

### 2. La convessità batte la regolarità

Ipotesi di partenza: la barriera C^∞ dovrebbe convergere meglio perché IPOPT è
un metodo Newton-type. Risultato opposto: la hinge² (C¹, ma convessa e inattiva
quasi ovunque) converge in 5.2 iterazioni medie contro le 14.4 della barriera
logistica (C^∞, ma non convessa e sempre attiva) — circa 4× più veloce. La
barriera logistica guadagna però in clearance (0.86 m contro 0.72 m): è un vero
trade-off di Pareto costo/sicurezza, non un'opzione dominata.

### 3. Il warm start non aiuta (in questo caso)

Test discriminante su due scenari × due piattaforme (4 casi): la soluzione
precedente traslata di un passo **non è mai** la miglior stima iniziale — è
sistematicamente la peggiore delle tre opzioni testate (traslata, riferimento,
cold start a zero). Il riferimento, calcolato comunque a ogni ciclo, è il
default migliore. Meccanismo plausibile proposto (non dimostrato): IPOPT è un
metodo interior-point le cui iterate devono restare strettamente interne al
dominio ammissibile, mentre l'ottimo precedente giace sul bordo con vincoli
attivi.

### 4. L'architettura conta più della sintonizzazione

Il condizionamento del path A* (ricampionamento + smoothing) aiuta
modestamente (6-17% di distanza residua in meno a budget di calcolo fisso). Ma
la **modalità di chiusura dell'anello** domina completamente: applicando
l'input ottimo direttamente restano 8.9-11.3 m dal goal dopo 150 cicli;
pubblicando il setpoint di lookahead ne restano 1.9-2.9 m. Usare l'ottimizzatore
come generatore di riferimento anziché controllore diretto vale diverse volte
più di qualunque taratura del riferimento stesso.

### Confermati come da teoria

- Ordine di troncamento: Euler 1.999, mid-point (RK2) 3.000 (fit log-log).
- Penalità esatta: slack L¹ zero a ρ≥200, slack L² decade come 1/ρ senza mai
  annullarsi — comportamento a soglia coerente col Teorema 6.3.1.
- Densità Jacobiano 0.4–2%, lineare in N (multiple shooting).

### Due scoperte impreviste durante l'implementazione

- **Vincoli di stato a k=0**: imporli anche sul primo passo (già fissato
  dall'uguaglianza `X[:,0]==x0`) non può cambiare la soluzione ma può rendere
  l'intero problema infeasible quando lo stato misurato esce momentaneamente
  dall'insieme ammissibile — situazione di routine con un anello interno reale
  che ha overshoot. Il fix ha portato il success rate da 0.46 a 1.00.
- **SCA autocontraddittorio**: se la traiettoria di linearizzazione attraversa
  un ostacolo, i passi oltre generano normali opposte e i semispazi risultanti
  sono contraddittori (`p_x ≤ 0.7` e `p_x ≥ 2.3` simultaneamente); la firma
  diagnostica è uno slack che non risponde più al peso della penalità.
  Risolto con `sca_iterations > 1`.

---