# Visualizzare il problema di ottimizzazione

Due strumenti in [`viz/`](../viz/) per guardare *come evolve* il problema che l'MPC
risolve mentre il G1 naviga. Rispondono a due domande diverse e vanno tenuti
distinti, perché vivono in spazi diversi.

| | pannello 1 — [`cost_field.py`](../viz/cost_field.py) | pannello 2 — [`decision_plane.py`](../viz/decision_plane.py) |
|---|---|---|
| spazio | piano del mondo (x, y) | spazio delle decisioni, R¹⁴¹ |
| domanda | dove sono i minimi *di navigazione*? | cosa sta facendo il solutore? |
| quota z | `c(x, y)`, costo di **stare** in un punto | `T₁(x)`, funzione di merito |
| il "pallino" | posizione reale del robot | iterati di IPOPT, x⁰ → x\* |
| copre del corso | §4.3 convessità, §4.2.6 MIP | §4.4.2–4.4.4, §6.1, §6.3.3 |

---

## 0. Il problema di partenza: 141 dimensioni

Con il profilo G1 (N=15) l'NLP ha **141 variabili decisionali** (X: 6×16 = 96,
U: 3×15 = 45). La funzione da disegnare è `J: R¹⁴¹ → R`: non è disegnabile, va
proiettata. Le proiezioni sensate sono quattro, e ognuna risponde a una domanda
diversa:

| | superficie | ci si può disegnare il percorso di IPOPT? |
|---|---|---|
| **A** | sezione in coordinate fisiche, es. (velocità, ampiezza dello scarto) | **no** — gli iterati non stanno su quella superficie |
| **B** | piano affine per due minimi locali + un terzo punto | **sì, esattamente** |
| **C** | piano degli autovettori dell'Hessiana | **sì, esattamente** |
| **D** | `c(x, y)` sul piano del mondo | no — è un altro spazio |

Il percorso x⁰ → x\* **non è una quinta opzione**: A/B/C/D scelgono *quale
superficie disegnare*, il percorso è *cosa disegnarci sopra*. La proiezione degli
iterati è esatta solo su un sottospazio affine dello spazio delle decisioni,
quindi solo su **B** e **C**.

Implementati: **D** (pannello 1) e **B** (pannello 2).

---

## 1. Pannello 1 — il paesaggio di navigazione `c(x, y)`

```bash
python3 viz/cost_field.py                                   # scenario u_trap
python3 viz/cost_field.py --scenario corridor --animate     # + GIF dell'anello chiuso
python3 viz/cost_field.py --reference goal                  # paesaggio in stile campo di potenziale
python3 viz/cost_field.py --set mpc_W_obs_sigmoid=600       # studio parametrico
```

Disegna, sul piano del mondo, il costo di **stare** in un punto:

```
c(p) = costo di inseguimento del riferimento  +  barriera degli ostacoli
```

con la quota `z` che è il costo, come le figure del corso. Sopra ci vanno il
pallino della posizione reale, la traiettoria predetta dall'MPC sull'orizzonte,
e il confine dell'insieme raggiungibile.

### Che cosa è e che cosa NON è

$$J(U) \;=\; \sum_{k=0}^{N} c(p_k) \;+\; \text{termini sull'ingresso}$$

`c` è il costo di **stare** in un punto; `J` è il costo di **una traiettoria**,
cioè la somma di `c` lungo di essa. **L'MPC minimizza J, non c.**

Conseguenza visibile, ed è il motivo per cui la figura è interessante: il
pallino **non segue la massima pendenza**, e può *salire* localmente se questo
abbassa la somma sull'orizzonte. È la differenza fra MPC e campo di potenziale
artificiale resa guardabile — e non è un dettaglio accademico: è il motivo per
cui l'MPC esce da una trappola a U dove un APF si incastra.

### Il termine di manovra non si somma

Verrebbe naturale aggiungere a `c` un terzo termine per il costo della manovra
necessaria ad arrivare in `p`. **È sbagliato**: cresce con il quadrato della
distanza, domina gli altri due e sposta il minimo globale addosso al robot. Il
campo direbbe "la cosa più economica è non muoversi", che è vero e inutile.

Viene invece usato per quello che significa davvero: definisce l'insieme
**raggiungibile**, cioè U_Σ, disegnato come regione. Sul G1 il risultato è molto
istruttivo, perché il robot non può indietreggiare né traslare:

```
tempo minimo per raggiungere un punto, a parità di distanza:
   a 0.3 m: DAVANTI 1.33 s   DIETRO 11.05 s  ->  8.3x   (dietro NON raggiungibile)
   a 0.6 m: DAVANTI 2.28 s   DIETRO 12.09 s  ->  5.3x   (dietro NON raggiungibile)
   a 0.9 m: DAVANTI 3.26 s   DIETRO 13.11 s  ->  4.0x   (dietro NON raggiungibile)
```

Il vincolo `vx ≥ 0` smette di essere un numero in uno YAML e diventa una regione
che si vede.

### Due modalità per il riferimento

- `--reference path` (default) — errore rispetto al **percorso A\* vero**,
  calcolato con il pianificatore del repo. È la restrizione alla posizione del
  termine `‖x − x_ref‖²_Q` dell'MPC: fedele. L'intero percorso è una valle a
  costo ≈ 0, quindi i minimi che ci stanno sopra sono attesi, non trappole.
- `--reference goal` — attrazione verso il goal. Non è ciò che l'MPC sente (il
  suo goal-seeking è delegato ad A\*), ma è il paesaggio in stile campo di
  potenziale, ed è la modalità in cui le trappole si vedono come minimi veri.

Senza il vero A\* la figura sarebbe un uomo di paglia: l'MPC da solo, con un
riferimento che attraversa gli ostacoli, non ha alcuna possibilità di evitarli.

---

## 2. Pannello 2 — lo spazio delle decisioni, con il percorso di IPOPT

```bash
python3 viz/decision_plane.py                                       # centred_pillar
python3 viz/decision_plane.py --set mpc_W_obs_sigmoid=600
python3 viz/decision_plane.py --objective                           # solo f, senza penalità
```

### Come si sceglie il piano

Una sezione 2-D casuale di R¹⁴¹ quasi certamente non contiene niente di
interessante. Qui il piano è costruito **perché** contenga la struttura:

1. si risolve l'NLP due volte, con warm start sbilanciato a sinistra e a destra,
   ottenendo due minimi locali `x*_L` e `x*_R` (se esistono);
2. il terzo punto di ancoraggio è l'iterato iniziale;
3. il piano affine per quei tre punti **contiene entrambi i minimi per
   costruzione**, quindi la biforcazione è nell'inquadratura invece che sperare
   di beccarla.

Il warm start si inietta usando il meccanismo che l'MPC ha già
(`_prev_u` / `_prev_x`): non serve modificare il tracker per questo.

### Perché si disegna la funzione di merito e non `f`

In multiple shooting X e U sono variabili **indipendenti**, legate dai vincoli di
dinamica. Un punto generico del piano quindi **non è ammissibile**, e disegnare
il solo `f` darebbe un paesaggio il cui minimo può cadere fuori dall'insieme
ammissibile. Si disegna invece la funzione di merito ℓ¹ della §6.3.3:

$$T_1(x) \;=\; f(x) \;+\; \sigma \,\lVert \text{violazione dei vincoli} \rVert_1$$

### σ non si sceglie a occhio

Il **Teorema 6.3.1** dice che la penalità ℓ¹ è **esatta** — il minimo della
funzione di merito coincide con quello del problema vincolato — non appena σ
supera il modulo del moltiplicatore di Lagrange. I moltiplicatori li restituisce
IPOPT (`opti.value(opti.lam_g)`), quindi **σ è una quantità letta dal problema,
non un parametro da tarare**:

```
max|lambda| dai moltiplicatori di IPOPT: 8.268e+04   ->   sigma = 1.24e+05 (esatta)
```

Lo si vede: con un σ euristico troppo piccolo (il primo tentativo usava
`percentile(f)/percentile(viol)` ≈ 2.2e3) il minimo della superficie cade **fra**
le due soluzioni invece che su di esse. È l'inesattezza contro cui il teorema
mette in guardia, e la figura la mostra. Con `--sigma` si può forzare un valore
e riprodurre l'errore di proposito.

### Registrazione degli iterati

Richiede una modifica minima a
[`mpc_tracker.py`](../src/a_star_mpc_planner/a_star_mpc_planner/mpc_tracker.py):
il flag `record_iterates` in `MPCConfig` registra `opti.debug.value(opti.x)` a
ogni iterazione tramite `opti.callback`. **Spento in esercizio**: costa una copia
del vettore delle variabili per iterazione e non serve al controllo.

---

## 3. Fedeltà: la garanzia che le figure non mentano

Una visualizzazione che disegna una funzione diversa da quella ottimizzata non
serve a niente. Quindi:

- il pannello 2 estrae `f` e `g` **direttamente dall'espressione CasADi** che
  IPOPT minimizza (`ca.Function('f', [opti.x, opti.p], [opti.f])`): non c'è
  niente da riscrivere e niente che possa divergere;
- il pannello 1 usa una replica numpy del termine di ostacolo, verificata da
  [`viz/test_fidelity.py`](../viz/test_fidelity.py), che valuta l'intero costo
  nei due modi e li confronta:

```bash
python3 viz/test_fidelity.py
```
```
costo da CasADi (opti.f) : 51182.9231139686
costo replicato in numpy : 51182.9231139686
errore assoluto 0.000e+00   relativo 0.000e+00
ESITO: FEDELE
```

Errore **esattamente zero**. Va rilanciato ogni volta che si tocca il costo
nell'NLP.

---

## 4. Cosa hanno già detto le due figure

Tre risultati, ottenuti con i parametri deployati del profilo G1.

### 4.1 L'orizzonte copre meno di un metro

Con N=15, dt=0.20 e v_ref=0.2 l'orizzonte copre **0.6 m** di percorso (0.9 m a
`vx_max`). L'insieme raggiungibile misurato è di **0.86 m**. Un ostacolo oltre
quella distanza è semplicemente fuori dall'orizzonte: la barriera non lo vede, e
il paesaggio non biforca per un motivo che **non ha niente a che fare con il peso
della barriera**. È il primo effetto da escludere prima di tarare qualsiasi cosa.

### 4.2 Con i parametri deployati il paesaggio NON biforca

Con un pilastro centrato dentro l'orizzonte, i due warm start opposti convergono
**allo stesso identico punto** in R¹⁴¹:

| `mpc_W_obs_sigmoid` | distanza fra le due soluzioni | esito |
|---|---|---|
| **120 (deployato)** | 0.0000 | un solo minimo: l'MPC tira dritto |
| 300 | 3.7424 | due minimi distinti |
| 600 | 4.0713 | due minimi distinti |
| 1200 | 4.5574 | due minimi distinti |

La biforcazione compare **fra 120 e 300**. Con `W_obs = 120` contro `Q_xy = 200`,
deviare costa più che tirare dritto: tutta l'evasione geometrica è delegata ad
A\* sulla griglia inflazionata, e il contributo dell'MPC all'obstacle avoidance è
prossimo a zero.

Lo stesso risultato compare in modo indipendente nella sezione (velocità,
ampiezza dello scarto): con `W_obs = 120` un solo minimo a scarto nullo, con 600
due minimi simmetrici separati da una cresta.

**Non è una raccomandazione di alzare `W_obs` a 600.** È la constatazione che il
valore attuale rende l'MPC quasi inerte sugli ostacoli, e che la decisione va
presa guardando anche il comportamento in simulazione — non solo il paesaggio.

### 4.3 Le biforcazioni si vedono anche a livello di A\*

Nello scenario `u_trap` con ripianificazione ogni 5 cicli, A\* **cambia lato una
volta** durante la missione: il riferimento salta da una classe di omotopia
all'altra mentre il robot avanza. È il fenomeno del Teorema 4.4.6 (regolarità di
x\*(ϑ)) al livello del pianificatore globale, e `cost_field.py` lo conta e
disegna sia il primo sia l'ultimo riferimento.

---

## 4-bis. Sui run VERI del G1 nel magazzino

Gli scenari sintetici servono a isolare un fenomeno; per guardare il problema
come si presenta davvero, entrambi i pannelli leggono una rosbag.

### Registrare

```bash
# terminale 1
ros2 launch g1_sim g1_a_star_mpc.launch.py
# terminale 2
./viz/record_run.sh corridoio_stretto
# poi si mandano i goal da RViz con "2D Goal Pose"; Ctrl-C per chiudere la bag
```

`record_run.sh` registra i topic necessari e **non** la griglia di occupazione,
che è grossa e che i pannelli ricalcolano comunque dal campo di costo.

### Guardare

```bash
python3 viz/bag_source.py    viz/bags/corridoio_stretto     # riepilogo
python3 viz/cost_field.py    --bag viz/bags/corridoio_stretto
python3 viz/decision_plane.py --bag viz/bags/corridoio_stretto --frame 120
```

Senza `--frame` si prende automaticamente il ciclo a **costo massimo**, che è
quello in cui il robot stava lavorando di più — di solito il più interessante.

### Cosa cambia rispetto agli scenari sintetici

| ingrediente | sintetico | da bag |
|---|---|---|
| posa | integrata offline | `/robot_pose`, la traiettoria **realmente percorsa** |
| ostacoli | array fisso | `/lidar/points_filtered`, già in `odom` |
| riferimento | A\* ricalcolato | `/a_star/path`, quello vero |
| traiettoria predetta | dal solve offline | `/mpc/predicted_path` |

Nel pannello 1 con `--bag` **non c'è nessuna simulazione**: la traiettoria
disegnata sulla superficie è quella che il robot ha percorso.

### Perché in replay e non in diretta

Il pannello 2 deve **ri-risolvere** l'NLP per ottenere gli iterati di IPOPT.
Farlo in diretta ruberebbe CPU al solutore che si sta misurando, falsando proprio
la grandezza di interesse. In replay il costo di calcolo non disturba nulla, e lo
stesso run si può rianalizzare quante volte si vuole cambiando i parametri con
`--set`.

### La ricostruzione esatta di x₀

Perché il ri-solve sia lo *stesso* problema e non uno simile, serve lo stato
iniziale esatto. Posizione e yaw si dedurrebbero da `/mpc/predicted_path[0]`, ma
le **velocità** sono stimate dentro `mpc_node` (media esponenziale sulle
differenze di posa) e non uscivano da nessuna parte.

`/mpc/diagnostics` è stata quindi estesa, restando compatibile all'indietro:

| indice | contenuto |
|---|---|
| 0–6 | come prima: successo, costo, solve_ms, media, fallimenti, security, vx adattivo |
| **7–12** | **x₀ = [px, py, yaw, vx, vy, ω]** passato al solutore |
| **13** | **iterazioni di IPOPT** dell'ultimo solve |

Le iterazioni non venivano né lette né esposte pur essendo gratis in
`sol.stats()`: sono la grandezza che dice quanto è condizionato il problema, e
sono il punto #2 del Blocco A della [roadmap](roadmap_teorica_noc.md).

Una bag registrata prima di questa modifica ha 7 campi invece di 14 e
`bag_source.py` lo dice esplicitamente invece di ricostruire uno stato inventato.

## 5. Scenari disponibili

| nome | cosa mette alla prova |
|---|---|
| `u_trap` | ostacolo concavo aperto verso il robot: la trappola classica dei campi di potenziale |
| `centred_pillar` | pilastro sulla retta verso il goal, **dentro l'orizzonte**: è il caso in cui il costo dovrebbe biforcare |
| `narrow_gap` | varco stretto: controprova, il minimo deve stare in mezzo |
| `corridor` | corridoio con pilastro sfalsato: caso realistico da magazzino |

Si aggiungono in [`viz/common.py`](../viz/common.py) decorando una funzione con
`@_reg`.

---

## 6. Limiti dichiarati

- **Il pannello 1 non è una visualizzazione di ottimizzazione.** Non può mostrare
  gli iterati, né il condizionamento, né la biforcazione nello spazio delle
  decisioni, perché quelle cose non vivono nel piano (x, y). Serve il pannello 2.
- **Il pannello 2 mostra un piano, non lo spazio.** Fuori dal piano il paesaggio
  può essere qualunque cosa. Il piano è scelto per contenere i due minimi, non
  per essere rappresentativo.
- **Manca l'opzione C** (piano degli autovettori dell'Hessiana), che è quella che
  renderebbe visibile il condizionamento e il legame con il tasso di convergenza
  della §4.4.3. È l'estensione naturale successiva.
- **Anello chiuso simulato, non ROS.** `closed_loop` riproduce l'impianto
  cinematico di `mujoco_sim` e il controllore proporzionale del nodo, ma non è un
  run reale: niente ritardi, niente rumore del LiDAR, niente TF. Per i numeri
  definitivi serve il replay da rosbag.

---

## 7. Nota di ambiente

Su questa macchina convivono **due matplotlib**: 3.10.7 in `~/.local` e 3.5.1 di
sistema. `mpl_toolkits` di sistema è un pacchetto *regolare*, e per le regole di
import di Python un pacchetto regolare trovato più avanti nel `sys.path` batte
una porzione di namespace trovata prima: `mpl_toolkits.mplot3d` viene quindi
risolto dalla 3.5.1 e fallisce contro l'API della 3.10 (`cannot import name
'docstring'`), lasciando la proiezione `'3d'` non registrata.

`common.ensure_mpl3d()` aggira il problema forzando la risoluzione accanto alla
matplotlib attiva e registrando esplicitamente la proiezione. **La correzione
vera è ripulire l'ambiente** (`pip uninstall` di una delle due, o un venv), ma
gli strumenti non ne dipendono.
