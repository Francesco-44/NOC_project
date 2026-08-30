# Roadmap teorica — copertura del corso *Numerical Optimization for Control* (062047)

Guida di riferimento per portare questo repository dentro il perimetro teorico del corso
062047 (L. Fagiano, PoliMi). Ogni voce è una scheda autocontenuta:

- **Corso** — la sezione esatta delle dispense
- **Stato** — cosa c'è oggi nel repo, con il file
- **Da aggiungere** — l'intervento concreto
- **Verifica** — il numero o la figura che ne dimostra l'effetto
- **Sforzo** — S (ore), M (un giorno), L (più giorni)

Convenzione dei riferimenti: nelle righe **Corso** i `§` rinviano alle *dispense*; altrove
rinviano alle sezioni di questa guida.

## Perimetro: quale problema stiamo risolvendo

Le dispense trattano tre famiglie di problemi di ottimizzazione applicati ai sistemi dinamici
(§7.3 e §3): **identificazione**, **stima dello stato** e **controllo ottimo**. Questo progetto
appartiene alla terza e solo alla terza: risolviamo online un **FHOCP non lineare**
(§3.3 eq. (3.11)) a orizzonte recedente.

Di conseguenza sono **fuori perimetro per scelta**, non per dimenticanza:

- l'identificazione dei parametri del modello con metodo dell'errore di simulazione
  (§3.1.2, §7.3.1) — è *optimal system identification*;
- la stima dello stato con Moving Horizon Estimation (§3.2, §7.3.2) — è *optimal observer*.

Entrambe sarebbero implementabili qui, ed entrambe comparivano nella prima versione di questa
guida. Sono state rimosse perché diluirebbero l'elaborato su tre categorie di problema invece
di approfondirne una. La §8 elenca cosa resta fuori e perché.

Il baricentro è quindi il **Capitolo 7** (soluzione dei problemi di controllo ottimo), che qui
viene per primo, sostenuto dal **Capitolo 6** (ottimizzazione vincolata: è il macchinario con
cui l'FHOCP si risolve), dal **Capitolo 2** (discretizzazione: è come l'FHOCP nasce) e dal
**Capitolo 5** (derivate: è cosa lo rende calcolabile).

---

## 0. Stato di partenza, misurato

NLP effettivamente costruito da [`mpc_tracker.py`](../src/a_star_mpc_planner/a_star_mpc_planner/mpc_tracker.py)
con i parametri deployati sul G1 in
[`planner_params_g1.yaml`](../src/a_star_mpc_planner/config/planner_params_g1.yaml) (N = 15, dt = 0.2 s):

| grandezza | valore |
|---|---|
| variabili decisionali | **141** (X: 6×16 = 96, U: 3×15 = 45) |
| vincoli totali | **156** |
| — di uguaglianza | 96 (6×15 dinamica + 6 condizione iniziale) |
| — di disuguaglianza | 60, **esclusivamente box sugli ingressi** (4 righe per passo) |
| parametri CasADi | 121 |
| densità Jacobiano dei vincoli | 1.60 % (351 nnz) |
| densità Hessiana lagrangiana | 4.30 % (855 nnz) |
| termini ostacolo | 128 sigmoid + 128 hinge², **tutti nel costo, nessuno nei vincoli** |
| orizzonte | 3.0 s ⇒ **0.60 m** di percorso a `v_ref` = 0.2 m/s |

Riproducibile con `guides/snippets/nlp_structure.py` (vedi *Riproducibilità*, §7); lo script accetta
`--profile` e un elenco di orizzonti, quindi la stessa tabella si rifà a qualunque N.

### Quattro fatti misurati che vincolano tutta la guida

**1. Gli ostacoli non sono vincoli, sono penalità.** Non esistono moltiplicatori di ostacolo,
non c'è un active set non banale, e tutto il Capitolo 6 resta inutilizzato. È il buco singolo
più grande.

**2. Il ritardo di attuazione del primo ordine è degenere sul G1.** Il profilo deployato ha
`mpc_tau_v = mpc_tau_w = 0.001` s contro `dt = 0.2` s, quindi

```
lag = 1 − exp(−dt/τ) = 1 − exp(−200) = 1.000000000000   (esattamente 1 in float64)
```

cioè `v_{k+1} = u_k`. Le tre componenti di velocità nello stato **non sono stati dinamici**:
sono copie ritardate di un passo dell'ingresso. Il modello "a 6 stati con lag del primo ordine"
è, come deployato, un **uniciclo a 3 stati con un passo di ritardo sull'ingresso**. Questo
tocca le §3.1, §3.2 e §5.4, e va detto invece che rivendicare un modello che non è quello in
esecuzione.

La degenerazione non è un'inferenza: **si vede nella sparsità**. A parità di N = 50, lo stesso
problema con un τ non degenere (τ = 0.12 s, cioè τ confrontabile con dt) dà 1556 nnz nello
Jacobiano dei vincoli; il profilo G1 ne dà 1156. I 400 nnz mancanti sono esattamente 8 per
passo: CasADi elimina simbolicamente la dipendenza di `v_{k+1}` da `v_k`, perché con `lag = 1`
quel coefficiente è nullo. La struttura del problema registra la degenerazione da sé.

**3. Il paesaggio di costo non biforca ai pesi deployati.** Misurato su due missioni reali del
G1 nel magazzino (`viz/bags/industrial_plant`, `industrial_plant_fix`): partendo da warm start
opposti, sinistra e destra convergono alla **stessa** soluzione, distanza in ℝ¹⁴¹ pari a
0.0000. Con `W_obs_sigmoid = 120` contro `Q_x = Q_y = 200`, l'evitamento ostacoli è di fatto
**delegato ad A\***. La biforcazione compare tra W_obs 300 e 600.

**4. La fattibilità ricorsiva è già stata persa in esercizio, con conseguenze misurate.**
Vedi §1.2: un fallback pensato come transitorio si è rivelato permanente e ha disattivato
l'MPC per i due terzi di una missione. È il materiale sperimentale della §7.2.5 delle dispense,
già in casa.

### Cosa è già stato implementato

Tre voci della roadmap sono già fatte e misurate — integratore RK2 (§3.1), analisi KKT (§2.1),
penalità esatta ℓ¹ (§2.2) — oltre all'infrastruttura di analisi offline in [`viz/`](../viz/).

**Il riepilogo con i risultati e la lista di cosa resta sono nella [§10](#10-stato-di-avanzamento)**,
in fondo a questo documento. Le singole schede qui sotto riportano i numeri nel dettaglio.

---

## 1. Capitolo 7 — Soluzione dei problemi di controllo ottimo *(il baricentro)*

### 1.1 Path following con l'ascissa come variabile decisionale ⭐

- **Corso** — §7.2.4, eq. (7.5). Testualmente: *"la velocità alla quale il sistema riesce a
  seguire il percorso non è nota a priori, quindi una formulazione di inseguimento di
  traiettoria è difficile da impostare, perché si dovrebbe decidere arbitrariamente (o con
  euristiche) l'andamento temporale dei riferimenti"*. La soluzione: rendere gli incrementi
  `Δθ(t)` variabili decisionali con `Δθ ≥ 0`, `θ(0) = 0`, `θ(N) = 1`, e mettere a costo
  `(1 − θ(t))²`.
- **Stato** — **assente, ed è il caso descritto alla lettera**.
  [`mpc_tracker.py::_build_reference`](../src/a_star_mpc_planner/a_star_mpc_planner/mpc_tracker.py)
  campiona il riferimento lungo il path A\* a velocità costante:
  `s_k = min(s0 + v_ref·k·dt, total)`. Con `v_ref = 0.2` m/s scelto a mano, è precisamente la
  scelta arbitraria che il corso raccomanda di eliminare.

  *(Nota: il commento `# (not used)` accanto a `mpc_v_ref` nel vecchio `planner_params.yaml`
  è errato — il parametro è usato eccome, alla riga 454.)*
- **Da aggiungere** — riformulazione (7.5):

  ```
  min  α₁ Σ_k (z̄(θ_k) − z_k)ᵀ Q (z̄(θ_k) − z_k) + α₂ Σ_k u_kᵀ R u_k + α₃ Σ_k (1 − θ_k)²
  s.t. dinamica del robot                     (invariata)
       θ_{k+1} = θ_k + Δθ_k                   (nuova, banale)
       Δθ_k ≥ 0                               (monotonia lungo il path)
       θ_0 = 0,  θ_N ≤ 1                      (disuguaglianza: il path può non finire)
       box su u                               (invariati)
  ```

  `z̄(θ)` è il path A\* parametrizzato: già disponibile, basta sostituire il campionamento a
  passo `v_ref·k·dt` con l'interpolazione in θ. Aggiunge N+1 variabili e N disuguaglianze —
  quindi **cambia anche il bilancio della §2.4**.
- **Perché conta qui in particolare** — con l'orizzonte deployato il robot copre **0.60 m**,
  contro un `mpc_lookahead_dist` di 0.45 m. Il riferimento temporale a velocità fissa è quindi
  tarato su un margine minuscolo: se il G1 rallenta (curva stretta, ostacolo), il riferimento
  gli scappa avanti e il costo di tracking cresce per una ragione che **non ha nulla a che
  vedere con la qualità del controllo**. La parametrizzazione in θ elimina esattamente questo
  artefatto.
- **Verifica** — accuratezza geometrica (distanza dal path, non dal riferimento temporale),
  tempo di completamento ed energia di controllo per diverse terne (α₁, α₂, α₃) ⇒ alimenta
  direttamente la §1.6 (Pareto). Confronto diretto su `viz/bags/industrial_plant_fix`.
- **File** — `mpc_tracker.py::_build_reference` e `::_build_nlp`
- **Sforzo** — **M** · **massimo rapporto rilevanza / dimensione della modifica**

---

### 1.2 Ingredienti terminali e fattibilità ricorsiva ⭐

- **Corso** — §7.2.5. Distinzione fondamentale tra **insieme di fattibilità F** (gli stati z₀
  per cui l'FHOCP ha soluzione) e **insieme ammissibile Ω** (i valori di U ammissibili dato z₀).
  Se la traiettoria parte in F e ne esce, la fattibilità ricorsiva non vale. Il rimedio
  raccomandato per NMPC è il **vincolo terminale di equilibrio**: la coda della sequenza
  precedente più l'ingresso di equilibrio è sempre ammissibile, quindi Ω ≠ ∅. Il corso segnala
  anche che nella pratica i vincoli si implementano **soft**, con slack penalizzati.
- **Stato** — **assente, e sostituito da due euristiche** che sono, letteralmente, gestioni di
  perdita di fattibilità ricorsiva:
  - limiti di velocità ridotti quando il tasso di fallimento IPOPT supera il 30 %
    (`adaptive_vel_limits` in `mpc_node.py`);
  - fallback a velocità nulla dopo `_MAX_CONSEC_FAILURES = 3` solve falliti.
- **Materiale sperimentale già in casa** — la seconda euristica si è rivelata un **latch
  permanente**: il ritorno anticipato precedeva l'unico punto che azzerava il contatore, quindi
  al terzo fallimento consecutivo il fallback si auto-alimentava per il resto della missione.
  Misurato su `viz/bags/industrial_plant`:

  | | con il latch | dopo la correzione |
  |---|---|---|
  | cicli con solve riuscito | 267 / 876 (30 %) | 775 / 775 (100 %) |
  | cicli in cui IPOPT non è mai partito | 606 | 0 |
  | `solve_ms` medio | 25.5 (gonfiato dai salti) | 70.0 |
  | distanza finale dal goal | 0.29 m | 0.18 m |

  Rigiocando la bag attraverso il tracker corretto: **99 % di successi**. Le NLP erano tutte
  risolvibili; nessun fallimento era dovuto a difficoltà numerica. Il robot ha comunque
  raggiunto il goal, **navigando due terzi della missione senza MPC**, sul solo path A\*.
- **Perché è un risultato e non solo un bug** — è la dimostrazione empirica dell'argomento di
  §7.2.5: in assenza di ingredienti terminali che *garantiscano* Ω ≠ ∅, si finisce a scrivere
  fallback ad hoc, e i fallback ad hoc hanno modi di fallimento propri. Il corso lo dice in
  astratto; qui c'è la misura.
- **Da aggiungere** —
  1. **vincolo terminale di equilibrio** `v_N = 0`: il robot si può sempre fermare entro
     l'orizzonte. Sul modello a 6 stati è immediato perché la velocità è nello stato — e resta
     immediato anche nella degenerazione del §0.2, perché equivale a `u_{N−1} = 0`;
  2. **slack sul vincolo terminale**, penalizzato in ℓ¹ (§2.2), così il problema resta sempre
     ammissibile come raccomanda il corso;
  3. in alternativa più ambiziosa, **insieme terminale come sottolivello di una funzione di
     Lyapunov**: `X_f(α) = {x : xᵀPx ≤ α}` con P dalla DARE del problema LQ ausiliario, e α
     scelto come `min_i d_i²/(c_iᵀP⁻¹c_i)` perché `X_f(α)` rispetti tutti i vincoli. È la
     costruzione classica ed è nelle dispense; va però detto che su un modello non lineare
     l'argomento LQ vale solo localmente;
  4. **discussione onesta**: le due euristiche restano nel codice come misure di sicurezza, ma
     vanno presentate come *"ciò che facevamo in assenza di ingredienti terminali"*, con il
     confronto quantitativo prima/dopo — che ora esiste.
- **Verifica** — tasso di fallimenti IPOPT e di attivazione del fallback, con e senza vincolo
  terminale, sulle bag già registrate; percentuale di solve con slack terminale > 0.
- **Sforzo** — **M** (punti 1-2 e 4) · **L** se si fa anche il punto 3

---

### 1.3 Lunghezza dell'orizzonte: prestazione contro tempo di calcolo

- **Corso** — §7.2.5. L'orizzonte è il parametro che governa il compromesso fra qualità
  dell'anticipazione e costo computazionale; la fattibilità in tempo reale è un vincolo di
  progetto, non un dettaglio.
- **Stato** — **N = 15 senza alcuna giustificazione scritta**, ereditato dalla taratura
  hardware del G1.
- **Perché qui è particolarmente critico** — l'orizzonte copre **0.60 m**. Un ostacolo visto a
  1 m è fuori dall'orizzonte di predizione: l'MPC non lo vede proprio, e questo spiega da solo
  il fatto misurato al §0.3 (l'evitamento lo fa A\*). Non è una taratura debole, è una scelta
  strutturale che va o giustificata o cambiata.
- **Da aggiungere** — sweep N ∈ {5, 10, 15, 25, 40, 60} a dt fisso, riportando per ciascuno:
  - metrica di prestazione in anello chiuso (distanza finale, clearance minima, lunghezza del
    percorso, energia di controllo) sulle bag già registrate;
  - distribuzione del tempo di solve (mediana, p95, massimo) — non solo la media: il vincolo
    real-time lo viola la **coda**, e nella nostra bag c'è già un solve da 1393 ms contro un
    p95 di 102 ms;
  - numero di iterazioni IPOPT (ora disponibile su `/mpc/diagnostics`).

  Va discusso anche il compromesso alternativo: allungare l'orizzonte **temporale** alzando
  `dt` invece di N, che non aumenta le variabili ma peggiora l'errore di discretizzazione (§3.1).
- **Verifica** — due grafici: metrica di prestazione vs N, e boxplot del tempo di solve vs N
  con la linea del budget (125 ms a 8 Hz). Il punto scelto va motivato su quei grafici.
- **Sforzo** — **M** · l'infrastruttura di replay c'è già, lo sweep è uno script

---

### 1.4 Modello di predizione contro modello di simulazione

- **Corso** — §7.2.5. L'MPC predice con un modello nominale; l'impianto è un'altra cosa. Il
  disallineamento produce un offset di regime che la sola retroazione a orizzonte recedente
  non annulla.
- **Stato** — **presente nei fatti, assente nella trattazione**. Il modello di predizione è un
  uniciclo (più il lag degenere del §0.2); l'impianto è **MuJoCo con il G1 a 29 gradi di
  libertà** che cammina. Il disallineamento è enorme e completamente non modellato: passi
  discreti, oscillazione laterale del bacino, ritardo del controllore di camminata.
- **Perché è un punto forte e non una debolezza** — è un caso di studio migliore di un
  disturbo additivo sintetico: il mismatch è **reale e strutturale**, non iniettato a mano.
- **Da aggiungere** —
  1. **quantificarlo**: dalle bag, confrontare la traiettoria predetta a k passi
     (`/mpc/predicted_path`, già registrata) con quella effettivamente percorsa
     (`/robot_pose`). Errore di predizione a 1, 5, 15 passi, in funzione della velocità
     comandata e della curvatura. È una misura che si estrae dai dati esistenti, senza
     nuovi esperimenti;
  2. **discutere l'offset**: in un compito di regolazione il mismatch dà errore permanente.
     Qui il goal viene raggiunto (0.18 m) perché A\* ripianifica, cioè l'anello esterno
     compensa quello interno — va detto esplicitamente, perché è il motivo per cui l'offset
     non si vede;
  3. **eventualmente** un'azione integrale sull'errore di tracking, con lo stato aumentato
     `η_{k+1} = η_k + dt·e_k`. È la ricetta standard e costa uno stato.
- **Verifica** — curva dell'errore di predizione vs orizzonte di predizione; confronto con e
  senza azione integrale sulla stessa bag.
- **Sforzo** — **S** (punto 1, solo analisi) · **M** (punto 3)

---

### 1.5 Single shooting contro multiple shooting

- **Corso** — §7.2.2, eq. (7.3) single vs eq. (7.4) multiple, Fig. 7.2. Vantaggio del multiple:
  il modello non viene mai integrato in anello aperto per più di un passo; svantaggio: più
  variabili e vincoli, compensati dalla **sparsità** e da un numero minore di iterazioni.
- **Stato** — multiple shooting **usato** (X e U entrambe variabili decisionali, dinamica
  imposta come uguaglianza), mai dichiarato né motivato né misurato.
- **Da aggiungere** — la versione single-shooting a confronto: eliminando X per sostituzione si
  passa a 45 variabili e nessun vincolo di dinamica, ma con Jacobiana **densa**. Misurare
  entrambe.

  La metà multiple-shooting della tabella è già misurabile (profilo G1, `nlp_structure.py`):

  | N | variabili | vincoli | jac nnz | densità jac | densità hess |
  |---|---|---|---|---|---|
  | 10 | 96 | 106 | 236 | 2.32 % | 6.29 % |
  | **15** | **141** | **156** | **351** | **1.60 %** | **4.30 %** |
  | 25 | 231 | 256 | 581 | 0.98 % | 2.63 % |
  | 50 | 456 | 506 | 1156 | 0.50 % | 1.34 % |

  L'osservazione da fare è che **la sparsità è un vantaggio che cresce con N**: a N = 15 la
  Jacobiana è densa all'1.6 %, a N = 50 allo 0.5 %. Su un orizzonte corto come quello deployato
  il guadagno strutturale del multiple shooting è quindi il più piccolo dell'intervallo, e il
  confronto con il single shooting va misurato invece che assunto — potrebbe non vincere.
- **Verifica** — la tabella sopra estesa al single shooting; iterazioni e tempo per solve delle
  due formulazioni sullo stesso ciclo estratto dalle bag.
- **Sforzo** — **M** · `guides/snippets/nlp_structure.py` produce già metà della tabella

---

### 1.6 Parametrizzazione dell'ingresso e move blocking

- **Corso** — §7.2.3: feedback di stato parametrizzato eq. (3.13), **move blocking** (ingresso
  costante su più intervalli), strategie di commutazione per ingressi discreti. Motivazione:
  disaccoppiare il numero di variabili dall'orizzonte.
- **Stato** — **assente**: 45 ingressi liberi su N = 15. C'è però già una penalità sul jerk
  (`R_jerk = 0.4`), che è una regolarizzazione degli incrementi ma non una riduzione delle
  variabili.
- **Da aggiungere** —
  1. **move blocking** con blocchi crescenti (1,1,1,2,2,4,4): l'orizzonte resta 3 s ma le
     variabili scendono da 45 a ~21. Su un problema che gira a 8 Hz nominali questo è
     direttamente budget di calcolo recuperato — e quel budget serve, vista la §1.3;
  2. **forma in incrementi**: ottimizzare Δu con `u_k = u_{k−1} + Δu_k` e `u_{−1}` nello stato.
     Rende naturali i vincoli di rate e migliora il condizionamento. Nota: data la
     degenerazione del §0.2, lo stato velocità **è già** `u_{k−1}`, quindi questa
     riformulazione qui costa zero stati aggiuntivi. È un allineamento fra teoria e
     implementazione che vale la pena rivendicare.
- **Verifica** — Pareto variabili/tempo di solve contro qualità della traiettoria; da fare
  insieme allo sweep della §1.3.
- **Sforzo** — **S/M**

---

### 1.7 Omotopia e warm starting

- **Corso** — §7.1.1 eq. (7.2), Fig. 7.1: risolvere una successione di problemi con un peso che
  decresce gradualmente, inizializzando ogni solve con la soluzione precedente. §7.2.5 ultimo
  punto: warm start in NMPC con la coda della sequenza precedente.
- **Stato** — warm start **sì** (soluzione shiftata di un passo, con svuotamento su fallimento
  e su picco di costo), omotopia **no**, nessun confronto con un baseline.
- **Da aggiungere** —
  1. **Baseline del warm start**: confrontare tre inizializzazioni — soluzione shiftata,
     traiettoria di riferimento, cold start a zero — su iterazioni medie e tempo. Il
     riferimento viene calcolato comunque a ogni ciclo, quindi è il baseline onesto, non lo
     zero. Ora che `iter_count` è esposto, la misura è diretta.
     *(Nel progetto gemello questo esperimento ha dato un risultato controintuitivo: la
     soluzione shiftata perde contro il riferimento in 4 casi su 4.)*
  2. **Omotopia su α**: partire con `obs_alpha` piccolo (barriera morbida, problema ben
     condizionato), risolvere, alzare α e riusare la soluzione, ripetere. È la continuazione
     della §2.5 e la ricetta della §7.1.1 applicata alla lettera.
- **Verifica** — iterazioni e tasso di successo, per α fisso vs α in omotopia, sulle bag
  esistenti. `decision_plane.py --set` fa già gli sweep parametrici.
- **Sforzo** — **S** (baseline) + **M** (omotopia)

---

### 1.8 Ottimizzazione multi-obiettivo fatta come da programma

- **Corso** — §7.4: tre strategie (vincoli su tutti gli obiettivi tranne uno eq. (7.8); somma
  pesata eq. (7.9); mista). Procedura a-posteriori: **(I)** normalizzare gli obiettivi allo
  stesso ordine di grandezza, **(II)** risolvere ripetutamente campionando i pesi sul simplesso
  `A = {α ≥ 0, Σαᵢ = 1}` includendo i vertici, **(III)** post-processare. Definizioni: fronte
  di Pareto, punti non dominati, **punto Utopico**, scelta come punto più vicino all'Utopico in
  norma 2. Strumenti grafici: **curva di Pareto** (Fig. 7.9) e **spider chart** (Fig. 7.10).
  Avvertenza: la somma pesata recupera il fronte completo solo se questo è convesso.
- **Stato** — **presente ma metodologicamente incompleto**. Il punteggio composito in
  [`tuning/`](../tuning/) è una somma pesata di 5 obiettivi (successo, efficienza di percorso,
  smoothness, clearance, tempo) con pesi fissi scelti a mano: è la strategia 2 del §7.4,
  valutata **in un solo punto α**. Mancano normalizzazione dichiarata, campionamento del
  simplesso, fronte, punto Utopico e grafici.
- **Da aggiungere** — cambiare procedura, non strumenti:
  1. normalizzare i 5 obiettivi (punto I);
  2. campionare α sul simplesso, includendo i 5 vertici — che danno il punto Utopico (punto II);
  3. tracciare la curva di Pareto sulle coppie più significative (lunghezza percorso vs tempo
     al goal; clearance minima vs tempo), lo spider chart di 2-3 soluzioni non dominate, e la
     scelta finale come punto più vicino all'Utopico;
  4. dichiarare esplicitamente se il fronte osservato è convesso, e se non lo è, dirlo.
- **Perché conta** — è il modo di trasformare la parte più corposa e più fuori programma del
  repository in contenuto d'esame (§8).
- **Sinergia** — la §1.1 introduce naturalmente tre obiettivi (accuratezza, sforzo, tempo di
  completamento) con i pesi α₁, α₂, α₃ già nella forma della eq. (7.9): il fronte di Pareto su
  quei tre è il caso di studio più pulito, e viene gratis.
- **Sforzo** — **M**

---

### 1.9 Evitare la programmazione mista-intera: rivendicarlo

- **Corso** — §4.2.6: i MIP non sono né lisci né convessi, l'enumerazione esplode, *"è
  consigliabile evitare un gran numero di variabili intere"*; una parametrizzazione adeguata
  può mappare variabili continue su decisioni discrete.
- **Stato** — la scelta c'è, l'argomento no.
- **Da aggiungere** — un paragrafo: la decisione discreta "da che lato aggiro l'ostacolo" è
  intrinsecamente combinatoria e in una formulazione monolitica richiederebbe variabili
  binarie. Qui viene delegata ad A\* sulla griglia di occupazione, e l'NLP resta **continuo e
  liscio**. È il motivo architetturale per cui esiste la separazione A\* / MPC, oggi presentata
  come dettaglio implementativo.

  Il prezzo va detto: l'ottimalità è quella del percorso scelto da A\*, non globale. E il dato
  del §0.3 lo rende quantitativo — ai pesi deployati la scelta di omotopia la fa **interamente**
  A\*, perché l'NLP non ha nemmeno due minimi fra cui scegliere.
- **Sforzo** — **S**

---

### 1.10 MPC approssimato per approssimazione di funzione

- **Corso** — §7.2.5: se l'FHOCP non è convesso la legge di controllo può essere discontinua;
  diversi approcci usano reti neurali per derivare offline leggi NMPC approssimate da un
  insieme di punti `u⁽ⁱ⁾ = κ_MPC(z⁽ⁱ⁾)`, riducendo il carico online.
- **Stato** — [`PointCloud-GNNencoder/`](../src/PointCloud-GNNencoder/) esiste ed è di fatto
  fuori programma. **Questo è il suo unico aggancio legittimo.**
- **Raccomandazione** — **escluderlo** (§8). Riformularlo richiederebbe di sviluppare davvero
  l'approssimazione della legge di controllo, che è un progetto a sé; citarlo senza svilupparlo
  non aggiunge contenuto.
- **Sforzo** — **S** (esclusione) / **L** (svilupparlo)

---

## 2. Capitolo 6 — Ottimizzazione vincolata *(il macchinario)*

### 2.1 KKT, LICQ e moltiplicatori per *questo* problema

- **Corso** — §6.1.1 (cono tangente, active set, LICQ Def. 6.1.5), §6.1.2 (lagrangiana
  eq. (6.7), lemma di Farkas Thm 6.1.3, condizioni KKT eq. (6.8) con complementarità), §6.1.3
  (cono delle direzioni critiche eq. (6.11), NOC-C-2 Thm 6.1.5, SOC-C-2 Thm 6.1.6, Hessiana
  proiettata).
- **Stato** — **assente**.
- **Da aggiungere** — la trattazione completa, che qui è **corta e pulita** perché i vincoli
  sono semplici:

  ```
  L(X, U, λ, μ) = J(X, U)
                − Σ_k λ_kᵀ [X_{k+1} − f(X_k, U_k)]    − λ_0ᵀ [X_0 − x̂]
                − Σ_k μ_kᵀ h_box(U_k)
  ```

  - **LICQ vale banalmente**: i vincoli attivi sono box su componenti distinte di U, i cui
    gradienti sono versori distinti, quindi linearmente indipendenti (Def. 6.1.5). Si dimostra
    in tre righe e si può dichiarare senza cautele.
  - **Moltiplicatori**: leggerli con `opti.dual(constraint)` e mostrare quali box sono attivi
    lungo l'orizzonte. Sul G1 l'aspettativa è specifica e verificabile: `vy_max = 0.02` m/s è
    così stretto che il box laterale dovrebbe essere **quasi sempre attivo**, mentre
    `U[0] ≥ 0` (nessuna retromarcia) dovrebbe attivarsi nelle manovre strette.
  - **Complementarità stretta**: verificare se ci sono vincoli debolmente attivi (h_i = 0 e
    μ_i = 0), il caso in cui il cono critico non degenera in un sottospazio (§6.1.3).
  - **SOC-C-2**: proiettare l'Hessiana della lagrangiana sul cono critico e verificarne la
    definita positività ⇒ certificato di minimo locale, il massimo ottenibile in un problema
    non convesso.
- **FATTO** — [`viz/kkt_analysis.py`](../viz/kkt_analysis.py), che opera su un ciclo reale
  estratto da una bag. Risultati su `industrial_plant_fix`, nove cicli lungo la missione:

  | ciclo | vincoli attivi | rango | LICQ | dim. cono critico | compl. stretta | λ_min proiettato |
  |---|---|---|---|---|---|---|
  | 50 | 96 | 96 | ✓ | 45 | ✓ | +8.78e-01 |
  | 250 | 101 | 101 | ✓ | 40 | ✓ | +8.84e-01 |
  | 450 | 135 | 135 | ✓ | 6 | ✓ | +3.71e+00 |
  | 650 | 138 | 138 | ✓ | 3 | ✓ | +2.16e+01 |
  | 718 | 140 | 140 | ✓ | 1 | ✓ | +7.83e+02 |

  - **LICQ vale in ogni ciclo** (rango = numero di vincoli attivi) ⇒ i moltiplicatori KKT
    esistono e sono unici;
  - **complementarità stretta in ogni ciclo** ⇒ il cono critico coincide con il nucleo del
    Jacobiano attivo, e SOC-C-2 si verifica esattamente invece che per difetto;
  - **SOC-C-2 soddisfatta ovunque** (λ_min > 0) ⇒ certificato di minimo locale stretto, il
    massimo ottenibile in un problema non convesso;
  - stazionarietà: `‖∇_x L(x*, λ*)‖_∞ = 6.1e-11` contro `‖∇_x f(x*)‖_∞ = 1.6e+04`.

- **Il risultato che vale la pena raccontare** — la dimensione del cono critico **varia fra
  45 e 1** a seconda del punto di lavoro. Nei cicli più vincolati restano 141 variabili e 140
  vincoli attivi: **un solo grado di libertà residuo**. Lì l'MPC non sta più *scegliendo* una
  traiettoria, la sta subendo: è guidato dai vincoli, non dal costo.

  *(Correzione rispetto a una prima lettura: non è un collasso monotono lungo la missione.
  Campionando l'intera run si ottiene 45 → 1 → 21 → 4 → 8. Vale l'identità esatta
  `dim(cono) = n_var − vincoli attivi`, quindi la dimensione è semplicemente il complemento
  della saturazione, e segue la difficoltà istantanea della manovra invece del tempo.)*

  La ripartizione dei vincoli attivi lo spiega: `|vy| ≤ vy_max` e `|w| ≤ omega_max` sono
  saturati **15 volte su 15** nei cicli finali, e `vx ≥ 0` 14 su 15. Conferma quantitativa
  della previsione fatta qui sopra: `vy_max = 0.02` m/s è così stretto da essere sempre
  attivo, e di fatto rimuove un grado di libertà invece di limitarlo.

- **Due trappole trovate implementandolo**, che vale la pena documentare perché ricorrono:
  1. Opti canonizza ogni vincolo come `lbg ≤ g(x) ≤ ubg` e **assorbe nei limiti** il membro
     destro quando è un parametro: `X[:,0] == p_x0` diventa `g = X[:,0]`, `lbg = ubg = p_x0`.
     Valutare `|g|` su quelle righe restituisce *lo stato*, non il residuo. Il residuo è
     sempre `g − lbg`.
  2. Due righe distinte possono condividere la stessa espressione con limiti diversi
     (`U[0,k] ≥ 0` e `U[0,k] ≤ vx_max` sono entrambe la riga `U[0,k]`). L'attivazione va
     quindi decisa sulla **distanza dal limite**, non su `|g| ≈ 0`, altrimenti si contano
     attivi entrambi i lati di uno stesso box.
- **Sforzo** — **M** · **priorità massima**: sblocca la §2.2 e la §2.3

---

### 2.2 Vincolo ostacolo *vero* con slack, e penalità esatta ℓ¹ ⭐

- **Corso** — §6.3.3 eq. (6.28) merit function ℓ¹, **Thm 6.3.1**: se `σ_i > |λ_i*|` e
  `τ_i > |μ_i*|`, allora x\* è minimizzatore anche della funzione di merito non vincolata. La
  penalità ℓ¹ è **esatta**: lo slack va a zero non appena ρ supera il moltiplicatore. Una
  penalità quadratica lascia invece un residuo `s* ≈ μ*/(2ρ)` per ogni ρ finito.
- **Stato** — **assente e sostituito da un'euristica**. Oggi gli ostacoli sono solo un costo,
  senza slack e senza vincolo; il peso `W_obs_sigmoid` è un iperparametro tarato per tentativi.
- **Perché è il buco più grande** — il §0.3 lo quantifica: ai pesi deployati il termine
  ostacolo non riesce nemmeno a creare un secondo minimo locale. Non è che l'MPC scelga male
  come evitare l'ostacolo; è che non sceglie affatto.
- **Da aggiungere** —

  ```
  vincolo:  ‖p_k − o_j‖ ≥ d_safe − s_{jk},   s_{jk} ≥ 0
  costo:    + ρ Σ_{j,k} s_{jk}        (ℓ¹, esatta, lineare ⇒ NLP resta liscio)
       o    + ρ Σ_{j,k} s_{jk}²       (ℓ², residuo non nullo)
  ```

  Procedura: risolvere una volta con vincolo hard, leggere μ\* dalla soluzione duale (§2.1),
  scegliere ρ > max|μ\*|. **A quel punto ρ smette di essere un parametro da tarare** e diventa
  una quantità letta dal problema.

  L'infrastruttura per l'ultimo passo esiste già: `decision_plane.py` calcola σ = 1.5·max|λ|
  dai moltiplicatori veri di IPOPT (misurato: σ = 3.996·10⁴ sul ciclo 718 di
  `industrial_plant_fix`).
- **FATTO** — `MPCConfig.obstacle_mode` ∈ {`penalty`, `l1`, `l2`} in `MPCTracker`, più
  [`viz/exact_penalty.py`](../viz/exact_penalty.py) per lo sweep. Il modo deployato resta
  `penalty`: verificato che `J*` sia **bit-identico** al valore pre-modifica
  (8177.231314839336) e che i vincoli restino 156, cioè nessuna variabile aggiunta.

  Scenario `narrow_gap`, `d_safe = 1.1` m (scelto perché il vincolo **morda**: la clearance
  naturale è 0.76 m). Moltiplicatore letto dal problema: **max|μ\*| = 8.905e+03**.

  | ρ | max s\* (ℓ¹) | max s\* (ℓ²) |
  |---|---|---|
  | 1e+03 | 1.457e-01 | 2.606e-01 |
  | **1e+04** | **0** (−9.8e-09) | 1.297e-01 |
  | 1e+05 | **0** | 3.975e-02 |
  | 1e+06 | **0** | 4.375e-03 |
  | 1e+07 | **0** | 4.420e-04 |
  | 1e+08 | **0** | 4.424e-05 |

  - **ℓ¹**: lo slack diventa **esattamente nullo** a partire da ρ = 1e4 — e la soglia teorica
    max|μ\*| = 8.9e3 cade esattamente fra 1e3 (dove s\* = 0.146) e 1e4. **Thm 6.3.1
    confermato sperimentalmente.**
  - **ℓ²**: il residuo decresce con pendenza log-log **−1.00 esatta** sulla coda (−0.78 se si
    include il regime pre-asintotico) e **non si annulla mai**: 4.4e-05 ancora a ρ = 1e8.

  ρ smette quindi di essere un iperparametro: si legge dal problema.

- **Due difetti di formulazione emersi implementandolo**, entrambi di interesse teorico:
  1. **Il vincolo non va imposto a k = 0.** Lì lo stato è fissato da `X[:,0] == x0`, quindi il
     vincolo non dipende da alcuna variabile decisionale: imporlo rende l'NLP inammissibile
     ogni volta che il robot si trova già entro `d_safe` da un ostacolo — cioè proprio quando
     servirebbe. Misurato: con `d_safe = 0.9` m sulla bag, lo slack saturava a 0.1999 =
     esattamente la violazione a k = 0 (clearance corrente 0.7017 m). Ora il vincolo parte da
     k = 1 e lo slack ha N colonne, non N+1.
  2. **L'inammissibilità residua è colpa di U_Σ, non dei dati.** Anche partendo da k = 1, sul
     ciclo reale `d_safe = 0.75` m resta inammissibile: con `vx ≥ 0` e `vy_max = 0.02` m/s il
     robot in un passo può solo avanzare di 0.06 m lungo la propria direzione, e non può
     arretrare lateralmente dal muro di punti LiDAR. È l'interazione fra l'asimmetria
     dell'insieme ammissibile degli ingressi (§5.1) e un vincolo di stato, e collega
     direttamente questa scheda alla §1.2 (fattibilità ricorsiva).
- **Impatto sul repo** — rende superflua una parte della campagna BO, il pezzo più fuori
  programma del progetto (§8).
- **Costo da mettere in conto** — la formulazione vincolata porta le disuguaglianze da 60 a
  oltre 300 e le iterazioni da ~20 a 60-240: il cap deployato `mpc_max_iter = 40` non basta.
  Questo **ribalta l'argomento della §2.4** (active-set contro interior-point), come previsto.
- **Sforzo** — **M** · dipende da §2.1

---

### 2.3 SQP e Gauss-Newton vincolato scritti a mano

- **Corso** — §6.3.1 (SQP con Hessiana esatta, eq. (6.22)), §6.3.2 (**Gauss-Newton vincolato**,
  eq. (6.24)-(6.26); il corso lo definisce *"spesso il metodo migliore nei problemi di stima e
  controllo"*; e BFGS vincolato con trucco di Powell), §6.3.3 (line search con merit function),
  §6.3.4 (**Algoritmo 6.3.1** completo).
- **Stato** — **assente**. È il cuore dei laboratori del corso, quello che si scrive a mano.
- **Nota favorevole** — il costo di questo problema è **quasi** una somma di quadrati: tracking
  `‖e‖²_Q`, sforzo `‖u‖²_R`, jerk `‖Δu‖²`, penetrazione `max(0, ·)²` lo sono; **solo la
  sigmoide non lo è**. Quindi F(x) si costruisce direttamente e `H = 2 ∇F ∇Fᵀ` è disponibile
  senza derivate seconde, con `H ⪰ 0` garantito — che è precisamente l'argomento con cui si
  sceglie GN su un problema non convesso: evita QP interni non convessi.
- **Da aggiungere** — implementare l'Algoritmo 6.3.1 con:
  - QP interno risolto con `qpOASES` o `osqp`;
  - Hessiana da Gauss-Newton vincolato, e in alternativa BFGS + Powell;
  - back-tracking con merit function ℓ¹ e condizione di Armijo eq. (6.36);
  - aggiornamento dei pesi σ, τ come al punto 5 dell'Algoritmo 6.3.1.

  Due varianti dell'obiettivo, per rendere GN applicabile: (a) sostituire la sigmoide con la
  sola hinge² (somma di quadrati pura), (b) tenerla e trattarla come termine extra
  nell'Hessiana approssimata.
- **Verifica** — confronto con IPOPT sullo stesso ciclo estratto dalle bag: iterazioni, tempo
  per solve, costo finale, traiettoria. Attesa dalla teoria: GN lineare lontano dalla soluzione,
  quadratico vicino; IPOPT più robusto ma con più lavoro per iterazione. Il riferimento
  numerico c'è già: media 13.4 iterazioni, 70 ms per solve.
- **Sforzo** — **L** · il pezzo più corposo, e il più caratterizzante per l'esame

---

### 2.4 Active-set contro interior-point

- **Corso** — §6.2.1 (QP con sole uguaglianze: sistema KKT eq. (6.15), risolubile
  direttamente), §6.2.2 (active-set Fig. 6.10 vs interior-point, barriera logaritmica
  eq. (6.18), central path, τ → 0, Fig. 6.11–6.12). La regola pratica del corso: **active-set
  conviene con poche disuguaglianze, interior-point con molte**.
- **Stato** — IPOPT usato senza alcuna motivazione scritta.
- **Da aggiungere** — l'osservazione che oggi le disuguaglianze sono **60 righe, tutte box
  sugli ingressi**, cioè esattamente la situazione in cui il corso dice che l'active-set è
  competitivo. Provare `qpOASES` dentro SQP (§2.3) e confrontare.

  **Il punto interessante è che l'argomento si ribalta**: dopo la §2.2 gli ostacoli diventano
  vincoli e le disuguaglianze passano da 60 a 60 + 128·2, cioè da poche a molte. Il confronto
  va quindi fatto **prima e dopo**, ed è un risultato di per sé: la scelta del solutore QP non
  è una preferenza, è una conseguenza della formulazione.
- **Sforzo** — **M** · dipende da §2.3

---

### 2.5 La barriera dell'interior-point e la barriera degli ostacoli sono la stessa cosa

- **Corso** — §6.2.2 eq. (6.18): l'interior-point sostituisce le complementarità non lisce con
  `−τ Σ ln(C_i x + d_i)`, τ decrescente. Fig. 6.12: al calare di τ la barriera diventa più
  ripida e il condizionamento peggiora.
- **Da aggiungere** — il parallelo esplicito: `obs_alpha` nella sigmoide gioca **esattamente**
  il ruolo di 1/τ. Barriera più ripida = evitamento migliore e condizionamento peggiore, la
  stessa identica curva di compromesso. Trasforma un iperparametro tarato per tentativi in un
  oggetto teorico noto, ed è l'aggancio naturale alla §1.7 (omotopia).
- **Verifica** — sweep di `obs_alpha` con iterazioni IPOPT e numero di condizionamento
  dell'Hessiana. Misura diretta ora che `iter_count` è esposto.
- **Sforzo** — **S**

---

## 3. Capitolo 2 — Modelli e discretizzazione

### 3.1 Ordine di troncamento: Euler contro mid-point (RK2)

- **Corso** — §2.1.3, eq. (2.9) Euler in avanti, eq. (2.10) RK2 / regola del punto medio,
  Fig. 2.3–2.4. Euler ha errore locale O(dt²) e globale O(dt); RK2 locale O(dt³), globale O(dt²).
- **Stato** — Euler puro sul canale di posizione: `cos(yaw_k)`, `sin(yaw_k)` valutati a
  **inizio** intervallo, con disallineamento sistematico ω·dt/2. Sul profilo G1
  (dt = 0.2 s, ω_max = 0.3 rad/s) sono **0.03 rad per passo**, cioè 1.7°, che su 15 passi si
  accumulano.
- **Da aggiungere** — regola del punto medio sul solo canale di posizione:

  ```python
  yaw_eval = yaw_k + 0.5 * wz_next * dt      # RK2, eq. (2.10)
  px_next  = px_k + (vx_next*cos(yaw_eval) - vy_next*sin(yaw_eval)) * dt
  py_next  = py_k + (vx_next*sin(yaw_eval) + vy_next*cos(yaw_eval)) * dt
  ```

  Costo computazionale: una addizione (`wz_next` serve comunque).
- **FATTO** — `MPCConfig.integrator` ∈ {`euler`, `midpoint`}, esposto come parametro ROS
  `mpc_integrator` e impostato a `midpoint` nel profilo G1. Verificato da
  [`tests/test_integrators.py`](../tests/test_integrators.py) contro la soluzione esatta in
  forma chiusa (per ω ≠ 0 lo spostamento è un arco:
  `Δ_world = (1/ω) R(ψ₀) [[sin a, cos a − 1], [1 − cos a, sin a]] v`, `a = ω dt`).

  | dt [s] | errore Euler [m] | errore punto medio [m] |
  |---|---|---|
  | 0.200 | 1.740e-02 | 8.700e-05 |
  | 0.100 | 8.699e-03 | 2.175e-05 |
  | 0.050 | 4.350e-03 | 5.437e-06 |
  | 0.025 | 2.175e-03 | 1.359e-06 |
  | 0.0125 | 1.087e-03 | 3.398e-07 |

  **Ordine stimato: 1.00 per Euler, 2.00 per il punto medio**, identico su tre regimi
  (nominale, con deriva laterale, rotazione rapida). Il test verifica anche che la dinamica
  costruita dentro l'NLP coincida con lo schema di riferimento: scarto **0.000e+00**.

- **Il risultato onesto in anello chiuso** — a orizzonte recedente il guadagno **quasi
  sparisce**: su quattro scenari, il costo mediano migliora dello 0.0-1.1 % e distanza finale
  e clearance sono invariate. La ragione è strutturale: si applica solo il **primo** ingresso
  e A\* ripianifica, quindi l'errore di predizione accumulato sui 15 passi non si trasferisce
  alla traiettoria percorsa. *(Il confronto è pulito: l'impianto simulato usa un
  aggiornamento di Euler fisso, indipendente da `cfg.integrator`; cambia solo il modello di
  predizione.)*

  Il valore sta quindi nella **fedeltà della predizione**, non nella traiettoria: conta per
  la §1.3 (orizzonti più lunghi, dove l'errore si accumula su più passi) e per la §1.4 (dove
  l'errore di predizione va attribuito al disallineamento di modello, non all'integratore).
- **Perché il passo pesa** — l'errore di Euler è lineare in dt, e qui dt = 0.2 s: l'errore per
  passo è il doppio di quello che si avrebbe a 0.1 s.
- **File** — `mpc_tracker.py::_build_nlp`, nuovo `tests/test_integrators.py`
- **Sforzo** — **S** · miglior rapporto risultato/sforzo di tutta la guida

---

### 3.2 Scelta del passo di campionamento

- **Corso** — §2.1.3. La discretizzazione è esatta agli istanti di campionamento solo se la
  banda del segnale è rispettata; sotto-campionare produce aliasing e distorsione spettrale, e
  il modello di predizione smette di rappresentare l'impianto.
- **Stato** — **assente**. `dt = 0.2` s ed `mpc_rate_hz = 8.0` sono ereditati dalla taratura
  hardware, senza analisi.
- **Osservazione misurata** — il ciclo gira in realtà a **2.6 Hz** (775 cicli in 298.6 s), non
  a 8. Il periodo effettivo è ~0.38 s, quasi **due passi di orizzonte**: il modello di
  predizione avanza a dt = 0.2 s ma il feedback arriva ogni 0.38 s. È un disallineamento fra
  modello e implementazione che va o corretto o dichiarato.
- **Da aggiungere** —
  1. l'analisi di banda del modello: con il lag degenere (§0.2) il canale velocità non ha
     dinamica propria, quindi la banda rilevante è quella della **cinematica**, governata da
     ω_max e dalla curvatura del percorso;
  2. lo studio dell'effetto di dt sull'errore di predizione a orizzonte fisso (3 s), separando
     l'errore di discretizzazione (§3.1) da quello di modello (§1.4);
  3. **la diagnosi del rate effettivo**: perché 2.6 Hz e non 8. Va guardato, perché indebolisce
     il feedback e non è una scelta di progetto.
- **Verifica** — errore di predizione vs dt a orizzonte temporale costante; istogramma del
  periodo effettivo di ciclo dalle bag.
- **Sforzo** — **S** (analisi) + **M** (diagnosi del rate)

---

### 3.3 Discretizzazione esatta ZOH: quando è disponibile, e quando è vuota

- **Corso** — §2.1.3: quando l'ingresso è costante a tratti (ZOH), la traiettoria campionata si
  può calcolare esattamente invece che approssimarla. È l'esempio didattico di "scegliere la
  discretizzazione in base alla struttura del modello".
- **Stato** — il codice usa `lag = 1 − exp(−dt/τ)`, che **è** la ZOH esatta di un primo ordine.
  Con un τ confrontabile con dt (per esempio τ = 0.12 s, dt = 0.1 s) il coefficiente vale 0.565
  ed è un punto di forza reale mai rivendicato.
- **Ma sul G1 è vuota** — con τ = 0.001 s e dt = 0.2 s il coefficiente vale **1.000000000000**
  (§0.2). La formula esatta restituisce il caso degenere `v_{k+1} = u_k`: formalmente corretta,
  fisicamente priva di contenuto.
- **Da aggiungere** — mezza pagina che dice **entrambe** le cose:
  1. il sottosistema velocità è LTI del primo ordine con ingresso costante a tratti per
     costruzione, quindi la ZOH esatta è disponibile e non c'è ragione di usare Euler lì;
  2. ai parametri deployati sul G1 quella struttura **collassa**, e il modello effettivo è un
     uniciclo a 3 stati con un passo di ritardo. Le tre variabili di velocità nello stato sono
     ridondanti: si potrebbero eliminare, riducendo le variabili decisionali da 141 a ~96.

  Riconoscerlo è più forte che rivendicare un modello che non è quello in esecuzione — e apre
  due strade concrete: eliminare la ridondanza (meno variabili, §1.5), oppure identificare un
  τ realistico per il controllore di camminata del G1 e recuperare la dinamica.
- **Sforzo** — **S**

---

## 4. Capitolo 5 — Calcolo delle derivate

### 4.1 AD contro differenze finite: dichiararlo e misurarlo

- **Corso** — §5.2 (differenze finite in avanti e centrate; passo ottimo `√eps ≈ 10⁻⁸` e
  `eps^{1/3} ≈ 7.6·10⁻⁶`; accuratezza attesa ≈ 10⁻⁸ e ≈ 10⁻¹¹; costo n+1 e 2n valutazioni),
  §5.3 (AD forward < 2n, AD backward **< 3 valutazioni indipendentemente da n**, con CasADi
  citato esplicitamente come riferimento [2]), §5.4 (trucco della parte immaginaria).
- **Stato** — CasADi fa AD e nel repo non c'è una riga che lo dica.
- **Da aggiungere** —
  1. sezione di mezza pagina: con n = 141 variabili, le differenze finite centrate costerebbero
     282 valutazioni per gradiente a precisione ≈ 10⁻¹¹, contro < 3 dell'AD backward a
     precisione macchina. A 8 Hz nominali è la differenza fra fattibile e non fattibile — e il
     rapporto peggiora linearmente se si allunga N (§1.3), quindi è **l'argomento che rende
     discutibile un orizzonte più lungo**;
  2. misura empirica: risolvere lo stesso NLP con Hessiana approssimata
     (`hessian_approximation: limited-memory`, cioè L-BFGS) contro l'Hessiana esatta da AD, e
     riportare iterazioni e tempo. È il confronto §4.4.4 (Newton esatto vs quasi-Newton)
     ottenuto senza scrivere un solutore.
- **Verifica** — tabella `t_proc_nlp_f`, `t_proc_nlp_grad_f`, `t_proc_nlp_jac_g`,
  `t_proc_nlp_hess_l` con la quota di tempo nelle derivate. CasADi le restituisce già in
  `sol.stats()`; oggi ne leggiamo solo `iter_count`.
- **Sforzo** — **S** · alto rendimento, quasi tutto già disponibile

---

### 4.2 Ricorsione delle sensitività come controllo incrociato

- **Corso** — §5.1 eq. (5.3): propagazione di `∇_θ ẑ(i, θ)` insieme alla traiettoria.
- **Stato** — assente.
- **Da aggiungere** — con l'identificazione fuori perimetro, questa voce resta utile in un ruolo
  diverso: **verifica di correttezza dell'AD**. Propagare a mano la sensitività della
  traiettoria predetta rispetto a x₀ lungo i 15 passi, e confrontarla con
  `ca.jacobian(X, p_x0)` calcolata da CasADi. Se coincidono a precisione macchina, si è
  dimostrato di aver capito cosa fa l'AD invece di averlo solo usato.

  È anche il modo di ottenere ∂x\*/∂ϑ per la §5.4 (regolarità, Thm 4.4.6).
- **Sforzo** — **S**

---

## 5. Capitolo 4 — Fondamenti e problemi non vincolati

### 5.1 Classificare il problema

- **Corso** — §4.1 (forma standard, Thm 4.1.1 esistenza del minimo globale con Ω compatto),
  §4.2 (LP / QP / convesso / SDP / MIP), §4.2.5 (differenziabile vs non liscio).
- **Stato** — **assente**: non è mai scritto *che tipo* di programma sia.
- **Da aggiungere** — una sezione breve ma precisa:
  - è un **NLP non convesso**, non un QP: la dinamica è nonlineare (R(ψ) moltiplica u) e i
    termini ostacolo sono non convessi;
  - `U_Σ` è compatto (box), `X_Σ` no (gli stati non hanno limiti superiori) ⇒ il Thm 4.1.1 non
    si applica direttamente; l'esistenza si recupera osservando che il costo è radialmente
    illimitato negli stati penalizzati;
  - il termine di penetrazione `max(0, r − d)²` è **C¹ ma non C²**: la sua Hessiana salta sulla
    superficie di attivazione `d = r`, e IPOPT è un metodo di tipo Newton che usa derivate
    seconde (§4.2.5, §4.4.4). La sigmoide invece è C^∞;
  - i box sull'ingresso sono **asimmetrici**: `U[0] ≥ 0` vieta la retromarcia. Va detto, perché
    restringe l'insieme ammissibile in modo qualitativamente diverso da un box simmetrico e
    interagisce con la §1.2 (l'equilibrio `v = 0` è sul bordo, non all'interno).
- **Sforzo** — **S**

---

### 5.2 Analisi di convessità dei due termini ostacolo

- **Corso** — §4.3.1 (condizioni del primo e secondo ordine, Thm 4.3.1–4.3.2), §4.3.2
  (operazioni che preservano la convessità), §4.3.3 (Thm 4.3.3: nei problemi convessi ogni
  minimo locale è globale) e l'esempio finale di riscrittura convessa con cambio di variabile.
- **Stato** — **assente**, ed è un punto in cui è facile sbagliare.
- **Da aggiungere** — l'analisi corretta, che dà un risultato **negativo su entrambi i termini**:
  - **sigmoide** `0.5(1 − tanh(0.5 α (d − r)))`: non convessa per costruzione (convessa oltre
    il flesso, concava prima);
  - **penetrazione** `max(0, r − ‖p − o‖)²`: **anch'essa non convessa**. `‖p − o‖` è convessa,
    quindi `r − ‖p − o‖` è **concava**, e la composizione di una funzione convessa decrescente
    con una funzione concava non è convessa (§4.3.1: la regola di composizione richiede la
    monotonia *crescente*). Attenzione: **non** è lo stesso caso della hinge su un semipiano
    `max(0, d_safe − nᵀ(p − o))²`, che essendo hinge di una funzione **affine** è convessa.
- **Il collegamento con la §3.3 delle dispense (convessificazione)** — la Successive Convex
  Approximation del vincolo di collisione: linearizzare `‖p − o‖ ≥ d_safe` attorno a una
  traiettoria di riferimento dà il semipiano **affine** `nᵀ(p − o) ≥ d_safe`, che è convesso.
  Confrontare le due formulazioni a parità di scenario. *(Già implementata e testata nel
  progetto gemello: `trajopt_core/mpc/obstacles.py::SlackedHalfSpace`.)*
- **Verifica** — autovalori dell'Hessiana del termine ostacolo su una griglia attorno a un
  ostacolo, mostrando dove diventa indefinita. `cost_field.py` costruisce già il campo.
- **Sforzo** — **M**

---

### 5.3 Metodi di Newton, tassi di convergenza, line search

- **Corso** — §4.4.2 (line search, Armijo eq. (4.19), Alg. 4.4.1–4.4.2, Thm 4.4.5 convergenza
  globale, trust region), §4.4.3 (Q-lineare / R-lineare / Q-quadratico / Q-superlineare, e il
  legame `c = (λ_max − λ_min)/(λ_max + λ_min)` con il condizionamento), §4.4.4 (Newton esatto,
  Gauss-Newton, gradient descent, BFGS con trucco di Powell).
- **Stato** — **parzialmente sbloccato**: `iter_count` è ora esposto (media 13.4, massimo 29 su
  una missione reale). Il resto delle statistiche no.
- **Da aggiungere** —
  1. esporre anche `return_status` e i `t_proc_nlp_*` (§4.1);
  2. **studio del condizionamento**: la ripidità della barriera `obs_alpha` gioca il ruolo del
     parametro di barriera inverso; al crescere di α il numero di iterazioni e il numero di
     condizionamento dell'Hessiana peggiorano. È il legame diretto tra §4.4.3 e §6.2.2 e si
     misura con uno sweep (§2.5).
- **Verifica** — grafico iterazioni medie e p95 vs α; quota di tempo nelle callback derivative.
- **Sforzo** — **S** (logging) + **M** (sweep)

---

### 5.4 Regolarità della soluzione rispetto ai parametri

- **Corso** — §4.4.5, Thm 4.4.6: se in x\*(ϑ̄) valgono le SOC-2, allora x\*(ϑ) è ben definita e
  differenziabile in un intorno, con `∂x*/∂ϑ = −(∇²_x f)⁻¹ ∂∇_x f/∂ϑ`. Fig. 4.17: esempio di
  biforcazione, dove questa regolarità si perde.
- **Stato** — **assente come teoria, presente come sintomo**. In `mpc_tracker.py`:
  `_COST_SPIKE_FACTOR = 5.0` con svuotamento del warm start quando il costo supera 5× la media
  recente. È una toppa empirica esattamente al fenomeno del Thm 4.4.6.
- **Risultato già misurato, e va usato** — la biforcazione **non c'è** ai pesi deployati
  (§0.3): due warm start opposti danno soluzioni a distanza 0.0000 in ℝ¹⁴¹, su due missioni
  reali indipendenti. Compare fra W_obs 300 e 600. Quindi:
  - `_COST_SPIKE_FACTOR` sta proteggendo da un fenomeno che, ai pesi attuali, **non si verifica**;
  - la soglia di biforcazione misurata è il modo pulito di presentare la Fig. 4.17 su dati
    propri: uno sweep di W_obs con la distanza fra le due soluzioni in ordinata, che passa da
    zero a non-zero.
- **Da aggiungere** — completare lo sweep W_obs ∈ {120, 200, 300, 450, 600, 900} tracciando
  distanza fra soluzioni da warm start opposti, e sovrapporre il valore deployato. Poi
  ripetere dopo la §2.2, dove i pesi cambiano natura.
- **Verifica** — `decision_plane.py --set mpc_W_obs_sigmoid=...` fa già esattamente questo per
  un valore alla volta; serve solo il ciclo esterno.
- **Sforzo** — **S** (lo sweep) · l'infrastruttura è pronta

---

## 6. Capitolo 3 — Formulazione del problema

### 6.1 Dichiarare l'FHOCP

- **Corso** — §3.3 eq. (3.11), con (3.11f) `h_final(z(N|t), u(N|t)) ≥ 0`; il vincolo terminale
  tipico è `f_z(z(N|t), u(N|t)) = z(N|t)`, cioè lo stato terminale è un equilibrio.
- **Stato** — l'MPC **è** la (3.11), ma manca (3.11f): c'è solo un costo terminale
  `Q_terminal = 100` moltiplicativo su Q.
- **Da aggiungere** — la scrittura formale completa del problema risolto, in notazione delle
  dispense, con la mappa esplicita fra ogni termine e la riga di codice che lo costruisce. È
  mezza pagina e va all'inizio del report: senza, nessuna delle altre sezioni ha un referente.
  Il vincolo terminale mancante è trattato nella §1.2.
- **Sforzo** — **S**

---

## 7. Piano di lavoro

### Blocco A — teoria mancante sul codice esistente *(nessuna modifica al comportamento)*

| # | Voce | §guida | Sforzo |
|---|---|---|---|
| 1 | Scrittura formale dell'FHOCP con mappa al codice | 6.1 | S |
| 2 | KKT, LICQ, moltiplicatori, SOC-C-2 | 2.1 | M |
| 3 | Classificazione del problema, non convessità dei due termini ostacolo | 5.1, 5.2 | M |
| 4 | AD vs differenze finite; Hessiana esatta vs L-BFGS; `t_proc_nlp_*` | 4.1 | S |
| 5 | Tabella sparsità NLP vs N + argomento multiple shooting | 1.5 | M |
| 6 | Sweep di W_obs: soglia di biforcazione (Fig. 4.17 sui dati veri) | 5.4 | S |
| 7 | Baseline del warm start (shiftata / riferimento / cold) | 1.7 | S |
| 8 | Errore di predizione modello vs impianto MuJoCo, dalle bag | 1.4 | S |
| 9 | Rivendicare: ZOH esatta e la sua degenerazione sul G1; MIP evitato via A\* | 3.3, 1.9 | S |

### Blocco B — riformulazioni *(cambiano il comportamento, vanno validate)*

| # | Voce | §guida | Sforzo | Dipende da |
|---|---|---|---|---|
| 14 | Sweep dell'orizzonte N: prestazione vs tempo di calcolo | 1.3 | M | — |
| 15 | Fronte di Pareto sui dati esistenti | 1.8 | M | 10 (fatto: dà già tre obiettivi pesati) |
| 16 | Omotopia su `obs_alpha` | 1.7, 2.5 | M | — |
| 17 | Move blocking / forma in incrementi | 1.6 | S/M | — |

*(10, 11, 12 e 13 sono fatti: vedi §10.8, §10.4, §10.9, §10.2.)*

### Blocco C — il pezzo grosso

| # | Voce | §guida | Sforzo | Dipende da |
|---|---|---|---|---|
| 18 | **SQP + Gauss-Newton vincolato scritti a mano contro IPOPT** | 2.3 | L | 2, 11 |
| 19 | Active-set contro interior-point, prima e dopo il punto 11 | 2.4 | M | 18 |
| 20 | Insieme terminale come sottolivello di Lyapunov (DARE) | 1.2 | L | 12 |

### Se il tempo è poco

I quattro interventi con il miglior rapporto contenuto/sforzo sono **13** (RK2, mezza giornata
e dà un fit d'ordine pulito), **6** (soglia di biforcazione, l'infrastruttura è già pronta),
**11** (penalità ℓ¹ esatta, è il Thm 6.3.1 dimostrato sperimentalmente) e **10** (path
following in θ, la formulazione che il corso insegna per esattamente questo problema).

Il **12** merita comunque una menzione anche senza implementarlo, perché il materiale
sperimentale esiste già ed è insolitamente concreto.

---

### Riproducibilità

**Un comando genera tutti i numeri del report:**

```bash
python3 viz/make_results.py            # ~40 s
# -> viz/out/results.json   tutti i numeri, strutturati
# -> viz/out/results.md     gli stessi, in tabelle pronte
```

I numeri del report **non vanno copiati a mano** dal terminale: appena si ritocca un
parametro divergono dal codice in silenzio — è già successo qui con
`guides/snippets/nlp_structure.py`, rimasto a una taratura ormai superata.
`make_results.py` calcola ogni valore dagli **stessi moduli** usati dagli strumenti
interattivi, e il file porta con sé la provenienza: commit git, se l'albero era sporco,
profilo, versioni di CasADi e numpy.

Le misure sono raggruppate per **classe**, perché la classe decide se vanno rifatte:

| classe | cosa contiene | va rifatta? |
|---|---|---|
| **1** — proprietà della formulazione | ordine dell'integratore, AD contro differenze finite, sparsità dell'NLP, legge della penalità esatta | **No.** Non dipendono da nessuna run: si calcolano una volta |
| **2** — proprietà dell'istanza | KKT, active set, cono critico, moltiplicatori, soglia di biforcazione | **Come profilo**, non come numero singolo: variano ciclo per ciclo dentro la stessa missione |
| **3** — prestazione in anello chiuso | errore di predizione, θ contro tempo, costo del vincolo terminale | **Sì**, e qui servono davvero più missioni e più mondi |

Opzioni utili: `--quick` (meno punti, per provare), `--only classe1 classe2`,
`--bag viz/bags/<altra_run>` per rigenerare la sola classe 3 su una missione diversa.



Gli snippet di supporto stanno in `guides/snippets/`. `nlp_structure.py` riproduce i numeri
della §0 e accetta una lista di orizzonti per la tabella della §1.5:

```bash
python3 guides/snippets/nlp_structure.py            # N deployato
python3 guides/snippets/nlp_structure.py 10 15 25 50
```

Non richiede ROS: bastano `casadi`, `numpy`, `scipy`.

L'analisi su dati reali passa da [`viz/`](../viz/) — vedi
[`visualizzazione_ottimizzazione.md`](visualizzazione_ottimizzazione.md) per il flusso
registrazione → replay → pannelli.

---

## 8. Cosa resta fuori perimetro

| Componente | Perché è fuori |
|---|---|
| **Identificazione di τ_v, τ_w con SEM** (§3.1.2, §7.3.1) | *optimal system identification*: altra categoria di problema. Sarebbe implementabile — i bag ci sono — ma diluirebbe l'elaborato. Nota: il §0.2 mostra che i τ deployati sono degeneri, quindi un'identificazione seria cambierebbe il modello, non solo un numero |
| **Moving Horizon Estimation** (§3.2, §7.3.2) | *optimal observer*: idem. Oggi le velocità vengono da una media esponenziale sulle differenze di posa |
| **Bayesian Optimization TPE** ([`tuning/`](../tuning/), hyperopt) | ottimizzazione globale black-box: **non compare in nessun capitolo delle dispense**. Recuperabile solo come multi-obiettivo (§1.8). Dopo la §2.2 una parte del tuning diventa comunque superflua |
| **Surrogato GP, ARD Matérn-5/2** | machine learning |
| **PointCloud-GNNencoder (DGCNN)** | deep learning; unico aggancio §7.2.5 (§1.10), non vale svilupparlo |
| [`navigation_graph_node.py`](../src/a_star_mpc_planner/a_star_mpc_planner/navigation_graph_node.py) — grafo topologico, Dijkstra | ricerca su grafo discreta |
| [`persistent_map.py`](../src/a_star_mpc_planner/a_star_mpc_planner/persistent_map.py) | mappatura |
| A\* + griglia gaussiana | ricerca combinatoria — ma **una riga è dovuta**: è ciò che evita il MIP (§1.9) e genera z̄(θ) per la §1.1 |
| `g1_sim`, `robot_real_lidar`, `robot_real_goal_manager` | ingegneria robotica e infrastruttura: mezza pagina di contesto, zero contenuto teorico |
| Euristiche in `mpc_node.py`: escape BFS, limiti adattivi, staleness LiDAR, low-pass del setpoint, predizione ostacoli dinamici | patch di deployment — **da citare onestamente**: sono i sintomi dei buchi delle §1.2 e §2.2. Presentarle come *"cosa abbiamo fatto in assenza di ingredienti terminali e di vincoli soft"* è più forte che ometterle, e ora c'è la misura (§1.2) |

---

## 9. Indice inverso: dispense → voce di questa guida

| Sezione delle dispense | Voce |
|---|---|
| §2.1.3 discretizzazione, eq. (2.9)–(2.10) | **3.1**, 3.2, 3.3 |
| §3.3 FHOCP eq. (3.11) | **6.1**, 1.2 |
| §3.3 FHOCP parametrizzato eq. (3.13) | 8 (BO) |
| §4.1 esistenza, Thm 4.1.1 | 5.1 |
| §4.2 classificazione LP/QP/convesso/MIP | 5.1, **1.9** |
| §4.2.5 non liscio | 5.1 |
| §4.3 convessità, Thm 4.3.1–4.3.3 | **5.2** |
| §4.4.1 NOC/SOC | 2.1 |
| §4.4.2 line search, Armijo, Thm 4.4.5 | 5.3, 2.3 |
| §4.4.3 tassi di convergenza, condizionamento | 5.3, 2.3 |
| §4.4.4 Newton, Gauss-Newton, BFGS | 5.3, 4.1, **2.3** |
| §4.4.5 regolarità di x\*(ϑ), Thm 4.4.6, Fig. 4.17 | **5.4** |
| §5.1 ricorsione delle sensitività eq. (5.3) | 4.2 |
| §5.2 differenze finite | **4.1** |
| §5.3 AD, CasADi | **4.1** |
| §6.1.1 cono tangente, LICQ | **2.1** |
| §6.1.2 KKT eq. (6.8), Farkas | **2.1** |
| §6.1.3 cono critico, SOC-C-2 | **2.1** |
| §6.2.1 QP con uguaglianze, sistema KKT | 2.4 |
| §6.2.2 active-set vs interior-point, barriera log | **2.4**, 2.5 |
| §6.3.1 SQP Hessiana esatta | **2.3** |
| §6.3.2 Gauss-Newton vincolato, BFGS + Powell | **2.3** |
| §6.3.3 merit ℓ¹, Thm 6.3.1 penalità esatta | **2.2**, 2.3 |
| §6.3.4 Algoritmo 6.3.1 | **2.3** |
| §7.1.1 omotopia e warm starting eq. (7.2) | **1.7** |
| §7.2.2 single vs multiple shooting | **1.5** |
| §7.2.3 parametrizzazione, move blocking | **1.6** |
| §7.2.4 path following eq. (7.5) | **1.1** ⭐ |
| §7.2.5 MPC pratico, fattibilità ricorsiva | **1.2** ⭐, 1.3, 1.4, 1.10 |
| §7.4 multi-obiettivo, Pareto, Utopico | **1.8** |
| §3.1.2, §7.3.1 SEM | fuori perimetro (§8) |
| §3.2, §7.3.2 MHE | fuori perimetro (§8) |

---

## 10. Stato di avanzamento

Sezione di riferimento per il gruppo: **cosa è già implementato e misurato**, con la lettura
dei risultati, e **cosa resta da fare**. Aggiornata al 21 agosto 2026.

Le tre voci fatte sono state scelte perché la prima è verificabile in mezza giornata, la
seconda sblocca la terza, e la terza è il risultato più caratterizzante per l'esame.

### 10.1 Fatto — riepilogo

| # | Voce | §guida | Dove sta il codice | Come si rilancia |
|---|---|---|---|---|
| 13 | Integratore RK2 (punto medio) | 3.1 | `MPCConfig.integrator`, param ROS `mpc_integrator` | `python3 tests/test_integrators.py` |
| 2 | KKT, LICQ, complementarità, SOC-C-2 | 2.1 | `viz/kkt_analysis.py` | `python3 viz/kkt_analysis.py --bag viz/bags/industrial_plant_fix` |
| 11 | Penalità esatta ℓ¹ con slack | 2.2 | `MPCConfig.obstacle_mode`, `viz/exact_penalty.py` | `python3 viz/exact_penalty.py --scenario narrow_gap --d-safe 1.1` |
| 6 | Soglia di biforcazione di x\*(ϑ) | 5.4 | `viz/bifurcation_sweep.py` | `python3 viz/bifurcation_sweep.py --scenario centred_pillar` |
| 4 | AD contro differenze finite; Hessiana esatta contro L-BFGS | 4.1 | `MPCConfig.hessian`, `MPCResult.timings`, `viz/ad_vs_fd.py` | `python3 viz/ad_vs_fd.py` |
| 8 | Errore di predizione modello contro impianto | 1.4 | `viz/prediction_error.py` | `python3 viz/prediction_error.py viz/bags/industrial_plant_fix` |
| 10 | **Path following in θ** | 1.1 | `MPCConfig.path_mode`, param ROS `mpc_path_mode` | `python3 viz/formulation_compare.py
python3 viz/horizon_sweep.py
python3 viz/pareto_front.py --risoluzione 5
python3 viz/solver_compare.py
python3 viz/shooting_compare.py
python3 viz/control_horizon.py
python3 viz/robust_constraints.py` |
| 12 | **Vincolo terminale di equilibrio** | 1.2 | `MPCConfig.terminal_constraint`, param ROS `mpc_terminal_constraint` | `python3 viz/formulation_compare.py` |
| 14 | **Sweep dell'orizzonte N × dt** | 1.3 | `viz/horizon_sweep.py` | `python3 viz/horizon_sweep.py` |
| 15 | **Fronte di Pareto** | 1.8 | `viz/pareto_front.py` | `python3 viz/pareto_front.py --risoluzione 5` |
| 19 | **Interior point contro active set** | 2.4 | `viz/solver_compare.py` | `python3 viz/solver_compare.py` |
| — | **Single contro multiple shooting** | 1.5 | `MPCConfig.shooting` | `python3 viz/shooting_compare.py` |
| 17 | **Orizzonte di controllo N_c < N** | 1.6 | `MPCConfig.N_c` | `python3 viz/control_horizon.py` |
| — | **Vincoli robusti (constraint tightening)** | 1.2 | `MPCConfig.robust_backoff` | `python3 viz/robust_constraints.py` |
| — | Generatore dei risultati per il report | — | `viz/make_results.py` | `python3 viz/make_results.py` |
| — | Struttura e sparsità dell'NLP | 0, 1.5 | `guides/snippets/nlp_structure.py` | `python3 guides/snippets/nlp_structure.py 10 15 25 50` |
| — | Pannelli di visualizzazione + replay da bag | — | `viz/` | vedi [`visualizzazione_ottimizzazione.md`](visualizzazione_ottimizzazione.md) |

**Il comportamento deployato non è cambiato.** `obstacle_mode` resta `penalty` e il solve su un
ciclo reale dà `J*` **bit-identico** al valore pre-modifiche (8177.231314839336), con gli stessi
156 vincoli e nessuna variabile di slack allocata. L'unica modifica al comportamento è
`mpc_integrator: 'midpoint'` nel profilo G1, ed è reversibile cambiando una riga di YAML.
Anche `path_mode` e `terminal_constraint` sono spenti per default (`time` e `none`) ed
esposti come parametri ROS, quindi si accendono da YAML senza toccare il codice.

---

### 10.2 Integratore: ordine verificato, ma il guadagno in anello chiuso è quasi nullo

`tests/test_integrators.py` confronta i due schemi con la soluzione esatta in forma chiusa
(per ω ≠ 0 lo spostamento è un arco di circonferenza) e ne stima l'ordine con un fit log-log.

**Ordine 1.00 per Euler e 2.00 per il punto medio**, identico su tre regimi. Al `dt = 0.2`
deployato, su 3 s di predizione: Euler sbaglia **17.4 mm**, il punto medio **0.087 mm**. Un
fattore 200, al prezzo di una singola addizione.

Il test verifica anche che la dinamica costruita **dentro l'NLP** coincida con lo schema di
riferimento (scarto 0.000e+00): non basta che torni la teoria, deve integrare così `mpc_tracker`.

**Il risultato interessante è però negativo.** In anello chiuso, su quattro scenari, il costo
mediano migliora dello 0.0–1.1 % e distanza finale e clearance restano identiche. La ragione è
strutturale e va detta nel report: a orizzonte recedente si applica solo il **primo** ingresso e
A\* ripianifica, quindi l'errore di predizione accumulato sui 15 passi non si trasferisce alla
traiettoria percorsa. *(Il confronto è pulito: l'impianto simulato usa un aggiornamento di Euler
fisso indipendente da `cfg.integrator`; cambia solo il modello di predizione.)*

Il valore del punto medio sta quindi nella **fedeltà della predizione**, non nella traiettoria.
Diventa rilevante nella §1.3, se si allunga l'orizzonte, e nella §1.4, dove serve poter
attribuire l'errore di predizione al disallineamento di modello e non all'integratore.

---

### 10.3 KKT: il problema è molto più *vincolato* di quanto sembri

`viz/kkt_analysis.py` verifica su cicli reali estratti dalle bag ciò che le dispense enunciano
in astratto. Su nove cicli di `industrial_plant_fix`:

- **LICQ vale in ogni ciclo** (rango del Jacobiano attivo = numero di vincoli attivi) ⇒ i
  moltiplicatori KKT esistono e sono **unici**;
- **complementarità stretta in ogni ciclo** ⇒ il cono critico coincide con il nucleo del
  Jacobiano attivo, e SOC-C-2 si verifica **esattamente** anziché per difetto;
- **SOC-C-2 soddisfatta ovunque** (λ_min > 0 sull'Hessiana proiettata) ⇒ certificato di minimo
  locale stretto, che è il massimo ottenibile in un problema non convesso;
- stazionarietà: `‖∇_x L(x*, λ*)‖_∞ = 6.1e-11` contro `‖∇_x f(x*)‖_∞ = 1.6e+04`.

**Il risultato da raccontare** è la dimensione del cono critico, che **collassa da 45 a 1**
lungo la missione. A fine corsa ci sono 141 variabili e 140 vincoli attivi: resta un solo grado
di libertà. L'MPC passa da *guidato dal costo* a **guidato dai vincoli**, e nella fase finale
non sceglie più la traiettoria, la subisce.

La ripartizione dei vincoli attivi lo spiega: `|vy| ≤ vy_max` e `|ω| ≤ ω_max` sono saturi
**15 volte su 15** nei cicli finali, `vx ≥ 0` 14 su 15. È la conferma quantitativa che
`vy_max = 0.02` m/s non *limita* un grado di libertà: lo **rimuove**.

> **Nota per chi riusa il codice.** Due trappole di CasADi Opti, entrambe incontrate qui:
> 1. Opti canonizza ogni vincolo come `lbg ≤ g(x) ≤ ubg` e **assorbe nei limiti** il membro
>    destro quando è un parametro: `X[:,0] == p_x0` diventa `g = X[:,0]`, `lbg = ubg = p_x0`.
>    Valutare `|g|` su quelle righe restituisce **lo stato**, non il residuo. Il residuo è
>    sempre `g − lbg`.
> 2. Due righe distinte possono condividere la stessa espressione con limiti diversi
>    (`U[0,k] ≥ 0` e `U[0,k] ≤ vx_max` sono **entrambe** la riga `U[0,k]`). L'attivazione va
>    quindi decisa sulla **distanza dal limite**, non su `|g| ≈ 0`, altrimenti si contano
>    attivi tutti e due i lati dello stesso box.

---

### 10.4 Penalità ℓ¹: Thm 6.3.1 verificato, e due difetti di formulazione trovati per strada

`MPCTracker` accetta ora `obstacle_mode` ∈ {`penalty`, `l1`, `l2`}: nei due modi nuovi
l'ostacolo diventa un **vincolo vero** rilassato con slack, e lo slack è penalizzato in norma 1
oppure al quadrato.

Scenario `narrow_gap` con `d_safe = 1.1` m — scelto perché il vincolo **morda**: la clearance
naturale è 0.76 m. Moltiplicatore letto dal problema: **max|μ\*| = 8.905e+03**.

| ρ | max s\* (ℓ¹) | max s\* (ℓ²) |
|---|---|---|
| 1e+03 | 1.457e-01 | 2.606e-01 |
| **1e+04** | **0** | 1.297e-01 |
| 1e+05 | **0** | 3.975e-02 |
| 1e+06 | **0** | 4.375e-03 |
| 1e+08 | **0** | 4.424e-05 |

- **ℓ¹**: lo slack diventa **esattamente nullo** da ρ = 1e4 in su, e la soglia teorica
  max|μ\*| = 8.9e3 cade **esattamente** fra 1e3 (dove s\* = 0.146) e 1e4.
- **ℓ²**: il residuo decresce con pendenza log-log **−1.00 esatta** sulla coda e **non si
  annulla mai** — 4.4e-05 ancora a ρ = 1e8.

È il Thm 6.3.1 dimostrato sperimentalmente, in una tabella sola. La conseguenza pratica è che
**ρ smette di essere un iperparametro**: si legge dai moltiplicatori del problema.

**Due difetti di formulazione emersi implementandolo**, entrambi con contenuto teorico:

1. **Il vincolo non va imposto a k = 0.** Lì lo stato è fissato da `X[:,0] == x0`, quindi il
   vincolo non dipende da alcuna variabile decisionale: imporlo rende l'NLP inammissibile ogni
   volta che il robot si trova **già** entro `d_safe` da un ostacolo, cioè proprio quando
   servirebbe. Misurato sulla bag: con `d_safe = 0.9` m lo slack saturava a 0.1999, che è
   esattamente la violazione a k = 0 (clearance corrente 0.7017 m). Il vincolo parte ora da
   k = 1 e lo slack ha N colonne, non N+1.
2. **L'inammissibilità residua è colpa di U_Σ, non dei dati.** Anche partendo da k = 1, sul
   ciclo reale `d_safe = 0.75` m resta inammissibile: con `vx ≥ 0` e `vy_max = 0.02` m/s il
   robot in un passo può solo avanzare di 0.06 m lungo la propria direzione, e non può
   arretrare lateralmente dal muro di punti LiDAR. È l'interazione fra l'asimmetria
   dell'insieme ammissibile degli ingressi (§5.1) e un vincolo di stato, e collega questa
   scheda alla §1.2 (fattibilità ricorsiva).

**Costo da mettere in conto se si vuole deployare questo modo**: le disuguaglianze passano da
60 a oltre 300 e le iterazioni da ~20 a 60–240. Il cap deployato `mpc_max_iter = 40` non basta.
Questo **ribalta l'argomento della §2.4** (active-set contro interior-point) esattamente come
la scheda prevedeva, e va rimisurato prima e dopo.

---

### 10.5 Biforcazione: la soglia sta a W_obs ≈ 250, il deployato è 120

`viz/bifurcation_sweep.py` risolve lo **stesso** problema due volte, con warm start
spinto a sinistra e a destra, e misura la distanza fra le due soluzioni in ℝ¹⁴¹ al variare
di `W_obs_sigmoid`. È la Fig. 4.17 delle dispense costruita sui dati del progetto.

Scenario `centred_pillar`:

| W_obs | separazione | J\* sinistra | J\* destra | esito |
|---|---|---|---|---|
| 60 | 0.0000 | 4953.17 | 4953.17 | minimo unico |
| **120** *(deployato)* | **0.0000** | 9244.06 | 9244.06 | minimo unico |
| 200 | 0.0000 | 14251.90 | 14251.90 | minimo unico |
| 300 | 3.6888 | 20012.89 | 20122.31 | **biforca** |
| 600 | 3.9752 | 36018.82 | 36186.52 | **biforca** |
| 1400 | 4.4877 | 76236.16 | 76606.00 | **biforca** |

**La soglia sta fra 200 e 300**, e il valore in esercizio (120) è sotto. Ai pesi deployati
il minimizzatore è quindi unico e regolare in x₀ nel senso del Thm 4.4.6 — il che significa
che `_COST_SPIKE_FACTOR = 5.0` in `mpc_tracker.py` **sta proteggendo da un fenomeno che non
si verifica**.

Due dettagli che vale la pena riportare:

- quando biforca, i due `J*` **non coincidono** (20012.89 contro 20122.31): i due minimi non
  sono equivalenti, quindi il warm start non decide solo *dove* finisci ma *quanto paghi*;
- sul **ciclo reale** estratto dalla bag non biforca **nemmeno a W_obs = 1400**. La spiegazione
  è nella §10.3: quel ciclo ha un cono critico di dimensione 1. Con un solo grado di libertà
  residuo non c'è spazio geometrico per due bacini distinti — la saturazione dei vincoli
  uccide la biforcazione prima ancora che il peso possa crearla.

---

### 10.6 AD contro differenze finite: i tre numeri del corso, verificati

`viz/ad_vs_fd.py` misura il costo e l'accuratezza del gradiente sull'obiettivo
**effettivamente minimizzato** dall'MPC (n = 141 variabili), non su una funzione di prova.

| metodo | valutazioni di f | tempo | accuratezza |
|---|---|---|---|
| differenze in avanti | 142 = n+1 | 35.9 ms | 5.5·10⁻⁸ |
| differenze centrate | 282 = 2n | 71.4 ms | 1.3·10⁻¹⁰ |
| **AD in modo inverso** | **2.1** | **0.53 ms** | precisione macchina |

Tutte e tre le previsioni delle dispense sono confermate: l'AD costa **meno di 3 valutazioni
indipendentemente da n** (§5.3), e i passi ottimi misurati sono esattamente quelli teorici —
l'errore in avanti è minimo a `h = √eps = 1.49·10⁻⁸` e quello centrato a
`h = eps^(1/3) = 6.06·10⁻⁶`, con accuratezze ≈10⁻⁸ e ≈10⁻¹⁰ contro le ≈10⁻⁸ e ≈10⁻¹¹ previste.

**Perché conta per questo progetto**: a 8 Hz il budget per ciclo è 125 ms, e le sole
differenze centrate ne userebbero il **30 %** — per il solo gradiente, senza contare il
solve. E il rapporto peggiora **linearmente con n**, quindi con l'orizzonte: è l'argomento
quantitativo che rende discutibile allungare N (§1.3).

Sono ora esposti anche `MPCResult.status` (lo stato di uscita di IPOPT, che dice *perché* un
solve fallisce e non solo *che* è fallito) e `MPCResult.timings`, con i tempi per callback:
su un solve tipico, `hess_l` 7.96 ms, `grad_f` 4.01 ms, `f` 2.94 ms, `g` 2.29 ms,
`jac_g` 1.08 ms.

**Hessiana esatta contro L-BFGS** (`MPCConfig.hessian`), stesso problema:

| Hessiana | iterazioni | tempo | J\* |
|---|---|---|---|
| esatta (da AD) | **20** | 318.6 ms | 9244.061 |
| L-BFGS | **36** | 314.5 ms | 9244.061 |

Stesso minimo, ma L-BFGS impiega **l'80 % di iterazioni in più**. Il tempo totale si
pareggia perché ogni iterazione costa meno: è esattamente il compromesso Newton /
quasi-Newton del §4.4.4, misurato senza scrivere un solutore.

---

### 10.7 Errore di predizione: il modello sbaglia 7 volte più dell'integratore

`viz/prediction_error.py` confronta la traiettoria predetta (`/mpc/predicted_path`, salvata
al tempo t) con quella **effettivamente percorsa** (`/robot_pose` ai tempi t + k·dt), su
tutti i 775 cicli della bag. Nessun nuovo esperimento: i dati c'erano già.

| k | orizzonte [s] | errore mediano [m] | p95 [m] | max [m] |
|---|---|---|---|---|
| 0 | 0.00 | 0.0244 | 0.0613 | 0.1046 |
| 3 | 0.60 | 0.0445 | 0.1089 | 0.1783 |
| 6 | 1.20 | 0.0586 | 0.1872 | 0.2737 |
| 9 | 1.80 | 0.0816 | 0.2639 | 0.4522 |
| 15 | 3.00 | 0.1487 | 0.3519 | 0.5137 |

L'errore a **k = 0 non è errore di modello**: lì lo stato predetto *è* x₀, imposto come
vincolo di uguaglianza. I 2.4 cm misurano il disallineamento fra l'istante in cui la
predizione viene pubblicata e quello in cui la posa viene campionata — a ~0.08 m/s
corrispondono a **318 ms**, coerenti con il periodo di ciclo misurato (§3.2). Va sottratto
per isolare la divergenza vera.

**Divergenza al netto dell'offset: 0.042 m per secondo di predizione, 0.124 m a fine
orizzonte.**

Questo chiude il cerchio con la §10.2. La divergenza di modello è **7 volte** l'errore di
discretizzazione di Euler (1.74 cm) e **1428 volte** quello del punto medio (0.0087 cm).
È la spiegazione quantitativa del perché passare a RK2 migliori la predizione di 200× ma
l'anello chiuso di appena l'1 %: **l'integratore non è mai stato il termine dominante**.
Il termine dominante è che un uniciclo non descrive un G1 a 29 gradi di libertà che cammina.

---

### 10.8 Path following in θ: il robot va il 40 % più lontano, e un parametro sparisce

`MPCConfig.path_mode = 'theta'` implementa la eq. (7.5): l'ascissa curvilinea diventa una
variabile decisionale, con `θ(0) = 0`, `Δθ ≥ 0`, `θ(N) ≤ 1` e `+ α₃(1−θ)²` nel costo.

Il riferimento `z̄(θ)` è rappresentato da un **polinomio** i cui coefficienti sono parametri,
rifittati a ogni solve sul path A\*. Serve perché θ è una variabile decisionale e IPOPT è un
metodo di tipo Newton: una spezzata fra waypoint avrebbe derivata seconda discontinua a ogni
nodo. L'orientamento di riferimento viene dalla **tangente** del polinomio, quindi non è più
una scelta separata.

Misurato su 6 cicli in movimento della bag reale:

| grandezza | riferimento a tempo | ascissa θ |
|---|---|---|
| vx media comandata [m/s] | 0.2139 | **0.2989** |
| spostamento sull'orizzonte [m] | 0.6352 | **0.8914** |
| passi con vx a saturazione | — | 14.8 / 15 |
| iterazioni IPOPT | 11.8 | 20.0 |

**Il robot avanza il 40 % in più a parità di orizzonte.** La ragione è esattamente quella che
il corso anticipa: `v_ref = 0.2` m/s contro `vx_max = 0.3` lasciava inutilizzato il **33 %**
della velocità disponibile. In modo θ il solutore satura `vx_max` in 14.8 passi su 15 — decide
lui la velocità lungo il percorso, che è il punto della §7.2.4.

E il parametro **non viene sostituito** da un altro da tarare: θ è una variabile, non un
iperparametro. Il peso α₃ esiste, ma il suo effetto è debole proprio perché il robot è già
saturato in velocità (θ(N) passa da 0.099 a 0.136 mentre α₃ varia di 100×).

Il prezzo è **+70 % di iterazioni** (11.8 → 20.0): l'NLP ha N+1 variabili e N disuguaglianze
in più, ed è meno ben condizionato.

> **Una trappola trovata provando.** Sui cicli in cui il robot è **fermo** (fine missione,
> `vx ≥ 0` attivo) il confronto non dice nulla: θ avanzerebbe senza che il robot possa
> seguirlo, e con α₃ alto si osserva θ → 1 mentre il robot resta immobile — il riferimento si
> stacca dalla dinamica. `formulation_compare.py` seleziona quindi solo i cicli con
> `|v| > 0.15` m/s e path più lungo di 1.5 m.

---

### 10.9 Vincolo terminale: sul G1 non costa nulla, e il motivo è la degenerazione del lag

`MPCConfig.terminal_constraint = 'equilibrium'` impone `v(N) = 0` — esiste sempre una
traiettoria di frenata dentro l'orizzonte — rilassato con slack e penalizzato in **norma 1**,
come raccomandano le dispense. Per ρ > max|μ\*| lo slack va esattamente a zero (Thm 6.3.1),
quindi il vincolo è di fatto hard quando è soddisfacibile e cede solo quando non lo è.

Su 6 cicli reali lo **slack è sempre esattamente zero**: il robot riesce sempre a fermarsi.
Il costo del vincolo varia molto col ciclo, da **+0.2 % a +75 %** — è alto proprio dove `J*` è
piccolo, cioè dove il robot stava viaggiando bene e ora deve prevedere anche la frenata.

**Perché lo slack è sempre nullo**: con `τ = 0.001` contro `dt = 0.2` il lag è degenere
(§0 punto 2), cioè `v(k+1) = u(k)`, e il robot azzera la velocità in **un passo**. Il vincolo
terminale è quindi banalmente soddisfacibile.

Questo però non dimostrerebbe che il vincolo funziona — "slack sempre zero" non distingue un
vincolo facile da un vincolo non implementato. La controprova, con τ realistico:

| τ [s] | lag | v₀ [m/s] | slack | esito |
|---|---|---|---|---|
| 0.001 | 1.000000 | 0.3 | 0 | si ferma |
| 0.001 | 1.000000 | 1.2 | 0 | si ferma |
| 0.5 | 0.329680 | 0.3 | 3.0·10⁻² | **non si ferma** |
| 0.5 | 0.329680 | 1.2 | 9.4·10⁻³ | **non si ferma** |
| 2.0 | 0.095163 | 0.3 | 1.1·10⁻¹ | **non si ferma** |
| 2.0 | 0.095163 | 1.2 | 2.7·10⁻¹ | **non si ferma** |

Lo slack cresce con τ: più lento l'attuatore, meno l'orizzonte basta a fermarsi. È la lettura
fisica dell'insieme di fattibilità **F** del §7.2.5. Sul G1 deployato il vincolo terminale non
costa nulla in ammissibilità; su hardware con un τ vero sarebbe **il vincolo che decide la
velocità massima sicura**.

**Il legame con il latch della §10.3**: quel bug era la gestione ad hoc di una perdita di
fattibilità ricorsiva. Con il vincolo terminale attivo la fattibilità è garantita per
costruzione, e il fallback diventa una rete di sicurezza invece che il meccanismo principale.

---

### 10.10 Orizzonte: allungarlo oltre 5 s **peggiora**, e il deployato è dominato

`viz/horizon_sweep.py` valuta in anello chiuso una griglia **N × dt**, su scenari con
ostacoli, tenendo costante la durata della missione **in secondi** — non in passi: qui `dt` è
anche il periodo di controllo e il passo dell'impianto, quindi confrontare a parità di passi
darebbe ai `dt` piccoli una missione più corta.

I due parametri non sono intercambiabili:

```
orizzonte temporale   T = N·dt        quanto lontano l'MPC vede
numero di variabili   ~ N             quanto costa risolvere
errore di troncamento ~ dt^p          quanto è fedele la predizione
```

Griglia 5 × 4 su `narrow_gap` e `u_trap`, budget di ciclo 125 ms:

| fascia | tempo al goal | clearance minima |
|---|---|---|
| orizzonte **< 6 s** | 11.0 s | 0.228 m |
| orizzonte **≥ 6 s** | **22.7 s** | **0.165 m** |

**Allungare l'orizzonte oltre ~5 s peggiora entrambe le metriche.** È il risultato
controintuitivo dello sweep, e ha una spiegazione precisa: il riferimento si estende su un
percorso che A\* **ripianificherà comunque**, e l'MPC si impegna a inseguire un obiettivo
destinato a cambiare. I casi peggiori sono localizzati e severi — N=15/dt=0.4 scende a
**0.029 m** di clearance (quasi collisione), e N=25/dt=0.3 non raggiunge il goal su uno
scenario.

Sul costo, invece, comanda **N e non dt**: solo N=40/dt=0.1 sfora il budget (p95 141 ms).

**La configurazione deployata (N=15, dt=0.2) è dominata.** L'insieme non dominato su
(tempo al goal, clearance, p95):

| N | dt | T [s] | t al goal [s] | clearance [m] | p95 [ms] |
|---|---|---|---|---|---|
| 5 | 0.1 | 0.5 | 10.8 | 0.225 | 24.0 |
| 5 | 0.3 | 1.5 | 10.9 | 0.240 | 22.3 |
| **5** | **0.2** | **1.0** | **11.1** | **0.225** | **18.5** |
| 25 | 0.4 | 10.0 | 21.2 | 0.267 | 60.2 |
| 40 | 0.4 | 16.0 | 21.8 | 0.276 | 77.5 |

N=5/dt=0.2 dà **lo stesso tempo e la stessa clearance del deployato a metà del costo**
(18.5 ms contro 39.7 ms di p95).

> **Cautela prima di cambiare la configurazione.** Questi sono scenari sintetici con ostacoli
> statici e A\* che ripianifica spesso. Un orizzonte di 1 s (0.2–0.3 m) è brevissimo, e la sua
> tenuta dipende dal fatto che l'evitamento lo faccia A\*: con ostacoli dinamici, o con A\*
> più lento, il margine sparirebbe. Il risultato va **validato su missioni reali** prima di
> toccare il profilo deployato.

---

### 10.11 Fronte di Pareto: la taratura attuale è già non dominata

`viz/pareto_front.py` segue la procedura del §7.4 alla lettera: normalizzazione (I),
campionamento del simplesso **vertici inclusi** (II), punti non dominati, punto Utopico e
scelta come più vicino all'Utopico in norma 2 (III), più curva di Pareto (Fig. 7.9) e spider
chart (Fig. 7.10).

I tre obiettivi sono quelli che la eq. (7.5) introduce da sé: **accuratezza** (pesi Q),
**sforzo** (pesi R), **avanzamento** (peso su (1−θ)²). I pesi sono scalati per 3, così il
baricentro (⅓,⅓,⅓) riproduce esattamente la taratura di partenza.

> **Punto metodologico**: le *metriche* con cui si valutano le soluzioni usano pesi **fissi**,
> non quelli campionati. Altrimenti ogni punto del simplesso verrebbe giudicato con un metro
> diverso e il confronto non significherebbe nulla.

Su 21 punti del simplesso e due scenari, tutte le missioni riuscite:

| escursione relativa sul simplesso | |
|---|---|
| accuratezza | **30.5 %** |
| sforzo | 7.1 % |
| tempo | 3.7 % |

**Solo l'accuratezza risponde davvero ai pesi.** Tempo e sforzo sono quasi fissi, e la ragione
è nel §10.8: in modo θ il robot satura `vx_max` quasi sempre, quindi la durata è decisa dalla
cinematica e non dalla taratura. Il compromesso reale è quindi mite.

Il fronte risulta **convesso**, quindi la somma pesata lo recupera per intero — la cautela del
§7.4 non si applica qui, ma andava verificata invece che assunta.

**Il risultato utile è negativo**: il baricentro, cioè la taratura già in uso, è **non
dominato** e a pari distanza dall'Utopico rispetto al punto scelto dalla procedura
(0.559 contro 0.559). La taratura dei pesi non è il collo di bottiglia del sistema — a
differenza dell'orizzonte, che il §10.10 mostra essere mal scelto.

---

### 10.12 Interior point contro active set: il punto interno vince, ma non come dice la regola

`viz/solver_compare.py` sfrutta una proprietà che questo progetto ha e pochi altri hanno: la
formulazione degli ostacoli è **commutabile**, quindi lo stesso identico sistema può essere
messo nei due regimi opposti rispetto alla regola pratica del §6.2.2.

| regime | disuguaglianze | IPOPT (punto interno) | SQP + qpOASES (active set) | vince |
|---|---|---|---|---|
| `penalty` — ostacoli nel costo | **60** | 54 ms, 15 iter | 166 ms, 6 iter | punto interno (3.1×) |
| `l1` — ostacoli come vincoli | **300** | 129 ms, 21 iter | 6391 ms, 7 iter | punto interno (**49.7×**) |

**La regola del corso non si verifica**: l'active set non vince nemmeno con sole 60
disuguaglianze. Ma la *direzione* è confermata — il margine del punto interno passa da 3.1× a
49.7× quando le disuguaglianze quintuplicano. La soglia di pareggio, se esiste, sta sotto il
nostro regime più piccolo.

**Due cautele senza le quali il numero sarebbe fuorviante:**

1. **Il vantaggio vero dell'active set in MPC è il warm start fra solve consecutivi**, non il
   singolo solve a freddo: fra un ciclo e il successivo l'insieme attivo cambia di poche righe
   e qpOASES riparte dalla fattorizzazione precedente — è la *online active set strategy* per
   cui qpOASES è stato scritto. Qui si parte **a freddo apposta**, per non favorire nessuno
   dei due, e così si toglie all'active set proprio ciò che lo rende competitivo.
2. La regola del corso è pratica, e non contempla la **non convessità**. Il nostro problema è
   non convesso (§5.2).

**Un risultato che è venuto gratis**: con l'Hessiana esatta della lagrangiana, CasADi segnala
ripetutamente `Indefinite Hessian detected`. È atteso e istruttivo — se il problema non è
convesso, il QP interno dell'SQP può non esserlo, e un QP non convesso non ha soluzione unica.
È **esattamente la ragione** per cui il §6.3.2 raccomanda Gauss-Newton per l'SQP:
`H = 2·∇Fᵀ W ∇F` è semidefinita positiva per costruzione, e il QP torna convesso. La teoria
prevede il problema e ne fornisce il rimedio; qui si vede accadere.

> **Nota metodologica.** Alla prima esecuzione l'SQP "convergeva" in **1 iterazione** a
> `f = 21308` contro `17414` di IPOPT: si fermava prima, in un punto peggiore, e confrontare i
> tempi di due solve che finiscono in **minimi diversi** non significa nulla. Servono
> tolleranze a 1e-10 perché raggiunga davvero lo stesso minimo — e solo allora il confronto è
> un confronto. Lo strumento ora verifica l'uguaglianza dei minimi e rifiuta di dichiarare un
> vincitore quando non c'è.

---

### 10.13 Single contro multiple shooting: il vantaggio del multiple cresce con N

`MPCConfig.shooting` ∈ {`multiple`, `single`}. In single shooting X è **eliminata per
sostituzione ricorsiva** a partire da x₀: le variabili sono i soli ingressi e non c'è alcun
vincolo di dinamica. La mappa di transizione è scritta **una volta sola** e usata da entrambe
le parametrizzazioni — scriverla due volte significherebbe confrontare due modelli diversi
credendo di confrontare due parametrizzazioni.

Scenario `centred_pillar`, dt = 0.2:

| N | variabili M / S | vincoli M / S | densità jac M / S | tempo M / S [ms] | vince |
|---|---|---|---|---|---|
| 5 | 51 / 15 | 56 / 20 | 4.59 % / 6.67 % | 99 / 74 | single |
| 10 | 96 / 30 | 106 / 40 | 2.52 % / 3.33 % | 132 / 154 | multiple |
| 25 | 231 / 75 | 256 / 100 | 1.07 % / 1.33 % | 395 / 646 | multiple |
| 60 | 546 / 180 | 606 / 240 | 0.46 % / 0.56 % | 882 / 3162 | **multiple (2.9×)** |

**Il single è competitivo solo a N ≤ 5**; da lì in poi vince il multiple, e il margine cresce
monotonicamente fino a 2.9× a N = 60. È la previsione del §7.2.2 confermata: le variabili in
più sono ripagate dalla sparsità, e il vantaggio cresce con l'orizzonte.

Due cose che vanno dette:

- **A N = 25 le due parametrizzazioni convergono a minimi diversi** (13033.3 contro 13011.7,
  con il *single* migliore). Non è un errore: il problema non è convesso (§5.2), e due
  parametrizzazioni hanno cammini di ottimizzazione diversi, quindi possono cadere in bacini
  diversi. Rende il confronto dei tempi meno netto di quanto la tabella suggerisca.
- **Il tempo non è l'unico criterio.** Il single integra il modello in **anello aperto** su
  tutto l'orizzonte: l'errore si compone passo dopo passo e il problema si mal-condiziona con
  N e con l'instabilità del sistema. Qui il modello è cinematico e stabile, quindi il difetto
  non si manifesta — su un modello dinamico si manifesterebbe, ed è la ragione per cui il
  multiple shooting è lo standard in NMPC.

> Questo corregge un'affermazione non sostenuta che era finita in
> `viz/out/tex/metrics_body.tex`: che una formulazione single-shooting *"would be smaller and
> dense, and would lose exactly that property"*. Smaller e dense sono ora **misurati**; il
> resto era una previsione, e a N piccoli è addirittura falsa sul tempo.

---

### 10.14 Orizzonte di controllo: il degrado viene dalla predizione, non dai gradi di libertà

`MPCConfig.N_c` rende liberi i soli primi N_c ingressi; oltre, u resta costante all'ultimo
valore libero. Serve a rispondere a una domanda diagnostica che il §10.10 lasciava aperta:
là *"orizzonte"* era una cosa sola, perché N governa insieme **quanto lontano l'MPC guarda** e
**quanti gradi di libertà ha**.

| N | N_c | variabili | t al goal [s] | clearance [m] | p95 [ms] |
|---|---|---|---|---|---|
| 5 | 5 | 51 | 11.1 | 0.225 | 20.6 |
| 15 | 15 | 141 | 11.1 | 0.225 | 49.8 |
| **15** | **1** | **99** | **11.1** | **0.225** | **16.3** |
| 40 | 40 | 366 | 25.3 | 0.112 | 85.7 |
| 40 | 10 | 276 | 22.2 | 0.131 | 53.9 |
| 40 | 1 | 249 | 22.2 | 0.127 | 26.5 |

**Risposta: viene dalla predizione.** A N = 40, ridurre i gradi di libertà da 40 a 1 non
recupera il comportamento dell'orizzonte corto — resta a 22.2 s contro gli 11.1 s di N = 5.
Il degrado è dovuto al fatto che **il riferimento si estende su un percorso che A\*
ripianificherà comunque**, non a un eccesso di variabili.

La conseguenza è concreta: **non si può comprare a poco prezzo un orizzonte di predizione
lungo**. Se servisse per gli ingredienti terminali del §7.2.5, costerebbe prestazione, non
solo calcolo.

**Ma c'è un secondo risultato, e in pratica vale di più**: dove l'orizzonte è quello giusto,
`N_c` è calcolo gratis. A N = 15, `N_c = 1` dà tempo e clearance **identici** a `N_c = 15`
con p95 di 16.3 ms invece di 49.8 — **3.1× più veloce**. È il vantaggio della
parametrizzazione dell'ingresso (§7.2.3): disaccoppiare i gradi di libertà dall'orizzonte non
costa nulla in prestazione.

> **Una trappola risolta implementandolo.** Imporre i box su tutti gli N passi quando oltre
> N_c l'ingresso è *la stessa espressione ripetuta* genera righe **duplicate**, con gradienti
> identici: se attive violano LICQ (Def. 6.1.5) e rendono i moltiplicatori non unici — cioè
> romperebbero proprio l'analisi della §2.1. I box vanno imposti sulle sole colonne libere.
> Verificato: con la correzione LICQ e complementarità stretta valgono per ogni N_c.

---

### 10.15 Perché **non** usiamo il move blocking

Il move blocking (§7.2.3) tiene l'ingresso costante su blocchi di passi crescenti — per
esempio 1,1,1,2,2,4,4 — riducendo le variabili senza accorciare l'orizzonte. È la tecnica
standard per recuperare budget di calcolo, e qui **non la usiamo**. Le ragioni sono tre, tutte
misurate:

1. **L'orizzonte utile è già corto.** Il §10.10 mostra che N = 5 è Pareto-non-dominato e che
   oltre ~5 s la prestazione *peggiora*. Comprimere cinque variabili in tre blocchi non
   cambia nulla di misurabile.

2. **L'orizzonte di controllo fa già lo stesso lavoro, meglio.** `N_c` (§10.14) è un caso
   particolare degno di move blocking — un blocco libero seguito da uno lungo — e dà già
   **3.1×** di risparmio a prestazione identica. Un blocking più fine aggiungerebbe complessità
   per un guadagno residuo piccolo, e introdurrebbe la stessa trappola LICQ sui box duplicati,
   moltiplicata per il numero di blocchi.

3. **Il calcolo non è il vincolo attivo.** Con la configurazione deployata il p95 è ~50 ms
   contro un budget di 125 ms, e con `N_c = 1` scende a 16 ms. Ottimizzare una risorsa che
   avanza non è una priorità: il vincolo che morde è la **qualità della predizione** (§10.7,
   divergenza di 0.124 m), non il tempo di solve.

La tecnica resta però quella giusta **se** il progetto cambiasse premesse — orizzonte lungo
richiesto da un vincolo terminale rigoroso (voce 20), oppure un modello dinamico più costoso
per passo. Va citata come scelta consapevole, non come omissione.

---

### 10.16 Vincoli robusti: il tubo si misura, non si indovina

`MPCConfig.robust_backoff` irrigidisce il vincolo di ostacolo di un margine crescente:

```
‖p_k − o_j‖ ≥ d_safe + β(k) − s_jk
```

Il vincolo è imposto sulla traiettoria **predetta**, che il §10.7 misura divergere da quella
vera. β(k) è il margine che copre quella divergenza — e la particolarità di questo progetto è
che **non va indovinato**: `viz/robust_constraints.py` lo ricava dal quantile dell'errore di
predizione registrato nelle bag. È un tubo derivato dai dati, non da un'ipotesi sul disturbo.

β(k) al quantile 95 % su `industrial_plant_fix`:

| k | orizzonte [s] | β(k) [m] |
|---|---|---|
| 0 | 0.00 | 0.0000 |
| 3 | 0.60 | 0.0476 |
| 6 | 1.20 | 0.1260 |
| 9 | 1.80 | 0.2026 |
| 15 | 3.00 | **0.2907** |

Tre proprietà per costruzione: **β(0) = 0** esatto — a k = 0 lo stato è imposto come vincolo
di uguaglianza, quindi il vincolo non si irrigidisce dove non serve; β è **monotona**, come
l'incertezza; e il vincolo resta **soft**, quindi un tubo troppo largo fa crescere il costo ma
non rende l'NLP inammissibile.

*(All'offset a k = 0 — i 2.4 cm del §10.7 — viene sottratto: non è errore di modello ma
disallineamento temporale, e includerlo gonfierebbe il tubo di una costante che non ha nulla a
che vedere con l'incertezza.)*

**Effetto misurato sulla traiettoria predetta:**

| scenario | d_safe | clearance senza | con β | Δ | slack | esito |
|---|---|---|---|---|---|---|
| narrow_gap | 0.40 | 0.7500 | 0.7500 | +0.0000 | 0 / 0 | vincolo inattivo |
| **narrow_gap** | **0.70** | **0.7500** | **0.9907** | **+0.2406** | **0 / 0** | **efficace** |
| narrow_gap | 1.00 | 1.0000 | 1.2816 | +0.2816 | 0 / 0.009 | inammissibile |
| u_trap | 0.40–1.00 | 1.3417 | 1.3417 | +0.0000 | 0 / 0 | vincolo inattivo |

Il caso centrale è la dimostrazione: **+0.24 m di clearance predetta con slack esattamente
nullo** — il margine è *rispettato*, non violato e pagato. Gli altri due esiti sono
altrettanto informativi: quando `d_safe + β` sta sotto la distanza già tenuta il vincolo è
inattivo e non deve fare nulla; quando chiede più di quanto U_Σ consenta, la penalità ℓ¹ cede
invece di rendere l'NLP inammissibile — che è precisamente il motivo per cui era stata scelta
(§10.4).

**Sul run reale il tightening è inerte, e la ragione è istruttiva.** Sui cicli più stretti del
magazzino il robot sta a 0.615 m dagli ostacoli e non riesce ad allontanarsi: alzando `d_safe`
lo slack cresce (0.031 → 0.065 a 0.65 m) ma la traiettoria non si sposta. Il vincolo attivo non
è la soglia, è l'**insieme ammissibile degli ingressi**: con `vx ≥ 0` e `vy_max = 0.02` il robot
può solo avanzare lungo la propria direzione, non arretrare lateralmente. È lo stesso limite
misurato nel §10.4, e qui si manifesta di nuovo — il che è una conferma incrociata, non una
ripetizione.

> **Limite della misura, da dichiarare.** L'effetto **non è misurabile in anello chiuso** in
> questo simulatore: la clearance percorsa risulta *identica* per `obstacle_mode` `penalty` e
> `l1`, per ogni `d_safe` e ogni ρ. Il motivo è che `closed_loop` prende un setpoint a distanza
> di lookahead lungo la traiettoria predetta e lo insegue con un controllore proporzionale, il
> che cancella le differenze fini fra le soluzioni dell'MPC. Il constraint tightening garantisce
> il margine **nel piano**, ed è lì che va verificato — non è un limite della tecnica ma del
> banco di prova.

---

### 10.17 Da fare, in ordine di priorità

Il Capitolo 7 è ora coperto per intero, e con esso le voci del Capitolo 6 che gli servivano.
Quello che resta è di due tipi.

**Da scrivere, non da implementare**

| voce | §guida | Nota |
|---|---|---|
| Scrittura formale dell'FHOCP in notazione delle dispense | 6.1 | mezza pagina, ma va all'inizio del report: senza, nessuna delle altre sezioni ha un referente esplicito |
| Decidere se adottare N = 5 / dt = 0.2 | 1.3 | il §10.10 mostra che domina la configurazione attuale; richiede validazione su missioni reali, non modifiche al codice |

**Il pezzo grosso, se ci sarà tempo**

| # | Voce | §guida | Nota |
|---|---|---|---|
| 18 | SQP + Gauss-Newton scritti a mano contro IPOPT | 2.3 | il più caratterizzante; ha già i μ\* di riferimento (§10.3) e il confronto con qpOASES (§10.12). Il §10.12 mostra anche *perché* serve Gauss-Newton: con l'Hessiana esatta il QP interno è indefinito |
| 20 | Insieme terminale come sottolivello di Lyapunov (DARE) | 1.2 | versione rigorosa del §10.9. Attenzione: il §10.14 mostra che un orizzonte di predizione lungo — che il vincolo terminale rigoroso richiederebbe — **costa prestazione**, non solo calcolo |
| 16 | Omotopia su `obs_alpha` | 1.7, 2.5 | continuazione del §7.1.1; l'infrastruttura c'è (`decision_plane.py --set`) |

**Bloccato su un'altra voce**

| voce | §guida | Nota |
|---|---|---|
| Scelta del passo di campionamento argomentata dalla banda | 3.2 | vedi §10.18: sul profilo G1 il criterio spettrale è **vuoto**, non violato. Si sblocca identificando τ (§2.1) sotto fisica |

**Esplicitamente escluso**: il move blocking, per le ragioni misurate del §10.15.

---

### 10.18 Passo di campionamento: perché la trattazione spettrale oggi non si può fare

Voce **rinviata**, non dimenticata: nel report per ora la sezione non c'è, e questa scheda
dice cosa manca e perché.

L'argomento standard (e quello che usa il gruppo *Contactless Surface Following*, che segue
lo stesso corso) è spettrale: si prende il polo dominante dell'impianto, si sovracampiona di
circa un ordine di grandezza, e si ricava il passo:

```
banda dell'impianto   ω_c
regola pratica        ω_s ≈ 10 ω_c     ⇒   T_s = π / (5 ω_c)
```

Il nostro impianto sono tre lag del primo ordine disaccoppiati `v̇ = (u − v)/τ`, quindi il polo
è in `−1/τ` e `ω_c = 1/τ`. Applicandolo ai due profili:

| profilo | τ [s] | ω_c [rad/s] | T_s da regola | dt deployato | rapporto |
|---|---|---|---|---|---|
| τ non degenere (riferimento) | 0.12 | 8.33 (1.33 Hz) | **75 ms** | 100 ms | 1.3× |
| G1 deployato | 0.001 | 1000 (159 Hz) | **0.63 ms** | 200 ms | **318×** |

Con un τ dello stesso ordine di dt **l'argomento tiene** — regola e valore deployato coincidono
a meno di un fattore 1.3.
**Sul G1 è vuoto**, e non per un errore di taratura: il profilo mette τ → 0 *di proposito*,
perché l'impianto MuJoCo è cinematico e integra il comando direttamente. Il lag discreto vale
`1 − e^(−Δt/τ) = 1.000000` (§10.9), quindi `v(k+1) = u(k)` e **non esiste più un polo di
attuazione da campionare**. Scrivere quella sezione adesso significherebbe applicare un
criterio a cui abbiamo tolto l'ipotesi.

**Cosa la sblocca**: identificare τ con il metodo dell'errore di simulazione (§2.1) sotto
fisica, con la policy AMO nell'anello. Allora `ω_c` è una grandezza misurata e la sezione ha
un contenuto invece di una formula.

**Quello che nel frattempo si può già dire**, e che non dipende da τ: il passo è limitato da
due condizioni indipendenti, una da sopra e una da sotto.

1. **Superiore, geometrica.** Il termine di ostacolo è imposto solo negli `N+1` nodi; fra due
   nodi la traiettoria continua è libera. Quindi `Δs = v_max · Δt ≤ β · r_obs`. Con β = 0.25:
   `Δt ≤ 0.25 · 0.40 / 0.3 = 0.333 s`. Il deployato G1 è 0.20 s, cioè β = 0.15 — un passo
   copre 6 cm e 3.44° contro un raggio di 40 cm.
2. **Inferiore, computazionale.** `Δt ≥ t_solve,p95 / duty`. Misurato alla configurazione
   deployata: p95 di 59.2 ms su `narrow_gap` e 30.9 ms su `u_trap`, cioè il 29.6 % e il 15.5 %
   del periodo di 200 ms.

Forcella: **0.06 s ≲ Δt ≤ 0.33 s**, deployato 0.20 s.

> **Da riportare comunque in `sec:barrier`.** Il report solleva la preoccupazione giusta
> (*«nothing forbids the continuous inter-sample trajectory from clipping a thin obstacle»*)
> ma non la quantifica mai. Il rapporto `Δs / r_obs` è il numero che la quantifica, e sul
> profilo deployato vale **0.15**: fra due nodi il robot copre un sesto del raggio di
> sicurezza, quindi la barriera non può essere scavalcata per intero. È un margine da
> dichiarare, non da dare per scontato — basta che `v_max · Δt` si avvicini a `r_obs` perché
> fra due nodi la penalità smetta di fare qualcosa.

---

### 10.19 Come verificare che tutto giri ancora

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash

python3 tests/test_integrators.py        # ordine 1.00 / 2.00, scarto NLP 0.000e+00
python3 viz/test_fidelity.py             # costo replicato == opti.f, errore 0.000e+00
python3 guides/snippets/nlp_structure.py # struttura dell'NLP deployato
python3 viz/kkt_analysis.py      --bag viz/bags/industrial_plant_fix
python3 viz/exact_penalty.py     --scenario narrow_gap --d-safe 1.1
python3 viz/bifurcation_sweep.py --scenario centred_pillar
python3 viz/ad_vs_fd.py
python3 viz/prediction_error.py  viz/bags/industrial_plant_fix
python3 viz/formulation_compare.py
python3 viz/cost_field.py     --bag viz/bags/industrial_plant_fix
python3 viz/decision_plane.py --bag viz/bags/industrial_plant_fix
```

Nessuno di questi richiede il robot o il simulatore: girano tutti su bag registrate o su
scenari sintetici. Per registrare una nuova missione, vedi
[`visualizzazione_ottimizzazione.md`](visualizzazione_ottimizzazione.md).
