"""
mujoco_world — build the MuJoCo model for navigation simulation.

Loads the G1 MJCF (which already has a free `floating_base_joint` and resolves
its own meshes) and augments its spec with:
  - the industrial warehouse geometry, replicated from sim/worlds/industrial.sdf,
    placed in geom group 3 so the simulated LiDAR can ray-cast against ONLY the
    environment (the robot's own body is excluded → no self-mapping);
  - a floor plane + lights (group 0, not seen by the LiDAR);
  - a `mid360` site on torso_link matching the URDF LiDAR mount, used as the
    ray origin (so the published cloud is consistent with the mid360_link TF).

Everything stays in a single MjSpec, so mesh paths from the G1 file remain valid.

SDF→MuJoCo conversions:
  - box <size> is a FULL extent in SDF but a HALF extent in MuJoCo.
  - cylinder: SDF <length> is full; MuJoCo size is [radius, half_length].
"""

import math
import numpy as np
import mujoco

# Geom group cast by the LiDAR (environment only; robot/floor excluded).
LIDAR_GROUP = 3

# mid360 mount on torso_link (from 29dof.urdf: pos + 0.04 rad pitch)
MID360_POS = (0.0002835, 3e-05, 0.40618)
MID360_PITCH = 0.04014257279586953


def _yaw_quat(yaw):
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _box(x, y, z, sx, sy, sz, rgba, yaw=0.0, group=None):
    """SDF box (full sizes) → MuJoCo box dict (half sizes).

    `group` a None significa LIDAR_GROUP, cioe' un ostacolo vero. Passando 0 si
    ottiene una geometria di SOLA DECORAZIONE: visibile nel viewer ma esclusa
    dal ray-cast, quindi invisibile al pianificatore. Serve per i marcatori di
    goal, che altrimenti sarebbero ostacoli piazzati esattamente dove il robot
    deve arrivare.
    """
    return dict(shape="box", pos=[x, y, z], size=[sx / 2, sy / 2, sz / 2],
                rgba=rgba, yaw=yaw, group=group)


def _marker(x, y, rgba):
    """Disco piatto a terra che segna un punto notevole. Decorazione pura:
    group 0, e comunque sotto z_min del filtro (0.15 m)."""
    return dict(shape="cyl", pos=[x, y, 0.01], size=[0.35, 0.01],
                rgba=rgba, yaw=0.0, group=0)


def _cyl(x, y, z, radius, length, rgba, group=None):
    return dict(shape="cyl", pos=[x, y, z], size=[radius, length / 2], rgba=rgba,
                yaw=0.0, group=group)


# Material colours (approx. of industrial.sdf)
_WALL = [0.8, 0.8, 0.8, 1]
_COL = [0.4, 0.4, 0.5, 1]
_RACK = [0.6, 0.4, 0.2, 1]
_PALLET = [0.7, 0.55, 0.2, 1]
_BOXC = [0.3, 0.5, 0.7, 1]
_GREEN = [0.35, 0.6, 0.4, 1]
_CONV = [0.30, 0.30, 0.34, 1]
_ARM = [0.9, 0.45, 0.1, 1]
_DARK = [0.2, 0.2, 0.2, 1]
_SHELF = [0.55, 0.4, 0.25, 1]
_FORK = [0.85, 0.7, 0.1, 1]


def _seg(x1, y1, x2, y2, height=2.5, thick=0.25, rgba=None):
    """Muro fra due punti del piano: e' il modo naturale di disegnare
    geometrie non convesse (una U e' tre segmenti, un corridoio quattro).

    Restituisce un box centrato a meta' segmento, lungo quanto il segmento e
    ruotato per allinearvisi. Altezza e quota sono scelte perche' il muro cada
    dentro la fascia che il filtro LiDAR tiene (z_min 0.15, z_max 1.60 nel
    frame odom, vedi config/lidar_filter_g1.yaml): un ostacolo tutto sopra o
    tutto sotto quella fascia verrebbe scartato e il pianificatore non lo
    vedrebbe mai.
    """
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    return dict(shape="box",
                pos=[(x1 + x2) / 2.0, (y1 + y2) / 2.0, height / 2.0],
                size=[length / 2.0, thick / 2.0, height / 2.0],
                rgba=rgba or _WALL, yaw=math.atan2(dy, dx))


def _arena_box(minx, maxx, miny, maxy, height=3.0, thick=0.2):
    """Perimetro rettangolare qualsiasi.

    Serve perche' nei mondi con muro il bordo va messo VICINO alle spalle del
    robot — entro max_lidar_range — e LONTANO davanti, dove ci deve stare il
    goal. Con un'arena centrata non si puo' avere entrambe le cose.

    Il perche' e' di comportamento, non estetico: se il muro perimetrale dietro
    al robot e' fuori portata, quando il G1 costeggia il muro lungo e trova il
    lato sbarrato non ha modo di sapere che neppure aggirare il mondo da quella
    parte funziona, quindi continua a provarci. Vedendolo, invece, torna subito
    indietro verso l'altro capo — che e' la decisione giusta e quella che si
    vuole osservare.
    """
    return [_seg(minx, maxy, maxx, maxy, height, thick),
            _seg(minx, miny, maxx, miny, height, thick),
            _seg(minx, miny, minx, maxy, height, thick),
            _seg(maxx, miny, maxx, maxy, height, thick)]


def _arena(hx, hy, height=3.0, thick=0.2):
    """Quattro muri perimetrali: il robot non puo' uscire dal mondo e la
    scappatoia attorno a un ostacolo resta una scelta, non una fuga."""
    return [_seg(-hx, hy, hx, hy, height, thick),
            _seg(-hx, -hy, hx, -hy, height, thick),
            _seg(-hx, -hy, -hx, hy, height, thick),
            _seg(hx, -hy, hx, hy, height, thick)]


def warehouse_geoms():
    """Magazzino industriale, ridisposto come SEQUENZA DI VARCHI SFALSATI.

    La disposizione originale (replica di industrial.sdf) lasciava un corridoio
    centrale largo e sgombro: andando da (-12, 0) a (10, 0) il G1 incontrava
    pochissimi ostacoli, le metriche uscivano piatte e i pannelli del costo non
    mostravano quasi struttura. Non era un errore della scena — era una scena
    pensata per apparire realistica, non per sollecitare il pianificatore.

    Qui gli stessi blocchi (scaffalature alte, scaffali bassi, nastri, pallet,
    celle robotizzate, casse, muletto, colonne) sono ridisposti in SEI cancelli
    lungo la rotta, ciascuno con i varchi spostati rispetto al precedente. Il
    robot non puo' mai puntare dritto al goal: a ogni cancello deve scegliere
    un'apertura, e quella scelta lo disallinea per il cancello successivo.

    CRITERIO DIMENSIONALE. Ogni varco e' largo almeno 2.0 m: con grid_std 0.31 e
    obstacle_threshold 0.10 il raggio di blocco e' 0.397 m per lato, quindi
    restano >= 1.2 m di canale libero. Sono strettoie vere ma percorribili — lo
    scopo e' far LAVORARE il pianificatore, non farlo fallire; per il fallimento
    ci sono i mondi non convessi dedicati.

    Le concavita' qui sono piccole di proposito (la nicchia di casse a (0, -6) e
    l'angolo di nastri a (6, 6), entrambe ~2 m): danno varieta' al paesaggio di
    costo senza duplicare i test che l_corridor e horseshoe fanno meglio.
    """
    g = []
    # ── perimetro (invariato: 30 x 20 m) ───────────────────────────────
    g += [_box(0, 10, 1.5, 30, 0.2, 3, _WALL), _box(0, -10, 1.5, 30, 0.2, 3, _WALL),
          _box(15, 0, 1.5, 0.2, 20, 3, _WALL), _box(-15, 0, 1.5, 0.2, 20, 3, _WALL)]

    # ── cancello 1, x = -10: due scaffalature, varco CENTRALE ──────────
    # Scaffalatura = 6 m: ruotata di 90 gradi copre 6 m in y.
    g += [_box(-10, 5.5, 1.25, 6, 0.6, 2.5, _RACK, yaw=math.pi / 2),   # y 2.5..8.5
          _box(-10, -5.5, 1.25, 6, 0.6, 2.5, _RACK, yaw=math.pi / 2)]  # y -8.5..-2.5
    # varchi: y in [-2.5, 2.5] centrale, piu' due da 1.5 m ai bordi

    # ── cancello 2, x = -6: scaffali bassi, varco spostato a NORD ──────
    g += [_box(-6, -2.25, 1.3, 0.6, 6.5, 2.6, _SHELF),   # y -5.5..1.0
          _box(-6, -8.0, 1.3, 0.6, 4, 2.6, _SHELF)]      # y -10..-6
    g += [_box(-6, 4.0, 0.075, 1.2, 0.8, 0.15, _PALLET), # pallet basso: sotto z_min,
          _box(-6, 4.0, 0.55, 0.9, 0.7, 0.8, _BOXC)]     # la cassa sopra invece si vede
    # UNICO varco: y > 1.0, cioe' a NORD. Lo scaffale scavalca y=0 di proposito:
    # se ogni cancello lasciasse un'apertura vicino all'asse, il robot passerebbe
    # dritto e la scena tornerebbe piatta — e' cio' che succedeva alla prima
    # stesura (escursione laterale totale 2.4 m su 22 m di rotta).

    # ── cancello 3, x = -2: nastri trasportatori, varco a SUD ──────────
    # Nastro = 8 m, alto 0.7: dentro la fascia del filtro (0.15..1.60).
    g += [_box(-2, 3.0, 0.35, 8, 0.7, 0.7, _CONV, yaw=math.pi / 2),    # y -1..7
          _box(-2, -7.5, 0.35, 5, 0.7, 0.7, _CONV, yaw=math.pi / 2)]   # y -10..-5
    # UNICO varco: y in [-5, -1], cioe' a SUD. Insieme al cancello 2 (solo nord)
    # forza un'oscillazione vera da y>1 a y<-1 in 4 m di avanzamento.

    # ── cancello 4, x = 1..3: campo di celle robotizzate e colonne ─────
    # Ostacoli piccoli e sparsi: qui non c'e' un varco unico, c'e' da slalomare.
    for (ax, ay) in [(1.5, 3.0), (2.5, -0.5), (1.5, -4.0)]:
        g += [_cyl(ax, ay, 0.25, 0.30, 0.5, _DARK),
              _cyl(ax, ay, 1.05, 0.12, 1.1, _ARM),
              _box(ax, ay, 1.5, 0.9, 0.18, 0.18, _ARM, yaw=0.6)]
    for (cx, cy) in [(0.0, 6.5), (3.0, 6.0), (0.5, -7.5)]:
        g.append(_cyl(cx, cy, 1.5, 0.15, 3.0, _COL))

    # ── cancello 5, x = 6: scaffalature, varco CENTRALE stretto ────────
    g += [_box(6, 4.5, 1.25, 6, 0.6, 2.5, _RACK, yaw=math.pi / 2),     # y 1.5..7.5
          _box(6, -4.5, 1.25, 6, 0.6, 2.5, _RACK, yaw=math.pi / 2)]    # y -7.5..-1.5
    # varco: y in [-1.5, 1.5] = 3 m

    # ── cancello 6, x = 9: scaffali bassi + muletto, varco a NORD ──────
    g += [_box(9, -3.5, 1.3, 0.6, 5, 2.6, _SHELF),       # y -6..-1
          _box(9, -8.0, 1.3, 0.6, 4, 2.6, _SHELF)]       # y -10..-6
    g += [_box(9.0, 3.0, 0.35, 1.1, 0.7, 0.7, _FORK, yaw=1.2),
          _box(9.0, 3.0, 1.05, 0.6, 0.6, 0.7, _FORK, yaw=1.2),
          _box(9.0, 3.0, 1.0, 0.1, 0.6, 2.0, _DARK, yaw=1.2)]
    # varchi: y in [-1, 2.4] e y in [3.6, 10]

    # ── due piccole concavita', per dare struttura al paesaggio ────────
    # Nicchia di casse (~2 m di bocca), aperta verso ovest.
    g += [_box(0.0, -6.0, 0.3, 0.6, 2.4, 0.6, _BOXC),
          _box(-1.0, -5.0, 0.3, 2.0, 0.6, 0.6, _BOXC),
          _box(-1.0, -7.0, 0.3, 2.0, 0.6, 0.6, _GREEN)]
    # Angolo di nastri, aperto verso sud-ovest.
    g += [_box(6.5, 7.0, 0.35, 3, 0.7, 0.7, _CONV),
          _box(8.0, 8.0, 0.35, 0.7, 3, 0.7, _CONV)]

    # ── arredo sparso: pallet e casse fra un cancello e l'altro ────────
    for (px, py, yw) in [(-8.5, 0.5, 0.0), (-4.0, -2.0, 0.4), (4.0, 2.0, 0.9),
                         (7.5, -0.5, 0.2), (-11.0, -3.0, 0.0)]:
        g += [_box(px, py, 0.075, 1.2, 0.8, 0.15, _PALLET, yaw=yw),
              _box(px, py, 0.475, 0.6, 0.4, 0.5, _BOXC, yaw=yw)]
    for (kx, ky, yw) in [(-7.5, 6.5, 0.4), (3.5, -6.5, 0.8), (11.5, 1.5, 0.3),
                         (-3.0, 7.5, 0.2)]:
        g.append(_box(kx, ky, 0.45, 0.5, 0.5, 0.9, _GREEN, yaw=yw))

    g += [_marker(-12.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(10.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g



# ---------------------------------------------------------------------------
# Mondi con ostacoli NON CONVESSI
# ---------------------------------------------------------------------------
# Il magazzino industriale e' fatto di ostacoli convessi e sparsi: A* ne esce
# sempre con una deviazione locale, e il pianificatore non viene mai messo
# davanti a un minimo locale vero. Questi tre mondi servono a quello.
#
# La misura che li rende non banali e' la FINESTRA DI A*: grid_half_width = 6.0
# significa 12x12 m centrati sul robot, e il LiDAR arriva a 8 m. Un ostacolo
# concavo piu' PICCOLO della finestra non e' una trappola — A* ne vede subito
# il contorno completo e lo aggira. Diventa una trappola solo quando la via
# d'uscita cade FUORI dalla finestra, cioe' quando il pianificatore deve
# decidere sapendo di non vedere abbastanza. Le quote qui sotto sono scelte
# per stare da quel lato.
#
# Tutti i muri sono alti 2.5 m: la fascia utile del filtro e' z in [0.15, 1.60]
# in frame odom, quindi la geometria e' vista per intero a ogni quota di
# scansione, senza dipendere dal beccheggio del busto.


def long_wall_geoms():
    """Muro MOLTO lungo, varco a NORD, goal esattamente dietro.

    [FIX] La versione precedente era troppo corta: arrivando davanti al centro
    del muro il robot vedeva gia' l'estremita' nord a 7.6 m, dentro la portata
    del LiDAR, quindi non doveva scommettere su nulla. Non era il problema che
    si voleva porre.

    REGOLA DI PROGETTO. Nessuna estremita' del muro deve cadere entro
    max_lidar_range (8 m) NEMMENO quando il robot e' arrivato addosso al centro.
    Con lo spawn a y=0 e il muro a x=0, l'estremita' nord sta a y=+9: da (-0.5,0)
    dista 9.0 m, da (-6,0) dista 10.8 m. In nessun momento il robot sa se e dove
    ci sia un'apertura. Sa solo che al centro non si passa.

    Deve quindi SCOMMETTERE su un lato e costeggiare finche' non trova sbocco —
    che e' il comportamento degli algoritmi Bug, e l'unica strategia completa
    quando l'ostacolo eccede il campo visivo.

    A nord il varco c'e' (y da 9 a 12). A sud il muro arriva al perimetro: chi
    sceglie sud percorre 12 m, scopre che e' chiuso e deve tornare. Il mondo
    speculare long_wall_south sposta il varco dall'altra parte, cosi' la coppia
    distingue "ha ragionato" da "gira sempre dalla stessa parte".
    """
    # Perimetro ovest a x=-10, cioe' 4 m dietro lo spawn: DENTRO la portata del
    # LiDAR. Cosi' il robot che costeggia il muro verso il lato sbarrato sa gia'
    # che non puo' aggirare il mondo da quella parte, e torna indietro invece di
    # insistere. Con il bordo fuori portata quell'informazione non ce l'ha.
    g = _arena_box(-8.0, 10.0, -12.0, 12.0)
    g += [_seg(0.0, -12.0, 0.0, 9.0, 2.5, 0.30)]
    g += [_marker(-6.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(6.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g


def horseshoe_geoms():
    """Trappola a U (ferro di cavallo) aperta verso il robot, goal oltre il fondo.

    [FIX] La U era profonda 5 m: il fondo cadeva dentro max_lidar_range (8 m)
    gia' dall'imboccatura, quindi il robot lo vedeva PRIMA di entrare e passava
    di lato subito. Non era una trappola, era un ostacolo convesso visto per
    intero. Ora la profondita' e' 12 m (bracci da x=-2 a x=10): dall'imbocco il
    fondo e' a 12 m, fuori portata, e compare solo quando il robot e' a x >= 2,
    cioe' 4 m DENTRO. E' la regola generale di questi mondi — una concavita' e'
    una trappola solo se e' PIU' PROFONDA della portata del sensore.

    Larghezza 7 m (y da -3.5 a 3.5): entrando, i bracci si vedono (sono a 3.5 m)
    ma il fondo no, quindi la U e' indistinguibile da un corridoio largo aperto.
    """
    g = _arena(18.0, 10.0)
    g += [_seg(10.0, -3.5, 10.0, 3.5, 2.5, 0.30),     # fondo
          _seg(-2.0, 3.5, 10.0, 3.5, 2.5, 0.30),      # braccio nord
          _seg(-2.0, -3.5, 10.0, -3.5, 2.5, 0.30)]    # braccio sud
    g += [_marker(-7.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(14.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g


def dead_end_geoms():
    """Corridoio stretto e lungo, CHIUSO in fondo, con il goal appena oltre.

    Il corridoio e' largo 2.0 m e lungo 12 m, imboccatura a x=-2 e fondo chiuso
    a x=+10. Il goal sta a (13, 0), cioe' subito dietro il fondo: la direzione
    del corridoio punta al goal, ed e' questo che lo rende una trappola
    convincente invece che un ostacolo qualunque.

    [FIX] La lunghezza dev'essere MAGGIORE di max_lidar_range (8.0 m), non
    uguale. Con 8 m il fondo era visibile gia' dall'imboccatura e il robot non
    entrava mai davvero: si osservava un dondolio nord/sud all'imbocco (A* che
    cambia lato a ogni ripianificazione) invece del rimpallo dentro-fuori che
    si vede in MuJoCo. Con 12 m il fondo compare solo quando il robot e' a
    x >= 2, cioe' 4 m DENTRO: a quel punto e' gia' impegnato, ed e' il caso in
    cui la retromarcia (mpc_vx_min = -0.15) serve davvero.

    Larghezza 2.0 m scelta di proposito: con grid_std 0.31 e obstacle_threshold
    0.10 il raggio di blocco implicato e' 0.397 m, quindi restano 1.2 m di
    canale libero. Il corridoio e' percorribile, e il test riguarda il vicolo
    cieco, non la strettoia.
    """
    g = _arena(16.0, 8.0)
    g += [_seg(-2.0, 1.0, 10.0, 1.0, 2.5, 0.30),      # parete nord
          _seg(-2.0, -1.0, 10.0, -1.0, 2.5, 0.30),    # parete sud
          _seg(10.0, -1.0, 10.0, 1.0, 2.5, 0.30)]     # FONDO CHIUSO
    # Il giro largo e' libero: la soluzione esiste, e passa a nord o a sud.
    g += [_marker(-6.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(13.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g


# ---------------------------------------------------------------------------
# Secondo gruppo: trappole con occlusione, ambiguita' destra/sinistra, e
# controlli PERCORRIBILI (che servono a scoprire i falsi positivi: un
# meccanismo di fuga che fa deviare anche dove si passa e' un peggioramento).
# ---------------------------------------------------------------------------


def l_corridor_geoms():
    """Corridoio a L largo 3 m, imbuto d'ingresso, DUE chiusure annidate.

      imbuto       due diagonali da (-4,+-3.5) a (-2,+-1.5)
      braccio est  x da -2 a 6, y in [-1.5, 1.5]   — chiuso a x=6
      piede nord   x in [3, 6], y da 1.5 a 10      — chiuso a y=10, lungo 8.5 m

    [FIX] La L e' COMPATTA di proposito, per non dover allargare la finestra di
    A*. Il vincolo e' geometrico: dalla cima del piede il robot deve avere
    l'imbocco (-2, 0) DENTRO la finestra di pianificazione, altrimenti A* non
    trova via d'uscita, a_star_node non pubblica nulla e il robot resta FERMO in
    fondo al piede. Con la cima a (6.5, 12) servivano 11 m in y e la finestra da
    10 non bastava; ora la cima e' a (4.5, 10), cioe' 6.5 m in x e 10 m in y.

    Il piede resta lungo 8.5 m: dall'angolo (~4.5, 1.5) la chiusura a y=10 dista
    8.5 m, appena oltre max_lidar_range (8 m). La cecita' si conserva — va
    percorso per sapere che e' chiuso — senza costringere A* a pianificare
    lontano.

    Le sole DIAGONALI dell'imbuto restano: convogliano verso l'imbocco quando il
    robot arriva disallineato, senza pero' chiudere l'alternativa esterna. La
    scelta di entrare resta una DECISIONE del pianificatore e non un obbligo
    geometrico — che e' cio' che si vuole misurare.

    Sequenza attesa: entra nella L, trova chiuso il braccio est, si butta nel
    piede, trova chiuso anche quello, torna indietro ed esce. E' l'unico mondo
    con due trappole annidate: se il tabu serve da qualche parte serve qui,
    perche' uscendo dal piede per rientrare nel braccio d_best non migliora.
    """
    g = _arena_box(-11.0, 12.0, -8.0, 12.0)
    # Imbuto corto (punte a x=-4, non -5): le diagonali formano barriera con le
    # pareti del braccio, quindi per uscire dalla L verso sud o nord bisogna
    # aggirarne la punta. Con la punta a x=-5, dal fondo del braccio (x=5) quel
    # giro cadeva ESATTAMENTE sul bordo della finestra da 10 e A* non trovava
    # percorso. A x=-4 il margine c'e'.
    g += [_seg(-4.0, 3.5, -2.0, 1.5, 2.5, 0.30),    # imbuto nord
          _seg(-4.0, -3.5, -2.0, -1.5, 2.5, 0.30),  # imbuto sud
          _seg(-2.0, -1.5, 6.0, -1.5, 2.5, 0.30),   # parete sud del braccio est
          _seg(-2.0, 1.5, 3.0, 1.5, 2.5, 0.30),     # LATO ALTO accorciato: apre a x=3
          _seg(3.0, 1.5, 3.0, 10.0, 2.5, 0.30),     # parete ovest del piede
          _seg(6.0, -1.5, 6.0, 10.0, 2.5, 0.30),    # parete est: chiude verso il goal
          _seg(3.0, 10.0, 6.0, 10.0, 2.5, 0.30)]    # FONDO CHIUSO in cima al piede
    g += [_marker(-6.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(10.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g



def long_wall_south_geoms():
    """Speculare di long_wall: varco a SUD, muro fino al perimetro NORD.

    Stessa regola di lunghezza: l'estremita' sud sta a y=-9, cioe' 9.0 m dal
    robot arrivato al centro. Serve in coppia con long_wall — con un mondo solo
    non si distingue una scelta ragionata da una preferenza fissa per un lato.
    """
    g = _arena_box(-8.0, 10.0, -12.0, 12.0)
    g += [_seg(0.0, -9.0, 0.0, 12.0, 2.5, 0.30)]
    g += [_marker(-6.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(6.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g


def long_wall_false_north_geoms():
    """Muro lungo; il varco nord sembra sbarrato ma cela un passaggio laterale.

      muro principale  x=0, y in [-9, 9]      estremita' fuori portata
      varco nord       y in [9, 12]           3 m
      varco sud        y in [-12, -9]         3 m, passante senza sorprese
      aletta           x=2, y da 7.6 a 12     4.4 m: SUPERA il varco (3 m)

    La scelta iniziale fra nord e sud e' cieca: da (-0.5, 0) le estremita' del
    muro principale distano 9.0 m, oltre max_lidar_range.

    Chi sceglie NORD passa il varco e trova l'aletta a 1 m, PIU' LARGA del varco
    stesso: di fronte sembra un muro pieno, e l'errore da evitare e' concludere
    alla prima parete. La via c'e', ma va cercata di lato — l'aletta scende fino
    a y=7.6, sotto il bordo del varco, quindi si passa aggirandone la punta sud
    nel varco fra il muro principale (x=0) e l'aletta (x=1). Il canale libero
    netto e' ~0.9 m: stretto, percorribile.

    [NOTA GEOMETRICA — VERIFICATA] L'aletta NON puo' essere avvicinata oltre
    x=2. Provata a x=1: fra la faccia del muro principale (x=0.15) e quella
    dell'aletta (x=0.85) restavano 0.70 m di luce grezza, e con un raggio di
    blocco di 0.397 m PER LATO il canale libero diventa negativo — A* non trova
    piu' alcun percorso dal lato nord e il mondo e' risolvibile solo da sud, che
    e' il contrario di cio' che questo mondo deve provare. A x=2 la luce e'
    1.70 m, cioe' 0.91 m netti: stretta e percorribile. Per stringerla ancora va
    abbassato prima grid_std.
    """
    g = _arena_box(-8.0, 10.0, -12.0, 12.0)
    g += [_seg(0.0, -9.0, 0.0, 9.0, 2.5, 0.30),        # muro principale
          _seg(2.0, 7.6, 2.0, 12.0, 2.5, 0.30)]        # aletta: 4.4 m, piu' lunga del varco (3 m)
    g += [_marker(-6.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(6.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g



def open_corridor_geoms():
    """CONTROLLO: identico a dead_end ma con il fondo APERTO.

    Non e' una trappola: e' il test dei falsi positivi. Un meccanismo che fa
    uscire dal vicolo cieco ma fa deviare anche da un corridoio percorribile
    peggiora il sistema invece di migliorarlo. Qui l'esito atteso e' il
    passaggio diretto, ~19 m, senza inversioni di marcia.
    """
    g = _arena(16.0, 8.0)
    g += [_seg(-2.0, 1.0, 10.0, 1.0, 2.5, 0.30),
          _seg(-2.0, -1.0, 10.0, -1.0, 2.5, 0.30)]
    g += [_marker(-6.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(13.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g


def zigzag_geoms():
    """CONTROLLO percorribile: corridoio largo 6 m con tre setti sfalsati.

    Nessun setto chiude del tutto: restano varchi di 2 m alternati a nord e a
    sud, quindi si passa a zig-zag senza mai dover uscire. Verifica due cose
    insieme: che il pianificatore non scambi un setto per una chiusura, e che
    l'MPC regga tre cambi di direzione ravvicinati con l'orizzonte a 5.25 s.

    Varco di 2.0 m contro un raggio di blocco di 0.397 m: restano 1.2 m di
    canale libero, percorribile con margine.
    """
    g = _arena(16.0, 8.0)
    g += [_seg(-2.0, 3.0, 14.0, 3.0, 2.5, 0.30),    # parete nord
          _seg(-2.0, -3.0, 14.0, -3.0, 2.5, 0.30),  # parete sud
          _seg(2.0, -3.0, 2.0, 1.0, 2.5, 0.30),     # setto 1, varco a nord
          _seg(6.0, 3.0, 6.0, -1.0, 2.5, 0.30),     # setto 2, varco a sud
          _seg(10.0, -3.0, 10.0, 1.0, 2.5, 0.30)]   # setto 3, varco a nord
    g += [_marker(-6.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(15.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g


def door_room_geoms():
    """CONTROLLO al limite: parete piena da un capo all'altro, con UNA porta.

    La parete tocca entrambi i lati del perimetro: non esiste alcun modo di
    aggirarla, l'unico passaggio e' la porta al centro, larga 1.6 m. Con
    grid_std 0.31 e obstacle_threshold 0.10 il raggio di blocco implicato e'
    0.397 m, quindi restano 0.81 m di canale libero contro un ingombro del
    robot di ~0.35 m di raggio: si passa, con poco margine.

    L'ARCHITRAVE sopra la porta e' geometria REALE ma invisibile al
    pianificatore: sta fra z=1.80 e z=2.40, cioe' interamente sopra lo z_max del
    filtro LiDAR (1.60 m nel frame odom, vedi lidar_filter_g1.yaml). Serve
    perche' nel viewer si veda una porta e non una fessura fra due muri; il
    ray-cast la colpisce, il filtro la scarta, e la nuvola che arriva ad A* e'
    identica a quella di un'apertura passante. E' anche un promemoria utile: la
    fascia di quota del filtro decide cosa ESISTE per il pianificatore, e un
    ostacolo fuori da quella fascia semplicemente non c'e'.

    Questo mondo NON testa la fuga, testa la taratura della griglia: sugli
    scenari sintetici A* rifiutava varchi da 0.9 m preferendo giri di 7 m, quindi
    la soglia di passabilita' sta fra 0.9 e 1.6 m. Se il G1 gira intorno invece
    di passare, il parametro e' grid_std — e conta piu' di qualunque meccanismo
    di escape, perche' un pianificatore che non passa dalle porte non serve in
    un magazzino.
    """
    g = _arena_box(-10.0, 10.0, -8.0, 8.0)
    g += [_seg(0.0, -8.0, 0.0, -0.8, 2.5, 0.30),     # stipite sud
          _seg(0.0, 0.8, 0.0, 8.0, 2.5, 0.30)]       # stipite nord
    # architrave: sopra la fascia del filtro, quindi solo visiva
    g += [_box(0.0, 0.0, 2.1, 0.30, 1.6, 0.6, _WALL)]
    g += [_marker(-6.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(6.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g



WORLDS = {
    "industrial": dict(geoms=warehouse_geoms, spawn=(-12.0, 0.0, 0.0),
                       goal=(10.0, 0.0),
                       desc="magazzino industriale (ostacoli convessi, sparsi)"),
    "long_wall":  dict(geoms=long_wall_geoms, spawn=(-6.0, 0.0, 0.0),
                       goal=(6.0, 0.0), timeout=420.0,
                       desc="muro 21 m, estremita' fuori portata, varco a NORD"),
    "horseshoe":  dict(geoms=horseshoe_geoms, spawn=(-7.0, 0.0, 0.0),
                       goal=(14.0, 0.0),
                       desc="U profonda 12 m aperta verso il robot, goal oltre il fondo"),
    # Goal sull'ASSE del corridoio: senza le alette perpendicolari (rimosse su
    # richiesta) un goal spostato a nord rende l'aggiramento esterno piu' corto
    # dell'imbocco, e A* non entra proprio — verificato. Sull'asse, entrare e' la
    # rotta apparente piu' breve. Il piede viene esplorato lo stesso: una volta
    # dentro e trovato chiuso il braccio est, e' l'unica continuazione che resta.
    "l_corridor": dict(geoms=l_corridor_geoms, spawn=(-6.0, 0.0, 0.0),
                       goal=(10.0, 0.0), timeout=600.0,
                       desc="L larga 3 m, due chiusure annidate, goal verso il piede"),
    # Goal spostato a NORD di proposito: il varco vero e' a sud, quindi il lato
    # nord sembra il piu' breve e il robot ci si dirige. Serve a provare proprio
    # il caso "esplora, scopre che non si passa, torna indietro e prende l'altro
    # lato" — con il goal sull'asse la scelta iniziale sarebbe un lancio di
    # moneta e l'esito non ripetibile fra un run e l'altro.
    "long_wall_south": dict(geoms=long_wall_south_geoms, spawn=(-6.0, 0.0, 0.0),
                       goal=(6.0, 4.0), timeout=480.0,
                       desc="varco a SUD ma goal a nord: costringe a esplorare prima a sinistra"),
    # Goal a NORD: il varco vero e' a sud, quindi il lato nord sembra il piu'
    # breve e il robot ci va SUBITO. Serve a rendere ripetibile la prova — con il
    # goal sull'asse la scelta iniziale e' un lancio di moneta.
    "long_wall_false_north": dict(geoms=long_wall_false_north_geoms, spawn=(-6.0, 0.0, 0.0),
                       goal=(6.0, 4.0),
                       timeout=540.0,
                       desc="muro 18 m; il varco nord sembra chiuso ma cela una strettoia"),
    "open_corridor": dict(geoms=open_corridor_geoms, spawn=(-6.0, 0.0, 0.0),
                       goal=(13.0, 0.0),
                       desc="CONTROLLO: come dead_end ma APERTO in fondo"),
    "zigzag":     dict(geoms=zigzag_geoms, spawn=(-6.0, 0.0, 0.0),
                       goal=(15.0, 0.0),
                       desc="CONTROLLO: corridoio largo con 3 setti sfalsati, si passa"),
    "door_room":  dict(geoms=door_room_geoms, spawn=(-6.0, 0.0, 0.0),
                       goal=(6.0, 0.0),
                       desc="CONTROLLO al limite: parete con una sola porta da 1.6 m"),
    "dead_end":   dict(geoms=dead_end_geoms, spawn=(-6.0, 0.0, 0.0),
                       goal=(13.0, 0.0),
                       desc="corridoio 2.0x12 m chiuso in fondo, goal appena oltre"),
}


def world_names():
    return sorted(WORLDS)


def world_info(name):
    if name not in WORLDS:
        raise ValueError(
            f"mondo sconosciuto: {name!r}. Disponibili: {', '.join(world_names())}")
    return WORLDS[name]


def _add_person(wb, idx, color):
    """Add one ~1.7 m humanoid silhouette as a MOCAP body (legs+torso+head).

    A mocap body is moved every step by writing data.mocap_pos/mocap_quat (no
    joint, no dynamics) — the kinematic-teleport equivalent of the Gazebo
    set_pose people. Its geoms live in LIDAR_GROUP so the simulated Mid-360
    ray-casts against them (the MPC + tracker then see a moving obstacle), and
    are visual-only (contype=conaffinity=0), matching the warehouse geoms.
    Returns the body name (mocap id is resolved after compile)."""
    name = f"person_{idx}"
    body = wb.add_body(name=name, mocap=True, pos=[0.0, 0.0, -5.0])
    parts = [
        # (pos_z, type, size)  — sizes are MuJoCo half-extents
        (0.45, mujoco.mjtGeom.mjGEOM_BOX, [0.15, 0.15, 0.45]),       # legs
        (1.15, mujoco.mjtGeom.mjGEOM_BOX, [0.225, 0.14, 0.325]),     # torso
        (1.62, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.13, 0.13, 0.0]),   # head
    ]
    for pz, gtype, size in parts:
        gg = body.add_geom(type=gtype, size=size, pos=[0.0, 0.0, pz], rgba=color)
        gg.group = LIDAR_GROUP
        gg.contype = 0
        gg.conaffinity = 0
    return name


def build_model(g1_xml_path, n_people=0, people_colors=None, world="industrial"):
    """Build the combined MuJoCo model (G1 + world). Returns (model, info).

    `world` sceglie la geometria fra quelle di WORLDS (industrial, long_wall,
    horseshoe, dead_end). info["world"] e info["world_spawn"] riportano la
    scelta al chiamante, cosi' mujoco_sim puo' posizionare il robot dove quel
    mondo ha senso senza che l'utente debba ricordarsi le coordinate.

    If n_people > 0, that many mocap "person" bodies are added (parked below the
    floor at z=-5 until the sim places them); mujoco_sim teleports them along
    line/circle patterns. people_colors is an optional list of [r,g,b] used
    cyclically for the silhouettes."""
    spec = mujoco.MjSpec.from_file(g1_xml_path)
    wb = spec.worldbody

    # The G1 MJCF already provides an (infinite) `floor` plane with a checker
    # `groundplane` material, plus a robot-sized statistic/extent. Reuse that
    # floor — do NOT add a second plane, two coplanar planes at z=0 z-fight into
    # a speckled mess. Instead:
    #   - enlarge stat.extent so the camera near/far clipping covers the whole
    #     warehouse (with the default extent=0.8 a top-down view clips it away);
    #   - drop the floor reflectance (mirror glare under bright light).
    spec.stat.extent = 18.0
    spec.stat.center = [0.0, 0.0, 1.0]
    try:
        spec.material('groundplane').reflectance = 0.0
    except Exception:
        pass

    # Even, glare-free lighting. The MuJoCo default light is a SPOT light, which
    # over the world origin creates a bright hotspot ("abbaglio"); and the G1's
    # headlight uses specular=0.9. Use a DIRECTIONAL light (parallel rays → no
    # hotspot) and kill all specular highlights (light + viewer headlight) so
    # light-coloured surfaces don't blow out to white when viewed from above.
    sun = wb.add_light(pos=[0, 0, 15], dir=[-0.3, -0.4, -1.0])
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.diffuse = [0.5, 0.5, 0.5]
    sun.specular = [0.0, 0.0, 0.0]
    sun.castshadow = 1
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    # World obstacles in LIDAR_GROUP
    winfo = world_info(world)
    for ge in winfo["geoms"]():
        if ge["shape"] == "box":
            gg = wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=ge["size"],
                             pos=ge["pos"], rgba=ge["rgba"], quat=_yaw_quat(ge["yaw"]))
        else:
            gg = wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=ge["size"],
                             pos=ge["pos"], rgba=ge["rgba"])
        _grp = ge.get("group")
        gg.group = LIDAR_GROUP if _grp is None else int(_grp)
        gg.contype = 0      # no collision (kinematic robot) — visual + ray only
        gg.conaffinity = 0

    # Dynamic people (mocap bodies, moved by mujoco_sim)
    default_colors = [[0.85, 0.15, 0.15], [0.95, 0.55, 0.1], [0.6, 0.2, 0.7]]
    colors = people_colors or default_colors
    person_names = []
    for i in range(int(n_people)):
        c = colors[i % len(colors)]
        person_names.append(_add_person(wb, i, [c[0], c[1], c[2], 1.0]))

    # LiDAR mount site on torso_link
    spec.body('torso_link').add_site(
        name='mid360', pos=list(MID360_POS),
        quat=[math.cos(MID360_PITCH / 2), 0.0, math.sin(MID360_PITCH / 2), 0.0])

    model = spec.compile()

    # actuated (non-free) joints → for /joint_states
    free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
    joint_names, qpos_adr = [], []
    for j in range(model.njnt):
        if j == free_jid:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None:
            continue
        joint_names.append(name)
        qpos_adr.append(int(model.jnt_qposadr[j]))

    # mocap indices for the people (data.mocap_pos is indexed by body_mocapid)
    person_mocap_ids = []
    for nm in person_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        person_mocap_ids.append(int(model.body_mocapid[bid]))

    info = dict(
        site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'mid360'),
        free_qpos_adr=int(model.jnt_qposadr[free_jid]),
        joint_names=joint_names,
        joint_qpos_adr=qpos_adr,
        lidar_group=LIDAR_GROUP,
        person_mocap_ids=person_mocap_ids,
        world=world,
        world_spawn=tuple(winfo["spawn"]),
        world_goal=tuple(winfo["goal"]),
        world_desc=winfo["desc"],
    )
    return model, info
