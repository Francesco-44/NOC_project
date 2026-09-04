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

    `group` left at None means LIDAR_GROUP, i.e. a real obstacle. Passing 0
    gives a DECORATION-ONLY geometry: visible in the viewer but excluded from
    the ray-cast, hence invisible to the planner. It is what the goal markers
    need, since otherwise they would be obstacles placed exactly where the
    robot is supposed to arrive.
    """
    return dict(shape="box", pos=[x, y, z], size=[sx / 2, sy / 2, sz / 2],
                rgba=rgba, yaw=yaw, group=group)


def _marker(x, y, rgba):
    """Flat disc on the ground marking a point of interest. Pure decoration:
    group 0, and below the filter's z_min (0.15 m) anyway."""
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
    """Wall between two points of the plane: the natural way to draw
    non-convex geometry (a U is three segments, a corridor four).

    Returns a box centred on the segment midpoint, as long as the segment and
    rotated to align with it. Height and elevation are chosen so that the wall
    falls inside the band the LiDAR filter keeps (z_min 0.15, z_max 1.60 in the
    odom frame, see config/lidar_filter_g1.yaml): an obstacle entirely above or
    entirely below that band would be discarded and the planner would never see
    it.
    """
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    return dict(shape="box",
                pos=[(x1 + x2) / 2.0, (y1 + y2) / 2.0, height / 2.0],
                size=[length / 2.0, thick / 2.0, height / 2.0],
                rgba=rgba or _WALL, yaw=math.atan2(dy, dx))


def _arena_box(minx, maxx, miny, maxy, height=3.0, thick=0.2):
    """Perimetro rettangolare qualsiasi.

    Needed because in the wall worlds the border must sit CLOSE behind the
    robot — within max_lidar_range — and FAR ahead, where the goal has to fit.
    A centred arena cannot have both.

    The reason is behavioural, not cosmetic: if the perimeter wall behind the
    robot is out of range, the G1 following the long wall towards the blocked
    side has no way of knowing that going around the world on that side does
    not work either, so it keeps trying. Seeing it, it turns back towards the
    other end straight away — which is the right decision, and the one the
    experiment is meant to observe.
    """
    return [_seg(minx, maxy, maxx, maxy, height, thick),
            _seg(minx, miny, maxx, miny, height, thick),
            _seg(minx, miny, minx, maxy, height, thick),
            _seg(maxx, miny, maxx, maxy, height, thick)]


def _arena(hx, hy, height=3.0, thick=0.2):
    """Four perimeter walls: the robot cannot leave the world, and going around
    an obstacle stays a choice rather than an escape."""
    return [_seg(-hx, hy, hx, hy, height, thick),
            _seg(-hx, -hy, hx, -hy, height, thick),
            _seg(-hx, -hy, -hx, hy, height, thick),
            _seg(hx, -hy, hx, hy, height, thick)]


def warehouse_geoms():
    """Industrial warehouse, laid out as a SEQUENCE OF STAGGERED GAPS.

    The original layout (a replica of industrial.sdf) left a wide, clear central
    corridor: going from (-12, 0) to (10, 0) the G1 met very few obstacles, the
    metrics came out flat and the cost panels showed almost no structure. That
    was not a mistake in the scene — it was a scene built to look realistic,
    not to stress the planner.

    Here the same blocks (tall racks, low shelves, conveyors, pallets, robot
    cells, crates, forklift, columns) are rearranged into SIX gates along the
    route, each with its opening offset from the previous one. The robot can
    never head straight for the goal: at every gate it has to pick an opening,
    and that choice misaligns it for the gate after.

    SIZING CRITERION. Every gap is at least 2.0 m wide: with grid_std 0.31 and
    obstacle_threshold 0.10 the blocking radius is 0.397 m per side, so at least
    1.2 m of free channel is left. They are genuine narrow passages, but
    passable — the point is to make the planner WORK, not to make it fail; for
    failure there are the dedicated non-convex worlds.

    The concavities here are deliberately small (the crate niche at (0, -6) and
    the conveyor corner at (6, 6), both ~2 m): they give the cost landscape some
    variety without duplicating the tests l_corridor and horseshoe do better.
    """
    g = []
    # ── perimeter (unchanged: 30 x 20 m) ──────────────────────
    g += [_box(0, 10, 1.5, 30, 0.2, 3, _WALL), _box(0, -10, 1.5, 30, 0.2, 3, _WALL),
          _box(15, 0, 1.5, 0.2, 20, 3, _WALL), _box(-15, 0, 1.5, 0.2, 20, 3, _WALL)]

    # ── cancello 1, x = -10: due scaffalature, varco CENTRALE ──────────
    # Scaffalatura = 6 m: ruotata di 90 gradi copre 6 m in y.
    g += [_box(-10, 5.5, 1.25, 6, 0.6, 2.5, _RACK, yaw=math.pi / 2),   # y 2.5..8.5
          _box(-10, -5.5, 1.25, 6, 0.6, 2.5, _RACK, yaw=math.pi / 2)]  # y -8.5..-2.5
    # gaps: y in [-2.5, 2.5] in the middle, plus two 1.5 m ones at the edges

    # ── gate 2, x = -6: low shelves, gap shifted NORTH ─────────────────
    g += [_box(-6, -2.25, 1.3, 0.6, 6.5, 2.6, _SHELF),   # y -5.5..1.0
          _box(-6, -8.0, 1.3, 0.6, 4, 2.6, _SHELF)]      # y -10..-6
    g += [_box(-6, 4.0, 0.075, 1.2, 0.8, 0.15, _PALLET), # low pallet: below z_min,
          _box(-6, 4.0, 0.55, 0.9, 0.7, 0.8, _BOXC)]     # the crate on top is seen
    # THE ONLY gap: y > 1.0, i.e. NORTH. The shelf crosses y=0 on purpose: if
    # every gate left an opening near the axis, the robot would go straight
    # through and the scene would be flat again — which is what happened in the
    # first draft (total lateral excursion 2.4 m over a 22 m route).

    # ── gate 3, x = -2: conveyors, gap SOUTH ───────────────────────────
    # Conveyor = 8 m, 0.7 high: inside the filter band (0.15..1.60).
    g += [_box(-2, 3.0, 0.35, 8, 0.7, 0.7, _CONV, yaw=math.pi / 2),    # y -1..7
          _box(-2, -7.5, 0.35, 5, 0.7, 0.7, _CONV, yaw=math.pi / 2)]   # y -10..-5
    # THE ONLY gap: y in [-5, -1], i.e. SOUTH. Together with gate 2 (north only)
    # it forces a real swing from y>1 to y<-1 within 4 m of forward travel.

    # ── gate 4, x = 1..3: field of robot cells and columns ─────────────
    # Small scattered obstacles: no single gap here, this one is a slalom.
    for (ax, ay) in [(1.5, 3.0), (2.5, -0.5), (1.5, -4.0)]:
        g += [_cyl(ax, ay, 0.25, 0.30, 0.5, _DARK),
              _cyl(ax, ay, 1.05, 0.12, 1.1, _ARM),
              _box(ax, ay, 1.5, 0.9, 0.18, 0.18, _ARM, yaw=0.6)]
    for (cx, cy) in [(0.0, 6.5), (3.0, 6.0), (0.5, -7.5)]:
        g.append(_cyl(cx, cy, 1.5, 0.15, 3.0, _COL))

    # ── gate 5, x = 6: racks, narrow CENTRAL gap ───────────────────────
    g += [_box(6, 4.5, 1.25, 6, 0.6, 2.5, _RACK, yaw=math.pi / 2),     # y 1.5..7.5
          _box(6, -4.5, 1.25, 6, 0.6, 2.5, _RACK, yaw=math.pi / 2)]    # y -7.5..-1.5
    # gap: y in [-1.5, 1.5] = 3 m

    # ── gate 6, x = 9: low shelves + forklift, gap NORTH ───────────────
    g += [_box(9, -3.5, 1.3, 0.6, 5, 2.6, _SHELF),       # y -6..-1
          _box(9, -8.0, 1.3, 0.6, 4, 2.6, _SHELF)]       # y -10..-6
    g += [_box(9.0, 3.0, 0.35, 1.1, 0.7, 0.7, _FORK, yaw=1.2),
          _box(9.0, 3.0, 1.05, 0.6, 0.6, 0.7, _FORK, yaw=1.2),
          _box(9.0, 3.0, 1.0, 0.1, 0.6, 2.0, _DARK, yaw=1.2)]
    # gaps: y in [-1, 2.4] and y in [3.6, 10]

    # ── two small concavities, to give the landscape some structure ────
    # Crate niche (~2 m mouth), open towards the west.
    g += [_box(0.0, -6.0, 0.3, 0.6, 2.4, 0.6, _BOXC),
          _box(-1.0, -5.0, 0.3, 2.0, 0.6, 0.6, _BOXC),
          _box(-1.0, -7.0, 0.3, 2.0, 0.6, 0.6, _GREEN)]
    # Conveyor corner, open towards the south-west.
    g += [_box(6.5, 7.0, 0.35, 3, 0.7, 0.7, _CONV),
          _box(8.0, 8.0, 0.35, 0.7, 3, 0.7, _CONV)]

    # ── scattered props: pallets and crates between one gate and the next ──
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
# Worlds with NON-CONVEX obstacles
# ---------------------------------------------------------------------------
# The industrial warehouse is made of convex, scattered obstacles: A* always
# gets out of them with a local detour, and the planner is never put in front
# of a genuine local minimum. These worlds are there for that.
#
# What makes them non-trivial is the A* WINDOW: grid_half_width = 6.0 means
# 12x12 m centred on the robot, while the LiDAR reaches 8 m. A concave obstacle
# SMALLER than the window is not a trap — A* sees its whole outline at once and
# goes around it. It becomes a trap only when the way out falls OUTSIDE the
# window, that is when the planner has to decide knowing it cannot see enough.
# The dimensions below are chosen to sit on that side.
#
#
# Every wall is 2.5 m tall: the filter's useful band is z in [0.15, 1.60] in the
# odom frame, so the geometry is seen in full at every scan elevation, without
# depending on how the torso pitches.


def long_wall_geoms():
    """VERY long wall, gap to the NORTH, goal exactly behind it.

    DESIGN NOTE. An earlier, shorter version was not enough: arriving in front
    of the middle of the wall the robot already saw the north end at 7.6 m,
    inside LiDAR range, so it had nothing to gamble on. That was not the
    problem this world is meant to pose.

    DESIGN RULE. No end of the wall may fall within max_lidar_range (8 m), NOT
    EVEN when the robot has come right up to the middle. With the spawn at y=0
    and the wall at x=0, the north end sits at y=+9: 9.0 m away from (-0.5, 0),
    10.8 m from (-6, 0). At no point does the robot know whether or where an
    opening is. All it knows is that the middle is blocked.

    So it has to BET on one side and follow the wall until it finds a way out —
    which is what Bug algorithms do, and the only complete strategy when the
    obstacle exceeds the field of view.

    North there is a gap (y from 9 to 12). South the wall reaches the perimeter:
    whoever picks south walks 12 m, finds it closed and has to come back. The
    mirrored world long_wall_south puts the gap on the other side, so the pair
    tells "it reasoned" apart from "it always turns the same way".
    """
    # West perimeter at x=-10, i.e. 4 m behind the spawn: INSIDE LiDAR range. A
    # robot following the wall towards the blocked side then already knows it
    # cannot get around the world that way, and turns back instead of insisting.
    # With the border out of range it does not have that information.
    g = _arena_box(-8.0, 10.0, -12.0, 12.0)
    g += [_seg(0.0, -12.0, 0.0, 9.0, 2.5, 0.30)]
    g += [_marker(-6.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(6.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g


def horseshoe_geoms():
    """U-shaped (horseshoe) trap opening towards the robot, goal past its back.

    DESIGN NOTE. A 5 m deep U was not enough: the back fell inside
    max_lidar_range (8 m) already from the mouth, so the robot saw it BEFORE
    entering and simply went around. That was not a trap, it was a convex
    obstacle seen in full. The depth is now 12 m (arms from x=-2 to x=10): from
    the mouth the back is 12 m away, out of range, and it only appears once the
    robot is at x >= 2, i.e. 4 m INSIDE. This is the general rule of these
    worlds — a concavity is a trap only if it is DEEPER than the sensor range.

    Width 7 m (y from -3.5 to 3.5): on entering, the arms are visible (they are
    3.5 m away) but the back is not, so the U is indistinguishable from a wide
    open corridor.
    """
    g = _arena(18.0, 10.0)
    g += [_seg(10.0, -3.5, 10.0, 3.5, 2.5, 0.30),     # back
          _seg(-2.0, 3.5, 10.0, 3.5, 2.5, 0.30),      # north arm
          _seg(-2.0, -3.5, 10.0, -3.5, 2.5, 0.30)]    # south arm
    g += [_marker(-7.0, 0.0, [0.2, 0.4, 0.9, 1]),
          _marker(14.0, 0.0, [0.2, 0.8, 0.3, 1])]
    return g


def dead_end_geoms():
    """Narrow, long corridor, CLOSED at the far end, with the goal just beyond.

    The corridor is 2.0 m wide and 12 m long, mouth at x=-2 and closed end at
    x=+10. The goal is at (13, 0), i.e. right behind that end: the corridor
    points at the goal, and that is what makes it a convincing trap rather than
    just another obstacle.

    DESIGN NOTE. The length must be GREATER than max_lidar_range (8.0 m), not
    equal to it. At 8 m the end was visible from the mouth and the robot never
    really went in: what one saw was a north/south wobble at the entrance (A*
    switching side at every replan) instead of the in-and-out bouncing MuJoCo
    shows. At 12 m the end only appears once the robot is at x >= 2, i.e. 4 m
    INSIDE: by then it is committed, and this is the case where reversing
    (mpc_vx_min = -0.15) genuinely matters.

    Width 2.0 m on purpose: with grid_std 0.31 and obstacle_threshold 0.10 the
    implied blocking radius is 0.397 m, so 1.2 m of free channel is left. The
    corridor is passable, and the test is about the dead end, not the squeeze.
    """
    g = _arena(16.0, 8.0)
    g += [_seg(-2.0, 1.0, 10.0, 1.0, 2.5, 0.30),      # north wall
          _seg(-2.0, -1.0, 10.0, -1.0, 2.5, 0.30),    # south wall
          _seg(10.0, -1.0, 10.0, 1.0, 2.5, 0.30)]     # CLOSED END
    # The long way round is free: a solution exists, north or south.
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
