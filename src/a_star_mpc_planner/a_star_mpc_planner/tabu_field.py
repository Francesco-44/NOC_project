"""
tabu_field — memoria delle zone gia' visitate senza progredire.

PROBLEMA. AStarPlanner._local_goal sceglie il bersaglio proiettando il goal
globale sul bordo della finestra lungo il raggio robot->goal. E' una regola
memoryless e puramente direzionale: davanti a un ostacolo concavo il raggio
punta DENTRO la concavita', quindi il goal locale ci finisce dentro. Il robot
entra, A* si accorge che e' chiuso e lo fa uscire, ma appena uscito il raggio
ripunta dentro e si rientra. Il ciclo limite non e' un caso sfortunato: e'
deterministico, e nessuna quantita' di memoria sugli OSTACOLI lo evita, perche'
il problema non e' dimenticare il muro, e' come si sceglie il bersaglio.

IDEA. Si penalizzano le celle gia' percorse senza progredire e si rende la
scelta del goal locale un argmin invece di una proiezione:

    goal_locale = argmin_{c in bordo libero} [ ||c - goal|| + w * tabu(c) ]

Con tabu identicamente nullo si ricade nel comportamento attuale, quindi il
meccanismo e' disattivabile e non invalida le campagne gia' registrate.

PERCHE' NON UN COSTO DI CELLA. Il costo di cella di A* (1 + w*(p/soglia)^2)
influenza il PERCORSO verso un bersaglio fisso, non la SCELTA del bersaglio:
rendere caro il vicolo lo fa percorrere piu' caro, non evitare. La leva giusta
e' il bersaglio.

CANCELLAZIONE LEGATA AL PROGRESSO, NON AL TEMPO. Un tabu che decade nel tempo
rimette il ciclo, solo con periodo piu' lungo: appena svanisce, il raggio
ripunta nella trappola. Qui si tiene d_best (minima distanza dal goal mai
raggiunta con QUESTO goal) e si azzera solo quando d_best migliora oltre il
valore che aveva all'accensione, cioe' quando c'e' la PROVA di essere usciti.

Riferimenti: la famiglia Bug (Bug1/Bug2/TangentBug) per il caso "muro lungo
trasversale" — il wall-following qui non e' programmato, emerge dal fatto che
non si puo' tornare dove si e' gia' stati; e la tabu search per il resto.
"""

from __future__ import annotations

import math

import numpy as np


class TabuField:
    """
    Conteggio visite in frame mondo, con rilevamento di stallo.

    Parameters
    ----------
    reso            : lato cella [m]; conviene coincida con grid_reso di A*.
    visit_radius    : raggio [m] entro cui una visita incrementa le celle. Non
                      si marca la sola cella del robot: il campo dev'essere
                      abbastanza largo da coprire un corridoio, altrimenti
                      l'argmin trova sempre una cella di bordo libera a fianco
                      e il tabu non morde.
    revisit_trigger : quante visite della STESSA cella fanno scattare lo stallo
                      da oscillazione. E' la firma del rimpallo osservato: nel
                      vicolo il robot non e' fermo, quindi un rilevatore basato
                      sullo spostamento non lo vedrebbe.
    stuck_window_sec/stuck_disp_m : stallo da incastro — spostamento netto sotto
                      soglia nella finestra temporale. E' il caso del muro
                      trasversale, dove il robot si pianta davanti senza oscillare.
    improve_margin  : quanto deve migliorare d_best perche' si consideri uscita.
    """

    def __init__(
        self,
        reso: float = 0.20,
        visit_radius: float = 0.60,
        revisit_trigger: int = 3,
        stuck_window_sec: float = 10.0,
        stuck_disp_m: float = 0.5,
        improve_margin: float = 0.5,
    ):
        self.reso = float(reso)
        self.visit_radius = float(visit_radius)
        self.revisit_trigger = int(revisit_trigger)
        self.stuck_window_sec = float(stuck_window_sec)
        self.stuck_disp_m = float(stuck_disp_m)
        self.improve_margin = float(improve_margin)

        self._counts: dict[tuple[int, int], float] = {}
        self._last_cell: tuple[int, int] | None = None
        self._trail: list[tuple[float, float, float]] = []   # (t, x, y)

        self.active = False          # tabu acceso?
        self.d_best = math.inf       # minima distanza dal goal mai raggiunta
        self._d_best_at_arm = math.inf
        self._armed_reason = ""
        self.n_arms = 0              # quante volte si e' acceso (diagnostica)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Azzeramento totale. Da chiamare al CAMBIO DI GOAL: un tabu ereditato
        da una missione precedente penalizzerebbe celle che per il nuovo goal
        sono la strada giusta."""
        self._counts.clear()
        self._last_cell = None
        self._trail.clear()
        self.active = False
        self.d_best = math.inf
        self._d_best_at_arm = math.inf
        self._armed_reason = ""

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(round(x / self.reso)), int(round(y / self.reso)))

    # ------------------------------------------------------------------

    def update(self, pose_xy, goal_xy, now: float) -> bool:
        """Registra una posa e aggiorna lo stato. Ritorna True se il tabu e' attivo.

        Va chiamata a ogni ciclo di ripianificazione, PRIMA di pianificare.
        """
        x, y = float(pose_xy[0]), float(pose_xy[1])
        d = float(np.hypot(x - goal_xy[0], y - goal_xy[1]))

        # d_best e la prova di uscita
        if d < self.d_best:
            self.d_best = d
        if self.active and self.d_best < self._d_best_at_arm - self.improve_margin:
            # Uscita dimostrata: si spegne e si ricomincia con la lavagna pulita,
            # altrimenti la scia lasciata durante la fuga penalizzerebbe il
            # percorso buono appena trovato.
            self._counts.clear()
            self.active = False
            self._armed_reason = ""

        # conteggio visite, su un disco e non sulla singola cella
        cell = self._key(x, y)
        if cell != self._last_cell:
            self._last_cell = cell
            r = int(math.ceil(self.visit_radius / self.reso))
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if math.hypot(di, dj) * self.reso > self.visit_radius:
                        continue
                    k = (cell[0] + di, cell[1] + dj)
                    # peso a cono: massimo al centro, nullo al bordo del disco
                    w = 1.0 - math.hypot(di, dj) * self.reso / self.visit_radius
                    self._counts[k] = self._counts.get(k, 0.0) + w

        self._trail.append((now, x, y))
        cutoff = now - self.stuck_window_sec
        while self._trail and self._trail[0][0] < cutoff:
            self._trail.pop(0)

        if not self.active:
            reason = self._stuck_reason(now)
            if reason:
                self.active = True
                self._d_best_at_arm = self.d_best
                self._armed_reason = reason
                self.n_arms += 1
        return self.active

    def _stuck_reason(self, now: float) -> str:
        # (1) oscillazione: una cella rivisitata piu' volte
        if self._last_cell is not None:
            if self._counts.get(self._last_cell, 0.0) >= self.revisit_trigger:
                return "oscillazione"
        # (2) incastro: spostamento netto trascurabile nella finestra
        if self._trail and (now - self._trail[0][0]) >= self.stuck_window_sec:
            x0, y0 = self._trail[0][1], self._trail[0][2]
            x1, y1 = self._trail[-1][1], self._trail[-1][2]
            if math.hypot(x1 - x0, y1 - y0) < self.stuck_disp_m:
                return "incastro"
        return ""

    # ------------------------------------------------------------------

    def penalty(self, xs, ys):
        """Penalita' tabu nei punti dati (array). Zero se il tabu e' spento."""
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        out = np.zeros(xs.shape, dtype=float)
        if not self.active or not self._counts:
            return out
        ix = np.round(xs / self.reso).astype(int)
        iy = np.round(ys / self.reso).astype(int)
        for n in range(out.size):
            out.flat[n] = self._counts.get((int(ix.flat[n]), int(iy.flat[n])), 0.0)
        return out

    def panic_reset(self) -> None:
        """Ripiego di completezza: quando il tabu penalizza OGNI direzione il
        robot resterebbe fermo per sempre. Si azzera il conteggio tenendo il
        tabu acceso, cosi' riparte esplorando e — non avendo piu' la scia —
        puo' scegliere il lato opposto del muro. Senza questo, il caso 'muro
        lungo con il varco dal lato sbagliato' non termina."""
        self._counts.clear()
        self._last_cell = None

    def status(self) -> str:
        return (f"tabu {'ON' if self.active else 'off'}"
                f"{'(' + self._armed_reason + ')' if self._armed_reason else ''} "
                f"celle={len(self._counts)} d_best={self.d_best:.2f} arms={self.n_arms}")
