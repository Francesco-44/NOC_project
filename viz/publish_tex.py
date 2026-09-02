#!/usr/bin/env python3
"""
Pubblica i .tex generati nel repo del report, senza clonarlo.

Perche' esiste: i tre file di viz/out/tex/ vanno riportati nel repo del report a
ogni rigenerazione (bag nuova, taratura ritoccata). Clonare 42 MB per copiare
56 KB non ha senso, e trascinarli nella UI di GitHub e' un gesto manuale che si
sbaglia in silenzio — file dimenticato, cartella sbagliata, tre commit separati.

Usa la Git Data API tramite `gh`, quindi:
  - nessun clone, nessun token da gestire (l'autenticazione e' quella di gh);
  - UN SOLO commit con tutti e tre i file, non uno per file;
  - se un file non e' cambiato non viene ricommittato.

Uso:
    python3 viz/publish_tex.py --dry-run          # mostra cosa farebbe
    python3 viz/publish_tex.py                    # pubblica
    python3 viz/publish_tex.py --dir Latex_noc/Metrics --branch main
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

FILES = ("metrics_macros.tex", "metrics_body.tex", "metrics_standalone.tex")

# Figure: nome LOCALE (che contiene bag e scenario) -> nome STABILE nel report.
#
# La mappatura non e' un vezzo. I file locali si chiamano
# errore_predizione_industrial_plant_fix.pdf: registrata una bag nuova il nome
# cambia, e un \includegraphics che punta al vecchio continua a compilare
# mostrando la figura sbagliata — nessun errore, figura stantia. Puntando a un
# nome stabile il report prende sempre l'ultima, e l'identita' della bag resta
# nella didascalia via \resBag, dove e' informazione e non dipendenza nascosta.
#
# Si pubblica il PDF, non il PNG: vettoriale, scala a qualunque dimensione.
from sync_report import FIGURES  # unica fonte della mappa figure


def gh(path: str, method: str = "GET", body: dict | None = None):
    """Chiamata all'API di GitHub attraverso gh, che ci mette l'autenticazione."""
    cmd = ["gh", "api", "-X", method, path]
    if body is not None:
        cmd += ["--input", "-"]
    try:
        out = subprocess.run(
            cmd, input=json.dumps(body) if body else None,
            capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit("gh non installato: https://cli.github.com")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"API {method} {path} fallita:\n{exc.stderr.strip()}")
    return json.loads(out) if out.strip() else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="Relo02/NOC_report")
    ap.add_argument("--branch", default="main")
    # La cartella nel repo del report e' Latex_noc/Metrics, con la M maiuscola,
    # ed e' quella che i .tex citano (\graphicspath{{Metrics/fig/}},
    # \restabdir{Metrics/tab}). Il default era minuscolo: su un filesystem che
    # distingue le maiuscole — e l'API di GitHub le distingue — pubblicare
    # senza --dir creava una SECONDA cartella accanto alla prima invece di
    # aggiornarla, e il report continuava a leggere quella vecchia senza che
    # niente segnalasse l'errore.
    ap.add_argument("--dir", default="Latex_noc/Metrics",
                    help="cartella di destinazione nel repo del report")
    ap.add_argument("--src", default=os.path.join(_HERE, "out", "tex"))
    ap.add_argument("--message", default=None)
    ap.add_argument("--report", default=None,
                    help="pubblica anche questo .tex, un livello sopra --dir")
    ap.add_argument("--no-figures", action="store_true",
                    help="pubblica solo i .tex, senza le figure")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # percorso remoto COMPLETO -> contenuto
    locali: dict[str, bytes] = {}
    D = args.dir.rstrip("/")
    for name in FILES:
        p = os.path.join(args.src, name)
        if not os.path.exists(p):
            raise SystemExit(f"manca {p}: eseguire prima  python3 viz/results_tex.py")
        locali[f"{D}/{name}"] = open(p, "rb").read()

    tab_dir = os.path.join(args.src, "tab")
    for p in sorted(glob.glob(os.path.join(tab_dir, "*.tex"))):
        locali[f"{D}/tab/" + os.path.basename(p)] = open(p, "rb").read()

    if args.report:
        if not os.path.exists(args.report):
            raise SystemExit(f"manca {args.report}")
        # un livello sopra --dir: Latex_noc/Metrics -> Latex_noc/Report_integrated.tex,
        # che e' l'unico posto dove i percorsi relativi del report risolvono
        # (\graphicspath{{Images/}}, \input{Configuration_files/...},
        # \bibliography{bibliography} sono tutti relativi a quella cartella).
        #
        # Report_metrics.tex non passa piu' di qui: e' superato da
        # Report_integrated.tex e vive dentro Metrics/, con i percorsi riscritti
        # per quel livello.
        parent = os.path.dirname(D) or D
        locali[f"{parent}/{os.path.basename(args.report)}"] = open(args.report, "rb").read()

    if not args.no_figures:
        figdir = os.path.dirname(os.path.abspath(args.src.rstrip("/")))
        for pattern, stabile in FIGURES:
            cand = sorted(glob.glob(os.path.join(figdir, pattern)),
                          key=os.path.getmtime, reverse=True)
            if not cand:
                print(f"  [figura assente: {pattern} — sezione senza immagine]",
                      file=sys.stderr)
                continue
            if len(cand) > 1:
                print(f"  [{pattern}: {len(cand)} candidati, prendo il piu' recente "
                      f"({os.path.basename(cand[0])})]", file=sys.stderr)
            locali[f"{D}/fig/" + stabile] = open(cand[0], "rb").read()

    # Provenienza: si LEGGE da results.json, non si ricalcola.
    #
    # Ricalcolarla qui era sbagliato e dava sempre "albero sporco": da quando
    # viz/out/ non e' piu' ignorato, generare il documento sporca l'albero per
    # costruzione, quindi al momento della pubblicazione git status e' sporco
    # SEMPRE. Ma il fatto che conta e' un altro: da quale stato del codice
    # nascono i NUMERI, ed e' registrato in results.json prima che la campagna
    # cominci. Il messaggio di commit deve dire la stessa cosa che dice il
    # documento, altrimenti le due provenienze si contraddicono.
    meta_path = os.path.join(os.path.dirname(args.src), "results.json")
    try:
        meta = json.load(open(meta_path))["meta"]
        sha_codice = meta["git_commit"][:7]
        branch_codice = meta["git_branch"]
        sporco = bool(meta["git_albero_sporco"])
        quando = meta.get("data_utc", "")[:10]
    except Exception as exc:
        raise SystemExit(
            f"provenienza illeggibile da {meta_path} ({exc}).\n"
            f"I .tex non vanno pubblicati senza: rigenerare con "
            f"python3 viz/make_results.py")
    msg = args.message or (
        f"metrics: misurate su {branch_codice}@{sha_codice} ({quando})"
        + (" [albero sporco: numeri non riproducibili da questo commit]"
           if sporco else ""))

    ref = gh(f"/repos/{args.repo}/git/ref/heads/{args.branch}")
    base_commit = ref["object"]["sha"]
    base_tree = gh(f"/repos/{args.repo}/git/commits/{base_commit}")["tree"]["sha"]

    # Cosa e' gia' lassu': i file identici non vanno ricommittati.
    def leggi_remoto(sub: str = "", base: str | None = None) -> None:
        radice = D if base is None else base
        percorso = f"{radice}/{sub}".rstrip("/")
        try:
            items = gh(f"/repos/{args.repo}/contents/{percorso}?ref={args.branch}")
        except SystemExit:
            return                              # la cartella non esiste ancora
        for item in items:
            rel = f"{radice}/{sub}{item['name']}"
            if item["type"] == "dir":
                leggi_remoto(f"{sub}{item['name']}/", base=radice)
            else:
                esistenti[rel] = item["sha"]

    esistenti: dict[str, str] = {}
    leggi_remoto()
    if args.report:
        leggi_remoto(base=os.path.dirname(D) or D)

    print(f"repo      {args.repo}  (branch {args.branch})")
    print(f"cartella  {args.dir}")
    print(f"messaggio {msg}\n")

    da_fare = []
    for name, data in sorted(locali.items()):
        # lo sha di git di un blob e' sha1("blob <len>\0" + contenuto)
        sha_locale = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        stato = ("invariato" if esistenti.get(name) == sha_locale
                 else ("aggiorna" if name in esistenti else "nuovo"))
        print(f"  {stato:<10} {name:<44} {len(data)/1024:7.1f} KB")
        if stato != "invariato":
            da_fare.append((name, data))

    if not da_fare:
        print("\nnulla da pubblicare: i file lassu' sono gia' identici.")
        return 0
    if args.dry_run:
        print(f"\n[dry-run] {len(da_fare)} file da pubblicare in un commit. "
              f"Rilanciare senza --dry-run.")
        return 0

    entries = []
    for name, data in da_fare:
        blob = gh(f"/repos/{args.repo}/git/blobs", "POST",
                  {"content": base64.b64encode(data).decode(), "encoding": "base64"})
        entries.append({"path": name, "mode": "100644",
                        "type": "blob", "sha": blob["sha"]})

    tree = gh(f"/repos/{args.repo}/git/trees", "POST",
              {"base_tree": base_tree, "tree": entries})
    commit = gh(f"/repos/{args.repo}/git/commits", "POST",
                {"message": msg, "tree": tree["sha"], "parents": [base_commit]})
    gh(f"/repos/{args.repo}/git/refs/heads/{args.branch}", "PATCH",
       {"sha": commit["sha"]})

    print(f"\npubblicato: {commit['sha'][:10]}")
    print(f"https://github.com/{args.repo}/tree/{args.branch}/{args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
