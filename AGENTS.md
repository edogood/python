# AGENTS.md — Regole operative per generatore notebook (Codex/Agent)

## Obiettivo
Generare notebook didattici che siano anche una documentazione consultabile:
- 1 libreria = 1 “pacchetto” notebook (idealmente 1 notebook; consentito split in più notebook SOLO se necessario per dimensione).
- Copertura 100% della *public API surface* definita in INVENTORY_AND_COVERAGE.md.
- Per ogni unità API (funzione, classe, metodo, proprietà significativa) produrre una scheda completa come in METHOD_SPEC.md.

## Principi non negoziabili
1) Accuratezza > completezza apparente
   - Se una cosa non è verificabile tramite introspezione o docstring, segnala l’incertezza e aggiungi una nota di validazione.
2) Copertura reale
   - Ogni elemento dell’inventario deve avere una sezione dedicata. Niente “liste riassuntive” al posto delle schede.
3) Chiarezza chirurgica
   - Spiegazioni in linguaggio semplice, poi dettaglio tecnico, poi edge case.
4) Riproducibilità
   - Esempi deterministici (seed fisso) e mini-test eseguibili.
5) Zero copia-incolla esteso da documentazione esterna
   - Usa docstring e firma come base; parafrasa. Non riportare blocchi lunghi di testo da fonti esterne.

## Workflow obbligatorio (alta-level)
A) Costruisci INVENTARIO (script/introspezione) e salvalo (anche come JSON/CSV se previsto dal tuo progetto).
B) Genera notebook seguendo NOTEBOOK_SPEC.md.
C) Per ogni item in inventario, genera una scheda METHOD_SPEC.md.
D) Esegui QUALITY_GATES.md + COVERAGE audit.
E) Se fallisce: itera finché passa.

## Regole su cosa includere
- Include: elementi pubblici (no underscore iniziale) della API definita, incluse classi e i loro metodi pubblici.
- Escludi: API private, simboli deprecati (a meno che siano ancora pubblici e documentati: in tal caso includi ma marca come deprecated).
- Se la libreria genera classi/metodi automaticamente (SDK generati), applica la strategia “Surface Contract” da INVENTORY_AND_COVERAGE.md (es. top-level clients + metodi chiamabili dall’utente).

## Regole di stile
- Notebook leggibile come manuale: indice, sezioni coerenti, anchor stabili.
- Ogni scheda:
  - Nome completo (fully-qualified) + signature
  - Spiegazione semplice → meccanismo → edge case → esempio → mini-test
- Evita fluff e ripetizioni.

## Output atteso
- Notebook(i) in `notebooks/<library_name>/...`
- Un file di audit della copertura (formato a scelta, ma deve indicare 0 missing).
- Nessun notebook “vuoto” o con placeholder.
