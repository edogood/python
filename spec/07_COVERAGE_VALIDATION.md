# Coverage validation — notebook 99 e (opzionale) loop nel generatore

## Input
- inventory_pandas_api.json
- coverage_*.json (uno per notebook di contenuto)

## Normalizzazione
- Inventory qualname è la sorgente di verità.
- Coverage deve contenere qualname (stesso formato).

## Output atteso nel 99_Coverage_Report.ipynb
Per ciascun target:
- totale inventariati
- totale coperti
- lista mancanti (qualname)
- lista deprecati (qualname) se deprecated==true
- lista alias (qualname -> alias_of) se alias_of non null

In aggiunta:
- report aggregato: totale complessivo coperto vs inventariato
- guardrail: se coverage include elementi non in inventory, segnalarli come "unknown_coverage"

## Regola di successo
- SUCCESS se e solo se:
  - per ogni target, missing è vuoto
  - unknown_coverage è vuoto (o esplicitamente consentito con motivazione)

## Loop di rigenerazione (max 3 iterazioni)
Definizione:
- Iterazione 1: genera inventory + notebook
- Dopo esecuzione notebook (manuale o automatizzata):
  - leggere coverage_*.json
  - calcolare missing
- Iterazione 2-3:
  - rigenerare notebook includendo esplicitamente sezioni per i mancanti

Nota pratica:
- Se non si eseguono notebook automaticamente, il "loop" non può chiudersi da solo.
- In tal caso il generatore deve:
  - offrire modalità `--execute` (se implementata senza dipendenze extra)
  - oppure produrre istruzioni in 00_INDEX su come eseguire e rilanciare fino a zero missing.

## Regole per includere mancanti
Un membro è "incluso" se:
- esiste una sezione scheda dedicata
- viene eseguito almeno un mini-test o un esempio minimo
- viene aggiunto a covered_members

Se una sezione è SKIPPED, non conta come copertura.
