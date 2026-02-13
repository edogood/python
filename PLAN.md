# PLAN.md — Piano di generazione notebook “API-complete”

## Scopo
Produrre notebook stile-documentazione per le librerie elencate in LIBRARY_MAP.md.
Ogni notebook deve coprire la public API surface al 100% secondo INVENTORY_AND_COVERAGE.md.

## Deliverable per ogni libreria
1) `notebooks/<lib>/<lib>.ipynb` (o split controllato: `notebooks/<lib>/part_01_*.ipynb`, etc.)
2) `notebooks/<lib>/coverage_report.(json|md|csv)` con:
   - totale elementi inventario
   - elementi documentati
   - missing = 0
   - esclusi + motivazione (se presenti)
3) `notebooks/<lib>/execution_log.md` (o equivalente) con:
   - ambiente
   - versioni principali
   - esito esecuzione completa

## Politica sullo split
- Default: 1 notebook per libreria.
- Se il numero di elementi supera soglia (es. 400–800 schede), consenti split in più notebook:
  - Partizione per submodule / classi principali.
  - Mantieni un `INDEX.ipynb` che linka le parti.
  - La copertura deve restare totale (sum delle parti = inventario).

## Vincoli operativi
- Niente dipendenze di rete per gli esempi (o, se inevitabile, cell opzionali chiaramente marcate).
- Esempi: piccoli, deterministici, con output controllabile.
- Mini-test: assert essenziali, non fragili.

## Librerie target
Vedi LIBRARY_MAP.md (source of truth).
